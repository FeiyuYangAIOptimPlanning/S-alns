"""
timing_subproblem.py — 时序子问题求解

对齐论文 §4.1 的四个子小节:
  §4.1.1 Time sequence for a UGV trip  (Gtrip window)
  §4.1.2 Time sequence for a UGV route (Groute backward recursion)
  §4.1.3 Time sequence for a UAV trip  (Atrip window, arr_hub 视角)
  §4.1.4 Time sequence for a UAV route (Aroute forward recursion)

核心约定（IMPLEMENTATION_PLAN v2）:
- Atrip 主变量为 arr_hub（= 论文 φ_{dw}），dep_fc/return_fc 为派生
- 时间桥: cust[j].t_rel_H = arr_hub of Atrip carrying j（不再 + t_hD）
- cust[j].t_due_H = τ^O of the Gtrip carrying j
- cust[j].t_due_FC = t_due_H - t_hD[hub]  (派生, 用于可视化 / gurobi 对照)
- refresh_all_timing 为 4-Pass sweep (A: UAV forward, B: UGV backward, C: 回刷 Atrip.late, D: 汇总)
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import logging

from .structures import Atrip, Gtrip, Aroute, Groute, Solution, Customer

if TYPE_CHECKING:
    from .data_loader import DataContainer
    from .travel_times import TravelTimes

log = logging.getLogger(__name__)


# ===========================================================================
# §4.1.1 Gtrip window
# ===========================================================================
def compute_gtrip_window(
    gtrip: Gtrip,
    data: "DataContainer",
    tt: "TravelTimes",
) -> None:
    """计算 Gtrip 的 early/late/eta, 填回 gtrip 字段.

    对应论文 §4.1.1:
        τ^{O,early} = max over j in seq of cust[j].t_rel_H
        τ^{O,late}  = min over j* of ( t_j*^{Due} − Σ arcs(hub, j*) )
    feasible iff late >= early.
    """
    gtrip.eta = tt.gtrip_eta(gtrip.hub, gtrip.seq)
    gtrip.load = sum(data.customers[j].weight for j in gtrip.seq)

    if not gtrip.seq:
        gtrip.early, gtrip.late = 0.0, 0.0
        gtrip.feasible = True
        return

    # early = max t_rel_H
    earlys = []
    for j in gtrip.seq:
        t_rel_H = data.customers[j].t_rel_H
        # 初始化 Phase 1 尚无 UAV, t_rel_H 取乐观上界 (t_rel + t_hD)
        if t_rel_H is None:
            t_rel_H = data.customers[j].t_rel + tt.t_hD[gtrip.hub]
        earlys.append(t_rel_H)
    gtrip.early = max(earlys)

    # late: 累计 hub→j1→j2→... 的 travel, 每到一个 j* 取 t_j*^Due - cum
    cum = 0.0
    prev = None  # -1 表示 hub
    late_candidates = []
    for j in gtrip.seq:
        if prev is None:
            cum += tt.t_hj_K[gtrip.hub][j]
        else:
            cum += tt.t_jj_K[prev][j]
        late_candidates.append(data.customers[j].t_due - cum)
        prev = j
    gtrip.late = min(late_candidates)
    gtrip.feasible = gtrip.late >= gtrip.early - 1e-9


# ===========================================================================
# §4.1.2 Groute backward recursion
# ===========================================================================
def schedule_groute_backward(
    groute: Groute,
    data: "DataContainer",
    tt: "TravelTimes",
) -> None:
    """backward 递推 groute 内每个 Gtrip 的 dep (= τ_{kw}^O).

    论文 §4.1.2:
        trips[-1].dep = trips[-1].late
        trips[i].dep  = min(trips[i].late, trips[i+1].dep - t_K_set - trips[i].eta)
    若任何 trip 的 dep < its early, groute 标 infeasible.
    """
    if not groute.trips:
        groute.feasible = True
        return

    n = len(groute.trips)
    # 先保证所有 trip 的 window 是最新的
    for g in groute.trips:
        compute_gtrip_window(g, data, tt)

    groute.trips[-1].dep = groute.trips[-1].late
    groute.feasible = True

    for i in range(n - 2, -1, -1):
        w = groute.trips[i]
        w_next = groute.trips[i + 1]
        tentative = w_next.dep - data.t_K_set - w.eta
        w.dep = min(w.late, tentative)
        if w.dep < w.early - 1e-9:
            groute.feasible = False

    # 逐 trip 推 cust_arr 和 ret
    for g in groute.trips:
        if not g.seq:
            g.ret = g.dep
            continue
        t_now = g.dep + tt.t_hj_K[g.hub][g.seq[0]]
        g.cust_arr[g.seq[0]] = t_now
        # (15): τ^F <= t_due
        if t_now > data.customers[g.seq[0]].t_due + 1e-9:
            g.feasible = False
            groute.feasible = False
        for i in range(len(g.seq) - 1):
            t_now += tt.t_jj_K[g.seq[i]][g.seq[i + 1]]
            g.cust_arr[g.seq[i + 1]] = t_now
            if t_now > data.customers[g.seq[i + 1]].t_due + 1e-9:
                g.feasible = False
                groute.feasible = False
        g.ret = t_now + tt.t_jh_K[g.seq[-1]][g.hub]

    # 逐 trip 把 dep 传给每个客户的 t_due_H
    for g in groute.trips:
        for j in g.seq:
            data.customers[j].t_due_H = g.dep
            data.customers[j].t_due_FC = g.dep - tt.t_hD[g.hub]


# ===========================================================================
# §4.1.3 Atrip window (arr_hub 视角)
# ===========================================================================
def compute_atrip_window(
    atrip: Atrip,
    data: "DataContainer",
    tt: "TravelTimes",
) -> None:
    """计算 Atrip 的 early/late, 填回 atrip.

    论文 §4.1.3 (重述为 arr_hub 视角):
        φ_v^{O,early} = max over j of (t_j^{Rel} + t_hD[hub])
        φ_v^{O,late}  = min over j of cust[j].t_due_H
            (t_due_H 由当前 Groute 的 τ^O 给出; 若暂无则视为 +∞)
    """
    atrip.load = sum(data.customers[j].weight for j in atrip.pkgs)

    if not atrip.pkgs:
        atrip.early, atrip.late = 0.0, float("inf")
        atrip.feasible = True
        return

    atrip.early = max(data.customers[j].t_rel + tt.t_hD[atrip.hub] for j in atrip.pkgs)

    lates = []
    for j in atrip.pkgs:
        t_due_H = data.customers[j].t_due_H
        if t_due_H is None:
            # 尚无下游 UGV 调度, 用客户 due 作上界
            t_due_H = data.customers[j].t_due
        lates.append(t_due_H)
    atrip.late = min(lates)
    atrip.feasible = atrip.late >= atrip.early - 1e-9


# ===========================================================================
# §4.1.4 Aroute forward recursion
# ===========================================================================
def schedule_aroute_forward(
    aroute: Aroute,
    data: "DataContainer",
    tt: "TravelTimes",
) -> None:
    """forward 递推 aroute 内每个 Atrip 的 arr_hub.

    论文 §4.1.4 (arr_hub 视角):
        trips[0].arr_hub = trips[0].early
        trips[i].arr_hub = max(
            trips[i].early,
            trips[i-1].return_fc + t_D_set + t_hD[trips[i].hub]
        )
    若 arr_hub > late, aroute 标 infeasible.
    """
    if not aroute.trips:
        aroute.feasible = True
        return

    for a in aroute.trips:
        compute_atrip_window(a, data, tt)

    aroute.feasible = True
    first = aroute.trips[0]
    first.arr_hub = first.early
    first.dep_fc = first.arr_hub - tt.t_hD[first.hub]
    first.return_fc = first.arr_hub + tt.t_hD[first.hub]
    if first.arr_hub > first.late + 1e-9:
        aroute.feasible = False

    for i in range(1, len(aroute.trips)):
        v = aroute.trips[i]
        v_prev = aroute.trips[i - 1]
        t_hD_curr = tt.t_hD[v.hub]
        earliest_feasible = v_prev.return_fc + data.t_D_set + t_hD_curr
        v.arr_hub = max(v.early, earliest_feasible)
        v.dep_fc = v.arr_hub - t_hD_curr
        v.return_fc = v.arr_hub + t_hD_curr
        if v.arr_hub > v.late + 1e-9:
            aroute.feasible = False

    # 传给客户的 t_rel_H
    for a in aroute.trips:
        for j in a.pkgs:
            data.customers[j].t_rel_H = a.arr_hub


# ===========================================================================
# 4-Pass refresh (§3.6 of plan v2)
# ===========================================================================
def refresh_all_timing(
    sol: Solution,
    data: "DataContainer",
    tt: "TravelTimes",
) -> None:
    """4-Pass sweep, authoritative refresh after any structural change.

    - Pass 0: 清空所有 customer 派生时间字段 (避免共享状态污染)
    - Pass A: UAV forward -> arr_hub, t_rel_H
    - Pass B: UGV backward -> τ^O, t_due_H, cust_arr
    - Pass C: 回刷 Atrip.late & 校验 Aroute forward 仍成立
    - Pass D: 汇总 feasibility flags
    """
    # --- Pass 0: 清空共享状态 ---
    for c in data.customers:
        c.t_rel_H = None
        c.t_due_H = None
        c.t_due_FC = None

    # --- Pass A ---
    for aroute in sol.aroutes:
        schedule_aroute_forward(aroute, data, tt)

    # --- Pass B ---
    for groute in sol.groutes:
        schedule_groute_backward(groute, data, tt)

    # --- Pass C ---
    for aroute in sol.aroutes:
        # 重算 late (基于新 t_due_H), 但 arr_hub 不再改 (保持 earliest)
        for a in aroute.trips:
            compute_atrip_window(a, data, tt)
            if a.arr_hub > a.late + 1e-9:
                a.feasible = False
                aroute.feasible = False

    # --- Pass D ---
    sol.feasible = (
        all(r.feasible for r in sol.aroutes)
        and all(r.feasible for r in sol.groutes)
    )


# ===========================================================================
# 增量更新接口 (§3.7 of plan v2) - 双向级联
# ===========================================================================
def touch_atrip(
    atrip: Atrip,
    sol: Solution,
    data: "DataContainer",
    tt: "TravelTimes",
) -> None:
    """atrip 本身 / 其所在 aroute 结构改变后调用.

    正向级联: Aroute forward -> arr_hub -> t_rel_H -> 下游 Gtrip.early
    反向: Aroute late 依赖 t_due_H (由下游 Gtrip 给), 已由下一次 refresh 处理
    """
    aroute = sol.aroute_of_atrip(atrip)
    if aroute is None:
        return
    schedule_aroute_forward(aroute, data, tt)
    # 级联所属 Gtrip: 每个 j ∈ atrip.pkgs 所在 Gtrip
    affected_groutes: set[int] = set()
    for j in atrip.pkgs:
        g = sol.cust_to_gtrip.get(j)
        if g is not None:
            gr = sol.groute_of_gtrip(g)
            if gr is not None:
                affected_groutes.add(id(gr))
    for gr in sol.groutes:
        if id(gr) in affected_groutes:
            schedule_groute_backward(gr, data, tt)
    # 反向: 被影响 Gtrip 的 τ^O 回传给其 Atrip.late
    for aroute2 in sol.aroutes:
        for a in aroute2.trips:
            compute_atrip_window(a, data, tt)
            if a.arr_hub > a.late + 1e-9:
                a.feasible = False
                aroute2.feasible = False


def touch_gtrip(
    gtrip: Gtrip,
    sol: Solution,
    data: "DataContainer",
    tt: "TravelTimes",
) -> None:
    """gtrip 本身 / 其 Groute 结构改变后调用.

    正向: Groute backward -> τ^O -> t_due_H -> Atrip.late
    反向: Atrip.late 变化 -> 所在 Aroute 可能失 feasible
    """
    groute = sol.groute_of_gtrip(gtrip)
    if groute is None:
        return
    schedule_groute_backward(groute, data, tt)
    # 反向: 对其客户所在的 Atrip 重算 late, 校验 Aroute
    for aroute in sol.aroutes:
        need_reverify = False
        for a in aroute.trips:
            compute_atrip_window(a, data, tt)
            if a.arr_hub > a.late + 1e-9:
                a.feasible = False
                need_reverify = True
        if need_reverify:
            aroute.feasible = False


def touch_customer(
    j: int,
    sol: Solution,
    data: "DataContainer",
    tt: "TravelTimes",
) -> None:
    a = sol.cust_to_atrip.get(j)
    g = sol.cust_to_gtrip.get(j)
    if a is not None:
        touch_atrip(a, sol, data, tt)
    if g is not None:
        touch_gtrip(g, sol, data, tt)
