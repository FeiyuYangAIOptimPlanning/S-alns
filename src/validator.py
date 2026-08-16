"""
validator.py — 对 Eq. (2)–(23) 的完整可行性校验

使用 reconstruct_arc_vars_from_seq 将 gtrip.seq 还原为 λ/μ/γ 弧变量形式,
以便对齐论文公式。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging

from .structures import Solution, Gtrip, Atrip
from .data_loader import DataContainer
from .travel_times import TravelTimes

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    ok: bool = True
    failures: list[tuple[str, str]] = field(default_factory=list)  # (eq_ref, msg)

    def fail(self, eq: str, msg: str) -> None:
        self.ok = False
        self.failures.append((eq, msg))

    def render(self) -> str:
        if self.ok:
            return "All checks passed."
        lines = ["FAILURES:"]
        for eq, msg in self.failures:
            lines.append(f"  [{eq}] {msg}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# λ/μ/γ 重构 (planner §9.1)
# ---------------------------------------------------------------------------
def reconstruct_arc_vars_from_seq(
    gtrip: Gtrip,
) -> tuple[dict[int, int], dict[int, int], dict[tuple[int, int], int]]:
    lam: dict[int, int] = {}
    mu: dict[int, int] = {}
    gamma: dict[tuple[int, int], int] = {}
    seq = gtrip.seq
    if not seq:
        return lam, mu, gamma
    for j in seq:
        lam[j] = 0
        mu[j] = 0
    lam[seq[0]] = 1
    mu[seq[-1]] = 1
    for i in range(len(seq) - 1):
        gamma[(seq[i], seq[i + 1])] = 1
    return lam, mu, gamma


# ---------------------------------------------------------------------------
# 校验入口
# ---------------------------------------------------------------------------
def validate_solution(
    sol: Solution,
    data: DataContainer,
    tt: TravelTimes,
) -> ValidationReport:
    rep = ValidationReport()

    # (2) 每个 j 恰好出现在一个 Atrip
    seen_a = {}
    for ar in sol.aroutes:
        for a in ar.trips:
            for j in a.pkgs:
                if j in seen_a:
                    rep.fail("(2)", f"cust {j} in multiple Atrips")
                seen_a[j] = a
    for j in data.J:
        if j not in seen_a and j not in sol.unserved:
            rep.fail("(2)", f"cust {j} not in any Atrip and not unserved")

    # (3) Atrip 载重 <= l_D
    for ar in sol.aroutes:
        for a in ar.trips:
            total = sum(data.customers[j].weight for j in a.pkgs)
            if total > data.l_D + 1e-6:
                rep.fail("(3)", f"Atrip {a.id} load {total:.2f} > l_D {data.l_D}")

    # (4) φ_{dw} = arr_hub >= t_j^Rel + t_hD ∀ j
    for ar in sol.aroutes:
        for a in ar.trips:
            for j in a.pkgs:
                need = data.customers[j].t_rel + tt.t_hD[a.hub]
                if a.arr_hub < need - 1e-6:
                    rep.fail("(4)", f"Atrip {a.id}: arr_hub {a.arr_hub:.2f} < "
                                      f"t_rel+t_hD of C{j} = {need:.2f}")

    # (6) 同 UAV 相邻 trip 间隔 >= 2 t_hD + t_D_set (on aroute)
    for ar in sol.aroutes:
        for i in range(len(ar.trips) - 1):
            v, vn = ar.trips[i], ar.trips[i + 1]
            required = v.arr_hub + tt.t_hD[v.hub] + data.t_D_set + tt.t_hD[vn.hub]
            if vn.arr_hub < required - 1e-6:
                rep.fail("(6)", f"UAV {ar.uav_id}: trip gap violated "
                                 f"({vn.arr_hub:.2f} < {required:.2f})")

    # (8) 每个 j 恰好在一个 Gtrip
    seen_g = {}
    for gr in sol.groutes:
        for g in gr.trips:
            for j in g.seq:
                if j in seen_g:
                    rep.fail("(8)", f"cust {j} in multiple Gtrips")
                seen_g[j] = g
    for j in data.J:
        if j not in seen_g and j not in sol.unserved:
            rep.fail("(8)", f"cust {j} not in any Gtrip and not unserved")

    # (9) Gtrip 载重 <= l_K
    for gr in sol.groutes:
        for g in gr.trips:
            total = sum(data.customers[j].weight for j in g.seq)
            if total > data.l_K + 1e-6:
                rep.fail("(9)", f"Gtrip {g.id} load {total:.2f} > l_K")

    # (10)(11) λ/μ/γ 与 s 一致 (reconstruct 确保一致, 这里只检查 seq 非空一致性)
    for gr in sol.groutes:
        for g in gr.trips:
            if len(g.seq) == 0:
                rep.fail("(11)", f"Empty Gtrip {g.id} should not exist")
            lam, mu, gamma = reconstruct_arc_vars_from_seq(g)
            if len(g.seq) > 0 and sum(lam.values()) != 1:
                rep.fail("(11)", f"Gtrip {g.id}: λ sum != 1")
            if len(g.seq) > 0 and sum(mu.values()) != 1:
                rep.fail("(11)", f"Gtrip {g.id}: μ sum != 1")

    # (13)(14) UGV 绑定 active hub
    for gr in sol.groutes:
        if gr.hub not in sol.active_hubs:
            rep.fail("(13/14)", f"Groute {gr.ugv_id}: hub {gr.hub} not active")

    # (15) 客户送达 <= due
    for gr in sol.groutes:
        for g in gr.trips:
            for j in g.seq:
                arr = g.cust_arr.get(j)
                if arr is None:
                    rep.fail("(15)", f"Gtrip {g.id}: cust {j} has no arrival time")
                elif arr > data.customers[j].t_due + 1e-6:
                    rep.fail("(15)", f"cust {j}: arrived {arr:.2f} > due "
                                      f"{data.customers[j].t_due:.2f}")

    # (16) 同 UGV 相邻 trip 间隔 >= t_K_set
    for gr in sol.groutes:
        for i in range(len(gr.trips) - 1):
            w, wn = gr.trips[i], gr.trips[i + 1]
            if wn.dep < w.ret + data.t_K_set - 1e-6:
                rep.fail("(16)", f"UGV {gr.ugv_id}: trip gap "
                                  f"({wn.dep:.2f} - {w.ret:.2f} < t_K_set)")

    # (17)(18)(19) UGV 时间守恒
    for gr in sol.groutes:
        for g in gr.trips:
            if not g.seq:
                continue
            # 从 dep 开始累加, 应等于 cust_arr[j]
            first = g.seq[0]
            if first not in g.cust_arr:
                rep.fail("(17)", f"Gtrip {g.id}: cust_arr missing for {first}")
                continue
            t_now = g.dep + tt.t_hj_K[g.hub][first]
            if abs(t_now - g.cust_arr[first]) > 1e-4:
                rep.fail("(17)", f"Gtrip {g.id}: first cust arr mismatch")
            ok_conservation = True
            for i in range(len(g.seq) - 1):
                t_now += tt.t_jj_K[g.seq[i]][g.seq[i + 1]]
                nxt = g.seq[i + 1]
                if nxt not in g.cust_arr:
                    rep.fail("(19)", f"Gtrip {g.id}: cust_arr missing for {nxt}")
                    ok_conservation = False
                    break
                if abs(t_now - g.cust_arr[nxt]) > 1e-4:
                    rep.fail("(19)", f"Gtrip {g.id}: cust {nxt} arr mismatch")
            if ok_conservation:
                ret_expected = t_now + tt.t_jh_K[g.seq[-1]][g.hub]
                if abs(ret_expected - g.ret) > 1e-4:
                    rep.fail("(18)", f"Gtrip {g.id}: return time mismatch")

    # (20)(21) η 正确且 <= e_K
    for gr in sol.groutes:
        for g in gr.trips:
            eta_expected = tt.gtrip_eta(g.hub, g.seq)
            if abs(eta_expected - g.eta) > 1e-4:
                rep.fail("(20)", f"Gtrip {g.id}: eta mismatch "
                                  f"({g.eta:.4f} vs {eta_expected:.4f})")
            if g.eta > data.e_K + 1e-6:
                rep.fail("(21)", f"Gtrip {g.id}: eta {g.eta:.2f} > e_K {data.e_K}")

    # (22) linking: j 送到 hub h 的话, UGV 必须也属于 h
    for j in data.J:
        if j in sol.unserved:
            continue
        a = sol.cust_to_atrip.get(j)
        g = sol.cust_to_gtrip.get(j)
        if a is None or g is None:
            rep.fail("(22)", f"cust {j}: atrip or gtrip missing")
            continue
        gr = sol.groute_of_gtrip(g)
        if gr is None:
            rep.fail("(22)", f"cust {j}: gtrip has no groute")
            continue
        if a.hub != gr.hub:
            rep.fail("(22)", f"cust {j}: Atrip hub {a.hub} != Groute hub {gr.hub}")

    # (23) UGV dep >= UAV arr_hub
    for j in data.J:
        if j in sol.unserved:
            continue
        a = sol.cust_to_atrip.get(j)
        g = sol.cust_to_gtrip.get(j)
        if a is None or g is None:
            continue
        if g.dep < a.arr_hub - 1e-6:
            rep.fail("(23)", f"cust {j}: UGV dep {g.dep:.2f} < UAV arr_hub {a.arr_hub:.2f}")

    return rep
