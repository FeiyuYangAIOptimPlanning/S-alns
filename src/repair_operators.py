"""
repair_operators.py — Greedy Insertion (GI) + Greedy Insertion with Noise (GIN)

对齐论文 §4.2.3. 修正点 (v2):
- 最终决策用 4-Pass refresh 后的 **真实 objective delta**
- 局部公式仅做 K 个候选的预筛
- GI 必须枚举 inactive hubs (§4.2.3 论文明确)
- GIN: Cost × random(0.1, 1.0)
"""
from __future__ import annotations
import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING, Optional
import logging

from .structures import Solution, Atrip, Gtrip, Aroute, Groute, ID_GEN
from .timing_subproblem import refresh_all_timing, compute_atrip_window, compute_gtrip_window
from .costs import cost

if TYPE_CHECKING:
    from .data_loader import DataContainer
    from .travel_times import TravelTimes

log = logging.getLogger(__name__)

# 预筛保留的 Top-K 候选.
# NOTE: these module-level caps bound the per-customer repair cost: every kept
# (gtrip, atrip) candidate triggers a full-solution deepcopy + refresh inside
# _try_option_pair, so cost per inserted customer ~= K_GTRIP * K_ATRIP * deepcopy.
# On large instances (many routes) deepcopy is expensive, so lowering these is
# the primary runtime lever. Use set_prescreen() to override from config/CLI.
K_PRESCREEN_GTRIP = 10
K_PRESCREEN_ATRIP = 8


def set_prescreen(k_gtrip: int | None = None, k_atrip: int | None = None) -> None:
    """Override the Top-K repair pre-screen caps (call once before run_alns)."""
    global K_PRESCREEN_GTRIP, K_PRESCREEN_ATRIP
    if k_gtrip is not None:
        K_PRESCREEN_GTRIP = int(k_gtrip)
    if k_atrip is not None:
        K_PRESCREEN_ATRIP = int(k_atrip)
    log.info(f"repair prescreen set: K_GTRIP={K_PRESCREEN_GTRIP}, K_ATRIP={K_PRESCREEN_ATRIP}")


# ---------------------------------------------------------------------------
# Repair behaviour toggles.
#
#   FAST_REPAIR (default ON, SHIP THIS) : evaluate each candidate by mutating
#       the live solution + global refresh + a precise structural undo, instead
#       of deep-copying the whole solution per candidate.  Behaviour-identical
#       to the deepcopy path (same trusted refresh, same true objective delta);
#       it only avoids the O(#routes) deepcopy.  Proven bit-for-bit identical to
#       the deepcopy ground truth on both repair operators by
#       scripts/validate_fast_repair.py (max|Δ|=0 over the full objective
#       trajectory, identical best / hubs / unserved), at ~4x speedup on the
#       220-customer subsample and more at full scale (deepcopy cost grows with
#       solution size).  This is the change that makes the full case tractable.
#
#   ENABLE_FRESH_FALLBACK (default OFF, do NOT ship on) : guarantee the
#       fully-fresh (NEW_GT_NEW_GR / NEW_AT_NEW_AR) option survives the
#       K-prescreen as a last-resort feasible insertion.  This DOES drive the
#       `discard_unserved` rate to ~0% -- but validation
#       (scripts/demo_fresh_fallback.py) shows it is NET-HARMFUL to optimisation
#       quality: across two repair operators and seeds {42, 7, 123} the fresh
#       fallback took improvement from ~0.1-4.5% down to 0.00% and never reached
#       a better objective than with it off.  Mechanism: a dedicated fresh trip
#       is always feasible but expensive (investment cost), so forcing it lets
#       greedy *complete* every repair with costly fresh vehicles; that floods
#       the acceptance test with large worse-cost moves, the SA temperature
#       cools on schedule, and the search never reaches the cheap-repack basin
#       that the high-discard search finds by effectively filtering for feasible
#       cheap repacks.  In other words the high discard rate is NOT the cause of
#       poor quality -- it is the search correctly rejecting destroy moves it
#       cannot repair cheaply; the fix is to make those no-op iterations cheap
#       (FAST_REPAIR), not to force completion.  Kept as a toggle only as a
#       diagnostic / feasibility crutch (e.g. if at full scale the search shows
#       near-zero acceptance and cannot complete a single repair); expect it to
#       trade quality for completion.
# ---------------------------------------------------------------------------
ENABLE_FRESH_FALLBACK = False
FAST_REPAIR = True


def set_fresh_fallback(enabled: bool) -> None:
    global ENABLE_FRESH_FALLBACK
    ENABLE_FRESH_FALLBACK = bool(enabled)
    log.info(f"repair fresh-vehicle fallback: {ENABLE_FRESH_FALLBACK}")


def set_fast_repair(enabled: bool) -> None:
    global FAST_REPAIR
    FAST_REPAIR = bool(enabled)
    log.info(f"repair fast (in-place, no deepcopy): {FAST_REPAIR}")


def _prescreen_with_fresh(opts_sorted: list[dict], k: int, fresh_kind: str) -> list[dict]:
    """Keep the top-k already-sorted-by-local-cost options, but guarantee the
    cheapest option of `fresh_kind` is retained as a last-resort feasible
    fallback.  If no such option exists the customer may be genuinely
    unservable and the (legitimate) discard stands."""
    kept = opts_sorted[:k]
    if ENABLE_FRESH_FALLBACK and not any(o["kind"] == fresh_kind for o in kept):
        for o in opts_sorted:          # cheapest-first -> first match is cheapest fresh
            if o["kind"] == fresh_kind:
                kept = kept + [o]
                break
    return kept


# ===========================================================================
# 应用一个 gtrip option
# ===========================================================================
def _apply_gtrip_option(sol: Solution, j: int, opt: dict, data: "DataContainer",
                        tt: "TravelTimes") -> None:
    """opt 描述一个 gtrip 插入选项, 修改 sol in-place."""
    kind = opt["kind"]
    if kind == "EXIST_GT":
        gt = opt["gt"]
        pos = opt["pos"]
        gt.seq.insert(pos, j)
    elif kind == "NEW_GT_EXIST_GR":
        gr = opt["groute"]
        new_gt = Gtrip(hub=gr.hub, seq=[j])
        gr.trips.append(new_gt)
    elif kind == "NEW_GT_NEW_GR":
        h = opt["hub"]
        new_gt = Gtrip(hub=h, seq=[j])
        new_gr = Groute(hub=h, trips=[new_gt])
        sol.groutes.append(new_gr)
        sol.active_hubs.add(h)
    else:
        raise ValueError(f"Unknown gtrip option kind: {kind}")


def _apply_atrip_option(sol: Solution, j: int, opt: dict, data: "DataContainer",
                        tt: "TravelTimes") -> None:
    kind = opt["kind"]
    if kind == "EXIST_AT":
        a = opt["atrip"]
        a.pkgs.append(j)
    elif kind == "NEW_AT_EXIST_AR":
        ar = opt["aroute"]
        h = opt["hub"]
        new_at = Atrip(hub=h, pkgs=[j])
        ar.trips.append(new_at)
    elif kind == "NEW_AT_NEW_AR":
        h = opt["hub"]
        new_at = Atrip(hub=h, pkgs=[j])
        new_ar = Aroute(trips=[new_at])
        sol.aroutes.append(new_ar)
    else:
        raise ValueError(f"Unknown atrip option kind: {kind}")


# ===========================================================================
# 枚举候选
# ===========================================================================
def _local_gtrip_cost(opt: dict, j: int, data: "DataContainer", tt: "TravelTimes",
                      sol: Solution) -> float:
    kind = opt["kind"]
    if kind == "EXIST_GT":
        gt = opt["gt"]
        pos = opt["pos"]
        old_eta = gt.eta
        trial_seq = gt.seq[:pos] + [j] + gt.seq[pos:]
        new_eta = tt.gtrip_eta(gt.hub, trial_seq)
        return data.cost_K_trav * (new_eta - old_eta)
    elif kind == "NEW_GT_EXIST_GR":
        gr = opt["groute"]
        new_eta = tt.gtrip_eta(gr.hub, [j])
        return data.cost_K_trav * new_eta
    elif kind == "NEW_GT_NEW_GR":
        h = opt["hub"]
        new_eta = tt.gtrip_eta(h, [j])
        c = data.cost_K_trav * new_eta + data.cost_K_inv
        if h not in sol.active_hubs:
            c += data.cost_H
        return c
    return float("inf")


def _local_atrip_cost(opt: dict, data: "DataContainer", tt: "TravelTimes") -> float:
    kind = opt["kind"]
    if kind == "EXIST_AT":
        return 0.0
    elif kind == "NEW_AT_EXIST_AR":
        h = opt["hub"]
        return 2 * data.cost_D_trav * tt.t_hD[h]
    elif kind == "NEW_AT_NEW_AR":
        h = opt["hub"]
        return 2 * data.cost_D_trav * tt.t_hD[h] + data.cost_D_inv
    return float("inf")


def _enumerate_gtrip_options(sol: Solution, j: int, data: "DataContainer",
                             tt: "TravelTimes") -> list[dict]:
    """全部 Gtrip 插入选项 (预筛 = 只要载重/续航 viable, 不算完整时间可行性)."""
    opts: list[dict] = []
    w_j = data.customers[j].weight
    # a) 已有 Gtrip 各位置
    for gt in sol.all_gtrips():
        if gt.load + w_j > data.l_K + 1e-9:
            continue
        for pos in range(len(gt.seq) + 1):
            trial_seq = gt.seq[:pos] + [j] + gt.seq[pos:]
            new_eta = tt.gtrip_eta(gt.hub, trial_seq)
            if new_eta > data.e_K + 1e-9:
                continue
            opts.append({"kind": "EXIST_GT", "gt": gt, "pos": pos, "hub": gt.hub})
    # b) 已有 Groute 建新 Gtrip
    for gr in sol.groutes:
        new_eta = tt.gtrip_eta(gr.hub, [j])
        if new_eta > data.e_K + 1e-9:
            continue
        if w_j > data.l_K + 1e-9:
            continue
        # 新 trip 的往返须可达
        if 2 * tt.t_hj_K[gr.hub][j] > data.e_K + 1e-9:
            continue
        opts.append({"kind": "NEW_GT_EXIST_GR", "groute": gr, "hub": gr.hub})
    # c) 新 Groute on all hubs (含 inactive)
    for h in data.H:
        new_eta = tt.gtrip_eta(h, [j])
        if new_eta > data.e_K + 1e-9:
            continue
        if w_j > data.l_K + 1e-9:
            continue
        if 2 * tt.t_hj_K[h][j] > data.e_K + 1e-9:
            continue
        opts.append({"kind": "NEW_GT_NEW_GR", "hub": h})
    return opts


def _enumerate_atrip_options(sol: Solution, j: int, hub_j: int,
                             data: "DataContainer", tt: "TravelTimes") -> list[dict]:
    opts: list[dict] = []
    w_j = data.customers[j].weight
    # 只考虑 hub = hub_j 的 Atrip (论文 linking: UAV 层与 UGV 层 hub 一致)
    # 已有 Atrip
    for a in sol.atrips_of_hub(hub_j):
        if a.load + w_j <= data.l_D + 1e-9:
            opts.append({"kind": "EXIST_AT", "atrip": a, "hub": hub_j})
    # 已有 Aroute 加新 Atrip
    for ar in sol.aroutes:
        if w_j <= data.l_D + 1e-9:
            opts.append({"kind": "NEW_AT_EXIST_AR", "aroute": ar, "hub": hub_j})
    # 新 Aroute + 新 Atrip
    if w_j <= data.l_D + 1e-9:
        opts.append({"kind": "NEW_AT_NEW_AR", "hub": hub_j})
    return opts


# ===========================================================================
# In-place candidate evaluation (FAST_REPAIR path) — no per-candidate deepcopy
# ===========================================================================
def _apply_pair_inplace(sol: Solution, j: int, gt_opt: dict, at_opt: dict,
                        data: "DataContainer", tt: "TravelTimes") -> dict:
    """Apply (gt_opt, at_opt) to the LIVE solution and return an undo token.
    Reuses the exact same structural mutations as the commit path, so the
    resulting structure is byte-identical to the deepcopy path."""
    gk = gt_opt["kind"]
    was_active = (gt_opt["hub"] in sol.active_hubs) if gk == "NEW_GT_NEW_GR" else None
    _apply_gtrip_option(sol, j, gt_opt, data, tt)
    _apply_atrip_option(sol, j, at_opt, data, tt)
    return {"gt": (gk, gt_opt, was_active), "at": (at_opt["kind"], at_opt)}


def _undo_pair_inplace(sol: Solution, undo: dict) -> None:
    """Exact inverse of _apply_pair_inplace (atrip first, then gtrip).  New
    objects were appended last, so popping restores the prior structure."""
    ak, at_opt = undo["at"]
    if ak == "EXIST_AT":
        at_opt["atrip"].pkgs.pop()              # j was appended last
    elif ak == "NEW_AT_EXIST_AR":
        at_opt["aroute"].trips.pop()            # new atrip appended last
    elif ak == "NEW_AT_NEW_AR":
        sol.aroutes.pop()                       # new aroute appended last
    gk, gt_opt, was_active = undo["gt"]
    if gk == "EXIST_GT":
        gt_opt["gt"].seq.pop(gt_opt["pos"])     # j inserted at pos
    elif gk == "NEW_GT_EXIST_GR":
        gt_opt["groute"].trips.pop()            # new gtrip appended last
    elif gk == "NEW_GT_NEW_GR":
        sol.groutes.pop()                       # new groute appended last
        if not was_active:
            sol.active_hubs.discard(gt_opt["hub"])


def _try_option_pair_inplace(
    sol: Solution, j: int, gt_opt: dict, at_opt: dict,
    data: "DataContainer", tt: "TravelTimes",
    base_cost: float, noise_factor: float = 1.0,
) -> Optional[float]:
    """FAST_REPAIR equivalent of _try_option_pair: same trusted global refresh
    and same true objective delta, but mutate-evaluate-undo on the live solution
    rather than deepcopy-per-candidate.  Leaves the structure exactly as found;
    timing fields are left dirty and are overwritten by the next refresh (the
    caller refreshes before the next enumeration / on commit)."""
    # hub linking (UAV hub must match UGV hub); defensive, holds by construction
    if at_opt["hub"] != gt_opt["hub"]:
        return None
    undo = _apply_pair_inplace(sol, j, gt_opt, at_opt, data, tt)
    refresh_all_timing(sol, data, tt)
    ok = sol.feasible
    if ok:
        for gg in sol.all_gtrips():
            for jj in gg.seq:
                arr = gg.cust_arr.get(jj)
                if arr is None or arr > data.customers[jj].t_due + 1e-6:
                    ok = False
                    break
            if not ok:
                break
    delta = (cost(sol, data, tt) - base_cost) if ok else None
    _undo_pair_inplace(sol, undo)
    return None if delta is None else delta * noise_factor


# ===========================================================================
# GI core
# ===========================================================================
def _try_option_pair(
    base_sol: Solution,
    j: int,
    gt_opt: dict,
    at_opt: dict,
    data: "DataContainer",
    tt: "TravelTimes",
    base_cost: float,
    noise_factor: float = 1.0,
) -> Optional[float]:
    """尝试在 deepcopy 上应用 (gt_opt, at_opt), 刷新, 返回 (cost_delta * noise) 或 None(infeasible)."""
    if FAST_REPAIR:
        return _try_option_pair_inplace(base_sol, j, gt_opt, at_opt, data, tt,
                                        base_cost, noise_factor)
    sol_try = deepcopy(base_sol)
    # 需要 re-lookup gt/at/gr/ar on deepcopy
    # 简化: 用 id_gen 赋予的 id 定位
    gt_opt_new = _relocate_option(gt_opt, sol_try)
    at_opt_new = _relocate_option(at_opt, sol_try)
    if gt_opt_new is None or at_opt_new is None:
        return None
    _apply_gtrip_option(sol_try, j, gt_opt_new, data, tt)
    sol_try.unserved.discard(j)
    sol_try.rebuild_cust_indexes()
    # 在 apply atrip option 之前, hub_j 已由 gt_opt 决定
    hub_j = sol_try.cust_to_hub.get(j)
    if hub_j is None:
        # cust_to_hub 里还没同步(新 Groute 场景), 直接从 at_opt 取
        hub_j = at_opt_new["hub"]
    # 但 at_opt 的 hub 必须与 gt_opt 一致
    if at_opt_new["hub"] != (gt_opt_new.get("hub") if "hub" in gt_opt_new
                             else _hub_of_gt_opt(gt_opt_new)):
        return None
    _apply_atrip_option(sol_try, j, at_opt_new, data, tt)
    sol_try.rebuild_cust_indexes()
    refresh_all_timing(sol_try, data, tt)
    if not sol_try.feasible:
        return None
    # 客户送达校验
    for gg in sol_try.all_gtrips():
        for jj in gg.seq:
            arr = gg.cust_arr.get(jj)
            if arr is None or arr > data.customers[jj].t_due + 1e-6:
                return None
    delta = cost(sol_try, data, tt) - base_cost
    return delta * noise_factor


def _hub_of_gt_opt(opt: dict) -> int:
    if "hub" in opt:
        return opt["hub"]
    if opt["kind"] == "EXIST_GT":
        return opt["gt"].hub
    if opt["kind"] == "NEW_GT_EXIST_GR":
        return opt["groute"].hub
    raise ValueError("Cannot determine hub of gt_opt")


def _relocate_option(opt: dict, sol_try: Solution) -> Optional[dict]:
    """由于 deepcopy 产生新对象引用, 用 id (Atrip.id / Gtrip.id / Aroute.uav_id / Groute.ugv_id)
    在 sol_try 里找对应的新对象."""
    new_opt = dict(opt)
    kind = opt["kind"]
    if kind == "EXIST_GT":
        target_id = opt["gt"].id
        found = None
        for gr in sol_try.groutes:
            for g in gr.trips:
                if g.id == target_id:
                    found = g
                    break
            if found:
                break
        if found is None:
            return None
        new_opt["gt"] = found
    elif kind == "NEW_GT_EXIST_GR":
        target_id = opt["groute"].ugv_id
        found = None
        for gr in sol_try.groutes:
            if gr.ugv_id == target_id:
                found = gr
                break
        if found is None:
            return None
        new_opt["groute"] = found
    elif kind == "EXIST_AT":
        target_id = opt["atrip"].id
        found = None
        for ar in sol_try.aroutes:
            for a in ar.trips:
                if a.id == target_id:
                    found = a
                    break
            if found:
                break
        if found is None:
            return None
        new_opt["atrip"] = found
    elif kind == "NEW_AT_EXIST_AR":
        target_id = opt["aroute"].uav_id
        found = None
        for ar in sol_try.aroutes:
            if ar.uav_id == target_id:
                found = ar
                break
        if found is None:
            return None
        new_opt["aroute"] = found
    # NEW_GT_NEW_GR, NEW_AT_NEW_AR 不需要 relocate
    return new_opt


def greedy_insertion(
    sol: Solution,
    data: "DataContainer",
    tt: "TravelTimes",
    rng: np.random.Generator,
    use_noise: bool = False,
) -> None:
    """GI (use_noise=False) 或 GIN (use_noise=True).

    对每个 unserved j (打乱序), 枚举 (gtrip option, atrip option) 配对,
    用局部成本排序, 对前 K 做真实 refresh 评估, 取真实 Δcost 最小.
    """
    unserved_list = list(sol.unserved)
    rng.shuffle(unserved_list)

    for j in unserved_list:
        gtrip_opts = _enumerate_gtrip_options(sol, j, data, tt)
        if not gtrip_opts:
            log.warning(f"GI: customer {j} has no gtrip option")
            continue
        # 预筛排序
        gtrip_opts.sort(key=lambda o: _local_gtrip_cost(o, j, data, tt, sol))
        gtrip_opts = _prescreen_with_fresh(gtrip_opts, K_PRESCREEN_GTRIP, "NEW_GT_NEW_GR")

        base_cost = cost(sol, data, tt)
        best = None  # (delta, gt_opt, at_opt)

        for gt_opt in gtrip_opts:
            hub_j = _hub_of_gt_opt(gt_opt)
            atrip_opts = _enumerate_atrip_options(sol, j, hub_j, data, tt)
            if not atrip_opts:
                continue
            atrip_opts.sort(key=lambda o: _local_atrip_cost(o, data, tt))
            atrip_opts = _prescreen_with_fresh(atrip_opts, K_PRESCREEN_ATRIP, "NEW_AT_NEW_AR")
            for at_opt in atrip_opts:
                nf = 1.0 if not use_noise else float(rng.uniform(0.1, 1.0))
                delta = _try_option_pair(sol, j, gt_opt, at_opt, data, tt,
                                         base_cost, noise_factor=nf)
                if delta is None:
                    continue
                if best is None or delta < best[0]:
                    best = (delta, gt_opt, at_opt)

        if best is None:
            log.warning(f"GI: customer {j} has no feasible (gt, at) pair")
            if FAST_REPAIR:
                refresh_all_timing(sol, data, tt)  # restore fields dirtied by in-place trials
            continue

        _, gt_opt, at_opt = best
        # 真正应用到 sol (注意 gt_opt/at_opt 引用的是 sol 的对象, 不是 deepcopy)
        _apply_gtrip_option(sol, j, gt_opt, data, tt)
        sol.unserved.discard(j)
        sol.rebuild_cust_indexes()
        _apply_atrip_option(sol, j, at_opt, data, tt)
        sol.rebuild_cust_indexes()
        refresh_all_timing(sol, data, tt)


def greedy_insertion_noise(
    sol: Solution,
    data: "DataContainer",
    tt: "TravelTimes",
    rng: np.random.Generator,
) -> None:
    greedy_insertion(sol, data, tt, rng, use_noise=True)


# ===========================================================================
# 注册
# ===========================================================================
REPAIR_OPS = {
    "GI":  greedy_insertion,
    "GIN": greedy_insertion_noise,
}
REPAIR_NAMES = list(REPAIR_OPS.keys())


# ===========================================================================
# Service-Structure guided repair (S-alns)
# ---------------------------------------------------------------------------
# Strategy (first-version, lightweight):
#   1. Group unserved customers by service cluster.
#   2. Sort within each group by due_time and distance-to-medoid.
#   3. For each customer, enumerate gtrip options with an enriched hub set:
#         active hubs ∪ top_m_hubs[j] ∪ {cluster preferred hub}
#       ∪ {adjacent-cluster preferred hubs} ∪ top-N inactive hubs
#      then re-rank with local cost, but force-preserve top-N inactive-hub
#      candidates so they don't get cut by K_PRESCREEN_GTRIP
#      ("inactive-hub candidate preservation" — Q4 first version).
#   4. Final acceptance still uses deepcopy + refresh_all_timing + true delta
#      (identical mechanism as baseline GI).
#   5. Any unserved customer left after this pass falls back to baseline GI/GIN.
# ===========================================================================

# 在 K_PRESCREEN_GTRIP 裁剪前强制保留的 inactive-hub 候选数
SERVICE_INACTIVE_HUB_RESERVE = 2


def _candidate_hub_set_for_customer(
    sol: Solution, j: int, service_structure, service_cfg,
) -> set[int]:
    """Build the enriched candidate hub set per customer (Eq.23 in the paper).

    Membership:
        - active hubs
        - service_structure.top_hubs[j]
        - cluster preferred hub
        - graph-neighbor clusters' preferred hubs
        - small quota of top inactive hubs by hub_pref
    """
    hubs: set[int] = set(sol.active_hubs)
    if service_structure is None:
        return hubs
    # 1. top-m hubs from softmax preference
    hubs.update(service_structure.top_hubs.get(j, []))
    # 2. own cluster preferred + neighbor preferred
    cid = service_structure.cluster_id_of_customer.get(j)
    cid_to_c = {c.id: c for c in service_structure.clusters}
    if cid is not None and cid in cid_to_c:
        c_own = cid_to_c[cid]
        if c_own.preferred_hub >= 0:
            hubs.add(c_own.preferred_hub)
        for nb in service_structure.cluster_graph.get(cid, []):
            c_nb = cid_to_c.get(nb)
            if c_nb is not None and c_nb.preferred_hub >= 0:
                hubs.add(c_nb.preferred_hub)
    # 3. top inactive hubs by per-customer preference
    j_idx = service_structure.customer_id_to_idx.get(j)
    if j_idx is not None:
        n_hub = service_structure.hub_pref.shape[1]
        order = np.argsort(-service_structure.hub_pref[j_idx])
        added = 0
        for h_local_idx in order:
            hid = service_structure.idx_to_hub_id[int(h_local_idx)]
            if hid not in sol.active_hubs:
                hubs.add(hid)
                added += 1
                if added >= SERVICE_INACTIVE_HUB_RESERVE:
                    break
    return hubs


def _enumerate_gtrip_options_service_guided(
    sol: Solution, j: int, data, tt,
    service_structure, service_cfg,
) -> list[dict]:
    """Like _enumerate_gtrip_options, but candidate hub set is enriched and the
    inactive-hub reserve is kept *before* K_PRESCREEN sorting."""
    base_opts = _enumerate_gtrip_options(sol, j, data, tt)
    if service_structure is None:
        return base_opts
    candidate_hubs = _candidate_hub_set_for_customer(
        sol, j, service_structure, service_cfg
    )
    # base_opts 已经枚举了所有 hubs (NEW_GT_NEW_GR over all data.H), 与 active
    # 子集.  这里要做的事情就是按 candidate_hubs 过滤 + 计算 local cost + 排序,
    # 但确保 inactive-hub 选项不被 K_PRESCREEN 全部裁掉.
    if not candidate_hubs:
        return base_opts
    filt: list[dict] = []
    for opt in base_opts:
        hub_h = _hub_of_gt_opt(opt) if "hub" in opt or opt["kind"] != "NEW_GT_NEW_GR" \
                                     else opt["hub"]
        if hub_h in candidate_hubs:
            filt.append(opt)
    if not filt:
        return base_opts
    return filt


def _service_guided_unserved_order(
    sol: Solution, data, service_structure,
) -> list[int]:
    """Order unserved customers: group by cluster, sort within group by
    (due_time, distance-to-medoid). Customers without cluster id appended at end.
    """
    unserved = list(sol.unserved)
    if service_structure is None:
        return unserved
    cust_id_to_idx = service_structure.customer_id_to_idx
    bucket_by_cluster: dict[int, list[int]] = {}
    no_cluster: list[int] = []
    for j in unserved:
        cid = service_structure.cluster_id_of_customer.get(j)
        if cid is None:
            no_cluster.append(j)
        else:
            bucket_by_cluster.setdefault(cid, []).append(j)
    cid_to_c = {c.id: c for c in service_structure.clusters}
    ordered: list[int] = []
    for cid in sorted(bucket_by_cluster.keys()):
        c = cid_to_c.get(cid)
        members = bucket_by_cluster[cid]
        if c is None:
            ordered.extend(members)
            continue
        medoid_idx = cust_id_to_idx.get(c.medoid, None)
        def key_fn(j):
            t_due = data.customers[j].t_due
            j_idx = cust_id_to_idx.get(j, None)
            d_med = (service_structure.D_cust[j_idx, medoid_idx]
                     if (j_idx is not None and medoid_idx is not None)
                     else 0.0)
            return (t_due, d_med)
        members.sort(key=key_fn)
        ordered.extend(members)
    ordered.extend(no_cluster)
    return ordered


def service_greedy_insertion(
    sol: Solution, data, tt, rng,
    service_structure=None, service_cfg=None, use_noise: bool = False,
) -> None:
    """Service-Structure guided greedy insertion.

    Acceptance criterion (deepcopy + refresh + true delta) is identical to
    baseline GI; only candidate generation and customer ordering differ.
    Customers whose service-guided pass fails are handled by a final
    baseline-GI fallback.
    """
    if service_structure is None:
        greedy_insertion(sol, data, tt, rng, use_noise=use_noise)
        return

    # Q4: cluster-group ordering (lightweight, NOT atomic batch insertion).
    ordered = _service_guided_unserved_order(sol, data, service_structure)
    # 重要: ordered 只用 once, 然后插入过程中 sol.unserved 变化 -> 处理新 unserved
    handled: set[int] = set()
    for j in ordered:
        if j not in sol.unserved or j in handled:
            continue
        gtrip_opts = _enumerate_gtrip_options_service_guided(
            sol, j, data, tt, service_structure, service_cfg
        )
        if not gtrip_opts:
            # 退到 baseline 全集枚举
            gtrip_opts = _enumerate_gtrip_options(sol, j, data, tt)
        if not gtrip_opts:
            log.warning(f"SGI: customer {j} has no gtrip option")
            continue

        # 预筛排序, 同时保留 SERVICE_INACTIVE_HUB_RESERVE 个 inactive-hub 候选
        # (inactive-hub candidate preservation).
        gtrip_opts_sorted = sorted(
            gtrip_opts,
            key=lambda o: _local_gtrip_cost(o, j, data, tt, sol),
        )
        # 强制保留 inactive hub options
        active_set = sol.active_hubs
        kept_inactive = []
        for o in gtrip_opts_sorted:
            h_o = _hub_of_gt_opt(o)
            if h_o not in active_set:
                kept_inactive.append(o)
            if len(kept_inactive) >= SERVICE_INACTIVE_HUB_RESERVE:
                break
        top_active = gtrip_opts_sorted[:K_PRESCREEN_GTRIP]
        # union 但保持顺序: top_active + kept_inactive 中尚未包含的
        merged: list[dict] = list(top_active)
        seen_ids = set(id(x) for x in merged)
        for o in kept_inactive:
            if id(o) not in seen_ids:
                merged.append(o)
                seen_ids.add(id(o))
        # guaranteed fully-fresh groute fallback (active or inactive hub) so a
        # reliably-feasible dedicated trip is always evaluated -> no artifact discards
        if ENABLE_FRESH_FALLBACK and not any(o["kind"] == "NEW_GT_NEW_GR" for o in merged):
            for o in gtrip_opts_sorted:
                if o["kind"] == "NEW_GT_NEW_GR":
                    merged.append(o)
                    break
        gtrip_opts = merged

        base_cost = cost(sol, data, tt)
        best = None  # (delta, gt_opt, at_opt)

        for gt_opt in gtrip_opts:
            hub_j = _hub_of_gt_opt(gt_opt)
            atrip_opts = _enumerate_atrip_options(sol, j, hub_j, data, tt)
            if not atrip_opts:
                continue
            atrip_opts.sort(key=lambda o: _local_atrip_cost(o, data, tt))
            atrip_opts = _prescreen_with_fresh(atrip_opts, K_PRESCREEN_ATRIP, "NEW_AT_NEW_AR")
            for at_opt in atrip_opts:
                nf = 1.0 if not use_noise else float(rng.uniform(0.1, 1.0))
                delta = _try_option_pair(sol, j, gt_opt, at_opt, data, tt,
                                          base_cost, noise_factor=nf)
                if delta is None:
                    continue
                if best is None or delta < best[0]:
                    best = (delta, gt_opt, at_opt)

        if best is None:
            # 留给 fallback 处理
            if FAST_REPAIR:
                refresh_all_timing(sol, data, tt)  # restore fields dirtied by in-place trials
            continue

        _, gt_opt, at_opt = best
        _apply_gtrip_option(sol, j, gt_opt, data, tt)
        sol.unserved.discard(j)
        sol.rebuild_cust_indexes()
        _apply_atrip_option(sol, j, at_opt, data, tt)
        sol.rebuild_cust_indexes()
        refresh_all_timing(sol, data, tt)
        handled.add(j)

    # Fallback: 剩余 unserved 用 baseline GI/GIN
    if sol.unserved:
        log.info(f"SGI fallback to baseline GI for {len(sol.unserved)} customers")
        greedy_insertion(sol, data, tt, rng, use_noise=use_noise)


def service_greedy_insertion_noise(
    sol: Solution, data, tt, rng,
    service_structure=None, service_cfg=None,
) -> None:
    service_greedy_insertion(
        sol, data, tt, rng,
        service_structure=service_structure,
        service_cfg=service_cfg,
        use_noise=True,
    )


REPAIR_OPS_STRUCTURE = {
    "SGI":  service_greedy_insertion,
    "SGIN": service_greedy_insertion_noise,
}
REPAIR_NAMES_STRUCTURE = list(REPAIR_OPS_STRUCTURE.keys())
