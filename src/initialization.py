"""
initialization.py — 四阶段 backward induction 初始化

对齐论文 §4.2.1:
  Phase 1: Generate Initial Gtrips   (hub activation + nearest-insert)
  Phase 2: Generate Initial Groutes  (backward concatenation)
  Phase 3: Generate Initial Atrips   (due_FC-sorted)
  Phase 4: Generate Initial Aroutes  (+ final earliest-refresh)
"""
from __future__ import annotations
import logging
import math
from copy import deepcopy

from .structures import Customer, Hub, Atrip, Gtrip, Aroute, Groute, Solution, ID_GEN
from .timing_subproblem import (
    compute_gtrip_window, compute_atrip_window,
    schedule_groute_backward, schedule_aroute_forward,
    refresh_all_timing,
)
from .data_loader import DataContainer
from .travel_times import TravelTimes

log = logging.getLogger(__name__)


# ===========================================================================
# Phase 1: Generate Initial Gtrips (§4.2.1 Phase-1)
# ===========================================================================
def phase1_generate_gtrips(
    data: DataContainer,
    tt: TravelTimes,
    r_c: float = 1.0 / 3.0,
) -> tuple[list[Gtrip], set[int], dict[int, int]]:
    """返回 (gtrips, active_hubs, cust_to_hub).

    论文中 r_c * e 为 **时间预算** (即 t_{h,j}^K <= r_c * e_K).
    """
    unserved: set[int] = set(data.J)
    active_hubs: set[int] = set()
    gtrips: list[Gtrip] = []
    cust_to_hub: dict[int, int] = {}
    cust_to_gtrip: dict[int, Gtrip] = {}

    def reachable_from_hub(h: int, unserved_set: set[int]) -> set[int]:
        """按论文: 2·t_{h,j}^K <= e_K AND t_rel + t_hD + t_{h,j} <= t_due AND t <= r_c*e_K."""
        res = set()
        for j in unserved_set:
            t_to = tt.t_hj_K[h][j]
            if 2 * t_to > data.e_K:
                continue
            if data.customers[j].t_rel + tt.t_hD[h] + t_to > data.customers[j].t_due + 1e-9:
                continue
            if t_to > r_c * data.e_K + 1e-9:
                continue
            res.add(j)
        return res

    while unserved:
        # --- Step 1: 选覆盖最多未服务客户的 hub ---
        best_h, best_cov = None, -1
        for h in data.H:
            if h in active_hubs:
                continue
            cov = len(reachable_from_hub(h, unserved))
            if cov > best_cov:
                best_cov, best_h = cov, h
        if best_h is None or best_cov <= 0:
            log.warning(f"Phase 1: 剩余 {len(unserved)} 客户无 hub 覆盖, 放宽条件")
            # 放宽: 忽略 r_c 约束, 只看 battery + time window
            for h in data.H:
                if h in active_hubs:
                    continue
                cov = 0
                for j in unserved:
                    if 2 * tt.t_hj_K[h][j] <= data.e_K and \
                       data.customers[j].t_rel + tt.t_hD[h] + tt.t_hj_K[h][j] \
                       <= data.customers[j].t_due + 1e-9:
                        cov += 1
                if cov > best_cov:
                    best_cov, best_h = cov, h
            if best_h is None or best_cov <= 0:
                log.error(f"Phase 1: 彻底无解, 放弃剩余客户 {unserved}")
                break
        active_hubs.add(best_h)

        # --- Step 2-5: 反复构建/插入 ---
        while True:
            # reachable filter (不含 r_c, 用标准battery+time)
            reachable_now = set()
            for j in unserved:
                if 2 * tt.t_hj_K[best_h][j] <= data.e_K and \
                   data.customers[j].t_rel + tt.t_hD[best_h] + tt.t_hj_K[best_h][j] \
                   <= data.customers[j].t_due + 1e-9:
                    reachable_now.add(j)
            if not reachable_now:
                break

            # Step 3: 新建 Gtrip, 起点取最近
            j_start = min(reachable_now, key=lambda j: tt.t_hj_K[best_h][j])
            new_gt = Gtrip(hub=best_h, seq=[j_start])
            # 用 t_rel_H = t_rel + t_hD 的乐观上界
            data.customers[j_start].t_rel_H = data.customers[j_start].t_rel + tt.t_hD[best_h]
            compute_gtrip_window(new_gt, data, tt)
            if not new_gt.feasible:
                # 单客户都不可行, 跳过
                log.warning(f"Phase 1: 客户 {j_start} 单点 Gtrip 不可行, skip")
                unserved.discard(j_start)
                continue
            gtrips.append(new_gt)
            cust_to_hub[j_start] = best_h
            cust_to_gtrip[j_start] = new_gt
            unserved.discard(j_start)

            # Step 4: 反复向所有当前 gtrips 尝试插入下一最近客户
            while True:
                reachable_now = set()
                for j in unserved:
                    if 2 * tt.t_hj_K[best_h][j] <= data.e_K and \
                       data.customers[j].t_rel + tt.t_hD[best_h] + tt.t_hj_K[best_h][j] \
                       <= data.customers[j].t_due + 1e-9:
                        reachable_now.add(j)
                if not reachable_now:
                    break
                j_next = min(reachable_now, key=lambda j: tt.t_hj_K[best_h][j])
                data.customers[j_next].t_rel_H = \
                    data.customers[j_next].t_rel + tt.t_hD[best_h]

                inserted = False
                # 论文: "first feasible Gtrip, within it shortest-total insertion"
                for gt in gtrips:
                    if gt.hub != best_h:
                        continue
                    # 找所有可行位置, 取使 eta 最小的
                    best_pos, best_eta = None, float("inf")
                    orig_seq = gt.seq[:]
                    for pos in range(len(orig_seq) + 1):
                        trial_seq = orig_seq[:pos] + [j_next] + orig_seq[pos:]
                        new_eta = tt.gtrip_eta(gt.hub, trial_seq)
                        new_load = gt.load + data.customers[j_next].weight
                        if new_eta > data.e_K:
                            continue
                        if new_load > data.l_K:
                            continue
                        # 时间窗检查
                        old_seq, old_load, old_eta = gt.seq, gt.load, gt.eta
                        gt.seq, gt.load = trial_seq, new_load
                        compute_gtrip_window(gt, data, tt)
                        ok = gt.feasible
                        gt.seq, gt.load, gt.eta = old_seq, old_load, old_eta
                        compute_gtrip_window(gt, data, tt)
                        if ok and new_eta < best_eta:
                            best_pos, best_eta = pos, new_eta
                    if best_pos is not None:
                        # 实际插入
                        gt.seq.insert(best_pos, j_next)
                        gt.load += data.customers[j_next].weight
                        compute_gtrip_window(gt, data, tt)
                        cust_to_hub[j_next] = best_h
                        cust_to_gtrip[j_next] = gt
                        unserved.discard(j_next)
                        inserted = True
                        break  # first-feasible-Gtrip
                if not inserted:
                    # 回 Step 3: 建新 Gtrip for j_next
                    new_gt = Gtrip(hub=best_h, seq=[j_next])
                    compute_gtrip_window(new_gt, data, tt)
                    if not new_gt.feasible:
                        log.warning(f"Phase 1: 新 Gtrip for {j_next} 不可行, skip")
                        unserved.discard(j_next)
                        continue
                    gtrips.append(new_gt)
                    cust_to_hub[j_next] = best_h
                    cust_to_gtrip[j_next] = new_gt
                    unserved.discard(j_next)

    return gtrips, active_hubs, cust_to_hub


# ===========================================================================
# Phase 2: Generate Initial Groutes (§4.2.1 Phase-2)
# ===========================================================================
def phase2_generate_groutes(
    gtrips: list[Gtrip],
    active_hubs: set[int],
    data: DataContainer,
    tt: TravelTimes,
) -> list[Groute]:
    groutes: list[Groute] = []
    for h in active_hubs:
        hub_gts = [g for g in gtrips if g.hub == h]
        if not hub_gts:
            continue
        # 按 early + eta (= earliest return time) 从 late 到 early
        hub_gts.sort(key=lambda g: g.early + g.eta, reverse=True)
        hub_groutes: list[Groute] = []
        # 第一个建新 Groute
        first = hub_gts[0]
        first.dep = first.late
        r0 = Groute(hub=h, trips=[first])
        schedule_groute_backward(r0, data, tt)
        hub_groutes.append(r0)

        for gt in hub_gts[1:]:
            # 按 groute 上 "earliest departing Gtrip"(trips[0]) 的 dep 排序
            sorted_rs = sorted(hub_groutes, key=lambda r: r.trips[0].dep)
            inserted = False
            for r in sorted_rs:
                w_star = r.trips[0]
                # 论文可插性: τ^early_w + η_w + t_K_set <= τ^O_{w*}
                if gt.early + gt.eta + data.t_K_set <= w_star.dep + 1e-9:
                    gt.dep = min(
                        w_star.dep - data.t_K_set - gt.eta,
                        gt.late,
                    )
                    r.trips.insert(0, gt)
                    schedule_groute_backward(r, data, tt)
                    if r.feasible:
                        inserted = True
                        break
                    else:
                        # 回滚
                        r.trips.pop(0)
                        schedule_groute_backward(r, data, tt)
            if not inserted:
                gt.dep = gt.late
                r_new = Groute(hub=h, trips=[gt])
                schedule_groute_backward(r_new, data, tt)
                hub_groutes.append(r_new)
        groutes.extend(hub_groutes)
    return groutes


# ===========================================================================
# Phase 3: Generate Initial Atrips (§4.2.1 Phase-3)
# ===========================================================================
def phase3_generate_atrips(
    gtrips: list[Gtrip],
    active_hubs: set[int],
    data: DataContainer,
    tt: TravelTimes,
    cust_to_hub: dict[int, int],
) -> list[Atrip]:
    atrips: list[Atrip] = []

    for h in active_hubs:
        # 先把每个客户的 t_due_H / t_due_FC 按所在 Gtrip 的 dep 赋值
        for gt in gtrips:
            if gt.hub != h:
                continue
            for j in gt.seq:
                data.customers[j].t_due_H = gt.dep
                data.customers[j].t_due_FC = gt.dep - tt.t_hD[h]

        cust_h = [j for j in data.J if cust_to_hub.get(j) == h]
        if not cust_h:
            continue
        # 按 t_due_FC 从 late 到 early
        cust_h.sort(key=lambda j: data.customers[j].t_due_FC, reverse=True)

        hub_atrips: list[Atrip] = []
        first_j = cust_h[0]
        a0 = Atrip(hub=h, pkgs=[first_j])
        compute_atrip_window(a0, data, tt)
        hub_atrips.append(a0)

        for j in cust_h[1:]:
            # sort by earliest departure late→early = by early DESC (arr_hub 视角)
            sorted_as = sorted(hub_atrips, key=lambda a: a.early, reverse=True)
            inserted = False
            for a in sorted_as:
                if a.load + data.customers[j].weight > data.l_D + 1e-9:
                    continue
                # trial
                old_pkgs = a.pkgs[:]
                a.pkgs.append(j)
                compute_atrip_window(a, data, tt)
                if a.feasible:
                    inserted = True
                    break
                else:
                    a.pkgs = old_pkgs
                    compute_atrip_window(a, data, tt)
            if not inserted:
                a_new = Atrip(hub=h, pkgs=[j])
                compute_atrip_window(a_new, data, tt)
                hub_atrips.append(a_new)
        atrips.extend(hub_atrips)
    return atrips


# ===========================================================================
# Phase 4: Generate Initial Aroutes (§4.2.1 Phase-4)
# ===========================================================================
def phase4_generate_aroutes(
    atrips: list[Atrip],
    data: DataContainer,
    tt: TravelTimes,
) -> list[Aroute]:
    if not atrips:
        return []
    # 按 early + 2*t_hD (earliest return) 从 late 到 early
    atrips_sorted = sorted(
        atrips,
        key=lambda a: a.early + 2 * tt.t_hD[a.hub],
        reverse=True,
    )
    aroutes: list[Aroute] = []
    first = atrips_sorted[0]
    first.arr_hub = first.late
    first.dep_fc = first.arr_hub - tt.t_hD[first.hub]
    first.return_fc = first.arr_hub + tt.t_hD[first.hub]
    aroutes.append(Aroute(trips=[first]))

    for v in atrips_sorted[1:]:
        sorted_ars = sorted(aroutes, key=lambda r: r.trips[0].arr_hub)
        inserted = False
        for r in sorted_ars:
            v_star = r.trips[0]
            # v 紧挨在 v_star 之前的可行性:
            # v.arr_hub + t_hD[v.hub] + t_D_set + t_hD[v_star.hub] <= v_star.arr_hub
            gap = v_star.arr_hub - tt.t_hD[v_star.hub] - data.t_D_set - tt.t_hD[v.hub]
            if v.early <= gap + 1e-9:
                v.arr_hub = min(gap, v.late)
                v.dep_fc = v.arr_hub - tt.t_hD[v.hub]
                v.return_fc = v.arr_hub + tt.t_hD[v.hub]
                r.trips.insert(0, v)
                inserted = True
                break
        if not inserted:
            aroutes.append(Aroute(trips=[v]))

    # Step 4: 统一 forward 更新为 earliest
    for r in aroutes:
        schedule_aroute_forward(r, data, tt)
    return aroutes


# ===========================================================================
# 汇总: four_phase_initialization
# ===========================================================================
def four_phase_initialization(
    data: DataContainer,
    tt: TravelTimes,
    r_c: float = 1.0 / 3.0,
) -> Solution:
    """论文 §4.2.1 四阶段初始化入口."""
    # Phase 1
    log.info("Phase 1: Generate initial Gtrips...")
    gtrips, active_hubs, cust_to_hub = phase1_generate_gtrips(data, tt, r_c=r_c)
    log.info(f"  -> {len(gtrips)} Gtrips, {len(active_hubs)} active hubs, "
             f"{len(cust_to_hub)}/{len(data.J)} customers placed")

    # Phase 2
    log.info("Phase 2: Generate initial Groutes...")
    groutes = phase2_generate_groutes(gtrips, active_hubs, data, tt)
    log.info(f"  -> {len(groutes)} Groutes")

    # Phase 3
    log.info("Phase 3: Generate initial Atrips...")
    atrips = phase3_generate_atrips(gtrips, active_hubs, data, tt, cust_to_hub)
    log.info(f"  -> {len(atrips)} Atrips")

    # Phase 4
    log.info("Phase 4: Generate initial Aroutes...")
    aroutes = phase4_generate_aroutes(atrips, data, tt)
    log.info(f"  -> {len(aroutes)} Aroutes")

    sol = Solution(
        active_hubs=active_hubs,
        aroutes=aroutes,
        groutes=groutes,
        unserved=set(data.J) - set(cust_to_hub.keys()),
    )
    sol.rebuild_cust_indexes()
    # 全局 4-Pass refresh
    refresh_all_timing(sol, data, tt)
    return sol


# ===========================================================================
# Service-Structure guided Phase 1 (S-alns)
# ---------------------------------------------------------------------------
# Replaces the hub-seeding loop with a service-structure score:
#     Seed(h) = Σ_r w_r * P_rh - Ω_h
#     w_r     = size_r + total_weight_r / l_K + avg_time_urgency_r
#     Ω_h     = (cost_H + cost_K_inv) / max(1, n_customers)   (gentle scaling)
# Customer-to-hub assignment inside each cluster uses the cluster preferred
# hub first, falling back to baseline coverage rules when infeasible.
# ===========================================================================
def phase1_generate_gtrips_service_guided(
    data: DataContainer,
    tt: TravelTimes,
    service_structure,
    r_c: float = 1.0 / 3.0,
) -> tuple[list[Gtrip], set[int], dict[int, int]]:
    """Service-structure guided variant of phase1_generate_gtrips.

    Returns the same (gtrips, active_hubs, cust_to_hub) triple as the baseline.
    """
    unserved: set[int] = set(data.J)
    active_hubs: set[int] = set()
    gtrips: list[Gtrip] = []
    cust_to_hub: dict[int, int] = {}
    cust_to_gtrip: dict[int, Gtrip] = {}

    # --- 1. compute cluster weights & hub seeding scores ---
    eps = 1e-6
    clusters = service_structure.clusters
    if not clusters:
        # 完全退化: 直接 fallback
        return phase1_generate_gtrips(data, tt, r_c=r_c)

    cluster_weight: dict[int, float] = {}
    for c in clusters:
        urgency = 0.0
        n = 0
        for j in c.customers:
            # avg 1/(slack + eps) under preferred hub
            if c.preferred_hub is not None and c.preferred_hub >= 0:
                slack = (data.customers[j].t_due
                         - data.customers[j].t_rel
                         - tt.t_hD[c.preferred_hub]
                         - tt.t_hj_K[c.preferred_hub][j])
                if slack > 0:
                    urgency += 1.0 / (slack + eps)
            n += 1
        urgency = urgency / max(1, n)
        cluster_weight[c.id] = (
            float(c.size)
            + c.total_weight / max(eps, data.l_K)
            + urgency
        )

    # opening penalty scaled gently to avoid dominating
    omega_h: dict[int, float] = {}
    n_cust = max(1, len(data.J))
    for h in data.H:
        omega_h[h] = (data.cost_H + data.cost_K_inv) / n_cust

    def seed_score(h: int) -> float:
        s = 0.0
        for c in clusters:
            s += cluster_weight[c.id] * c.hub_pref.get(h, 0.0)
        return s - omega_h[h]

    hub_order = sorted(data.H, key=lambda h: -seed_score(h))
    init_cfg = getattr(service_structure, "config", None)
    use_dynamic_hub_selection = bool(
        getattr(init_cfg, "init_dynamic_hub_selection", False)
    )

    def reachable_from_hub(h: int, unserved_set: set[int]) -> set[int]:
        res = set()
        for j in unserved_set:
            t_to = tt.t_hj_K[h][j]
            if 2 * t_to > data.e_K:
                continue
            if data.customers[j].t_rel + tt.t_hD[h] + t_to > data.customers[j].t_due + 1e-9:
                continue
            if t_to > r_c * data.e_K + 1e-9:
                continue
            res.add(j)
        return res

    def reachable_loose(h: int, unserved_set: set[int]) -> set[int]:
        """Drop the r_c filter."""
        res = set()
        for j in unserved_set:
            if 2 * tt.t_hj_K[h][j] <= data.e_K and \
               data.customers[j].t_rel + tt.t_hD[h] + tt.t_hj_K[h][j] \
               <= data.customers[j].t_due + 1e-9:
                res.add(j)
        return res

    def dynamic_hub_choice(unserved_set: set[int]) -> int | None:
        """Choose a hub using residual S-information and running-cost effort.

        The cost term is deliberately a movement-cost proxy rather than a
        response-time metric: UAV effort is the number of capacity-equivalent
        ferry trips times FC--hub travel, while UGV effort is a star-routing
        approximation over the customers currently covered by the hub.
        """
        inactive = [h for h in data.H if h not in active_hubs]
        if not inactive:
            return None

        strict_coverage = {h: reachable_from_hub(h, unserved_set) for h in inactive}
        loose_coverage = {h: reachable_loose(h, unserved_set) for h in inactive}
        eligibility = strict_coverage if any(strict_coverage.values()) else loose_coverage
        feasible_hubs = [h for h in inactive if eligibility[h]]
        if not feasible_hubs:
            return None

        # The construction loop below serves every battery/time-window-feasible
        # customer after a hub is selected (it intentionally drops r_c).  Score
        # that actual residual service set, while keeping r_c as the preferred
        # eligibility filter, so selection and execution use consistent scope.
        coverage = {h: loose_coverage[h] for h in feasible_hubs}

        residual_by_cluster: dict[int, float] = {}
        for c in clusters:
            remaining = sum(1 for j in c.customers if j in unserved_set)
            residual_by_cluster[c.id] = remaining / max(1, c.size)

        service_raw: dict[int, float] = {}
        coverage_raw: dict[int, float] = {}
        running_raw: dict[int, float] = {}
        for h in feasible_hubs:
            covered = coverage[h]
            service_raw[h] = sum(
                cluster_weight[c.id]
                * residual_by_cluster[c.id]
                * c.hub_pref.get(h, 0.0)
                for c in clusters
            )
            coverage_raw[h] = len(covered) / max(1, len(unserved_set))

            total_weight = sum(data.customers[j].weight for j in covered)
            uav_trips = max(1, int(math.ceil(total_weight / max(eps, data.l_D))))
            uav_cost = uav_trips * 2.0 * data.cost_D_trav * tt.t_hD[h]
            ugv_cost = sum(
                2.0 * data.cost_K_trav * tt.t_hj_K[h][j] for j in covered
            )
            running_raw[h] = (uav_cost + ugv_cost) / max(1, len(covered))

        def minmax(values: dict[int, float], h: int) -> float:
            lo, hi = min(values.values()), max(values.values())
            return 0.0 if hi - lo <= eps else (values[h] - lo) / (hi - lo)

        w_service = float(getattr(init_cfg, "init_service_weight", 0.5))
        w_coverage = float(getattr(init_cfg, "init_coverage_weight", 0.5))
        w_cost = float(getattr(init_cfg, "init_marginal_cost_weight", 0.25))

        scores: dict[int, float] = {}
        for h in feasible_hubs:
            scores[h] = (
                w_service * minmax(service_raw, h)
                + w_coverage * minmax(coverage_raw, h)
                - w_cost * minmax(running_raw, h)
            )

        choice = max(
            feasible_hubs,
            key=lambda h: (scores[h], len(coverage[h]), -running_raw[h], -int(h)),
        )
        diagnostics = getattr(service_structure, "diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics.setdefault("initialization_hub_decisions", []).append({
                "remaining_customers": len(unserved_set),
                "selected_hub": int(choice),
                "candidates": [
                    {
                        "hub": int(h),
                        "score": float(scores[h]),
                        "service": float(service_raw[h]),
                        "coverage": int(len(coverage[h])),
                        "marginal_running_cost": float(running_raw[h]),
                    }
                    for h in sorted(feasible_hubs)
                ],
            })
        return choice

    # --- 2. iterate hubs in service-guided order, build gtrips identically to baseline ---
    hub_iter = list(hub_order)
    while unserved and hub_iter:
        # 选下一个 service-preferred hub 中能覆盖最多 unserved 的
        # (与 baseline 一样, 用 coverage 作 tie-break, 但优先序由 seed_score 决定)
        candidate_h = dynamic_hub_choice(unserved) if use_dynamic_hub_selection else None
        if candidate_h is None and not use_dynamic_hub_selection:
            for h in hub_iter:
                if h in active_hubs:
                    continue
                cov = len(reachable_from_hub(h, unserved))
                if cov > 0:
                    candidate_h = h
                    break
            if candidate_h is None:
                # 放宽: 不用 r_c
                for h in hub_iter:
                    if h in active_hubs:
                        continue
                    cov = len(reachable_loose(h, unserved))
                    if cov > 0:
                        candidate_h = h
                        break
        if candidate_h is None:
            log.warning(f"service-guided Phase 1: no hub can cover remaining "
                        f"{len(unserved)} customers; will fallback")
            break

        best_h = candidate_h
        active_hubs.add(best_h)
        hub_iter = [h for h in hub_iter if h != best_h] + [best_h]  # cycle past

        # 复用 baseline Step 2-5 的 nearest-insert 逻辑
        while True:
            reachable_now = set()
            for j in unserved:
                if 2 * tt.t_hj_K[best_h][j] <= data.e_K and \
                   data.customers[j].t_rel + tt.t_hD[best_h] + tt.t_hj_K[best_h][j] \
                   <= data.customers[j].t_due + 1e-9:
                    reachable_now.add(j)
            if not reachable_now:
                break

            j_start = min(reachable_now, key=lambda j: tt.t_hj_K[best_h][j])
            new_gt = Gtrip(hub=best_h, seq=[j_start])
            data.customers[j_start].t_rel_H = data.customers[j_start].t_rel + tt.t_hD[best_h]
            compute_gtrip_window(new_gt, data, tt)
            if not new_gt.feasible:
                log.warning(
                    f"service-guided Phase 1: customer {j_start} single-stop infeasible, skip"
                )
                unserved.discard(j_start)
                continue
            gtrips.append(new_gt)
            cust_to_hub[j_start] = best_h
            cust_to_gtrip[j_start] = new_gt
            unserved.discard(j_start)

            while True:
                reachable_now = set()
                for j in unserved:
                    if 2 * tt.t_hj_K[best_h][j] <= data.e_K and \
                       data.customers[j].t_rel + tt.t_hD[best_h] + tt.t_hj_K[best_h][j] \
                       <= data.customers[j].t_due + 1e-9:
                        reachable_now.add(j)
                if not reachable_now:
                    break
                j_next = min(reachable_now, key=lambda j: tt.t_hj_K[best_h][j])
                data.customers[j_next].t_rel_H = \
                    data.customers[j_next].t_rel + tt.t_hD[best_h]

                inserted = False
                for gt in gtrips:
                    if gt.hub != best_h:
                        continue
                    best_pos, best_eta = None, float("inf")
                    orig_seq = gt.seq[:]
                    for pos in range(len(orig_seq) + 1):
                        trial_seq = orig_seq[:pos] + [j_next] + orig_seq[pos:]
                        new_eta = tt.gtrip_eta(gt.hub, trial_seq)
                        new_load = gt.load + data.customers[j_next].weight
                        if new_eta > data.e_K:
                            continue
                        if new_load > data.l_K:
                            continue
                        old_seq, old_load, old_eta = gt.seq, gt.load, gt.eta
                        gt.seq, gt.load = trial_seq, new_load
                        compute_gtrip_window(gt, data, tt)
                        ok = gt.feasible
                        gt.seq, gt.load, gt.eta = old_seq, old_load, old_eta
                        compute_gtrip_window(gt, data, tt)
                        if ok and new_eta < best_eta:
                            best_pos, best_eta = pos, new_eta
                    if best_pos is not None:
                        gt.seq.insert(best_pos, j_next)
                        gt.load += data.customers[j_next].weight
                        compute_gtrip_window(gt, data, tt)
                        cust_to_hub[j_next] = best_h
                        cust_to_gtrip[j_next] = gt
                        unserved.discard(j_next)
                        inserted = True
                        break
                if not inserted:
                    new_gt = Gtrip(hub=best_h, seq=[j_next])
                    compute_gtrip_window(new_gt, data, tt)
                    if not new_gt.feasible:
                        log.warning(
                            f"service-guided Phase 1: new gtrip for {j_next} infeasible, skip"
                        )
                        unserved.discard(j_next)
                        continue
                    gtrips.append(new_gt)
                    cust_to_hub[j_next] = best_h
                    cust_to_gtrip[j_next] = new_gt
                    unserved.discard(j_next)

    return gtrips, active_hubs, cust_to_hub


def four_phase_initialization_service_guided(
    data: DataContainer,
    tt: TravelTimes,
    service_structure,
    r_c: float = 1.0 / 3.0,
) -> Solution:
    """S-alns 入口: Phase 1 用 service-guided 版本; Phase 2/3/4 直接复用 baseline.

    任何阶段失败 -> 自动 fallback 到 four_phase_initialization.
    """
    try:
        log.info("S-alns Phase 1: service-guided gtrips...")
        gtrips, active_hubs, cust_to_hub = phase1_generate_gtrips_service_guided(
            data, tt, service_structure, r_c=r_c
        )
        log.info(f"  -> {len(gtrips)} Gtrips, {len(active_hubs)} active hubs, "
                 f"{len(cust_to_hub)}/{len(data.J)} customers placed")

        if not gtrips or len(cust_to_hub) == 0:
            log.warning("service-guided Phase 1 produced no gtrips, falling back to baseline")
            return four_phase_initialization(data, tt, r_c=r_c)

        log.info("Phase 2: Generate initial Groutes...")
        groutes = phase2_generate_groutes(gtrips, active_hubs, data, tt)
        log.info("Phase 3: Generate initial Atrips...")
        atrips = phase3_generate_atrips(gtrips, active_hubs, data, tt, cust_to_hub)
        log.info("Phase 4: Generate initial Aroutes...")
        aroutes = phase4_generate_aroutes(atrips, data, tt)

        sol = Solution(
            active_hubs=active_hubs,
            aroutes=aroutes,
            groutes=groutes,
            unserved=set(data.J) - set(cust_to_hub.keys()),
        )
        sol.rebuild_cust_indexes()
        refresh_all_timing(sol, data, tt)
        return sol
    except Exception as e:
        log.error(f"four_phase_initialization_service_guided crashed: {e}; "
                  f"falling back to baseline four_phase_initialization")
        return four_phase_initialization(data, tt, r_c=r_c)
