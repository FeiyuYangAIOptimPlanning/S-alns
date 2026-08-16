"""
destroy_operators.py — 9 个 destroy 算子 (§4.2.2)

命名 (按论文):
  1. RR   — Random Removal
  2. KRR  — k-Random Removal (k=3, 论文 Table 3 加粗)
  3. GR   — Gtrip Removal
  4. AR   — Atrip Removal
  5. ZR   — Zone Removal (ξ=9%, 论文 Table 3 加粗)
  6. WCR  — Worst Cost Removal
  7. HR   — Hub Removal
  8. SR   — Shaw Removal (β=(0.4,0.3,0.3))
  9. GRR  — Groute Removal
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
import logging
import math

from .structures import Solution, Atrip, Gtrip, Aroute, Groute
from .timing_subproblem import refresh_all_timing
from .costs import cost

if TYPE_CHECKING:
    from .data_loader import DataContainer
    from .travel_times import TravelTimes

log = logging.getLogger(__name__)

# --- optional removal cap (see _remove_customers_from_sol) ---
# None = no cap = original behavior. set_max_remove() turns it on.
MAX_REMOVE: "int | None" = None
_CAP_RNG = np.random.default_rng(0)


def set_max_remove(cap: "int | None", seed: "int | None" = None) -> None:
    """Cap the number of customers any destroy operator removes per call.

    cap=None disables (default). Bounds per-iteration repair work so the
    large-removal operators stay affordable on large instances.
    """
    global MAX_REMOVE, _CAP_RNG
    MAX_REMOVE = None if cap is None else int(cap)
    if seed is not None:
        _CAP_RNG = np.random.default_rng(int(seed))
    log.info(f"destroy MAX_REMOVE set to {MAX_REMOVE}")


# ===========================================================================
# 辅助: 把客户从一个解的 UAV + UGV 两层都移除
# ===========================================================================
def _remove_customers_from_sol(sol: Solution, customers: set[int]) -> None:
    """从双层结构中彻底移除一批客户, 放入 sol.unserved.

    If a removal cap is set via set_max_remove(), at most MAX_REMOVE customers
    are removed per call (a random subset). This bounds the per-iteration repair
    work (repair cost ~ #removed * K_gtrip * K_atrip * deepcopy), making the
    large-removal operators (ZR/GR/HR/CR/ACR/BRR/HMR) affordable on large
    instances while still letting them consolidate. Default (None) = no cap =
    original behavior.
    """
    if MAX_REMOVE is not None and len(customers) > MAX_REMOVE:
        arr = np.array(sorted(customers))
        customers = set(_CAP_RNG.choice(arr, size=MAX_REMOVE, replace=False).tolist())
    for a in list(sol.all_atrips()):
        a.pkgs = [j for j in a.pkgs if j not in customers]
    for g in list(sol.all_gtrips()):
        g.seq = [j for j in g.seq if j not in customers]
    sol.unserved.update(customers)
    sol.clean_empty()
    sol.rebuild_cust_indexes()


# ===========================================================================
# 1. RR — Random Removal
# ===========================================================================
def random_removal(sol: Solution, data: "DataContainer", tt: "TravelTimes",
                   rng: np.random.Generator) -> None:
    served = [j for j in data.J if j not in sol.unserved]
    if not served:
        return
    j = int(rng.choice(served))
    _remove_customers_from_sol(sol, {j})
    refresh_all_timing(sol, data, tt)


# ===========================================================================
# 2. KRR — k-Random Removal (k=3)
# ===========================================================================
def k_random_removal(sol: Solution, data: "DataContainer", tt: "TravelTimes",
                     rng: np.random.Generator, k: int = 3) -> None:
    served = [j for j in data.J if j not in sol.unserved]
    if not served:
        return
    k_eff = min(k, len(served))
    chosen = set(int(x) for x in rng.choice(served, size=k_eff, replace=False))
    _remove_customers_from_sol(sol, chosen)
    refresh_all_timing(sol, data, tt)


# ===========================================================================
# 3. GR — Gtrip Removal
# ===========================================================================
def gtrip_removal(sol: Solution, data: "DataContainer", tt: "TravelTimes",
                  rng: np.random.Generator) -> None:
    all_gts = sol.all_gtrips()
    if not all_gts:
        return
    idx = int(rng.integers(0, len(all_gts)))
    gt = all_gts[idx]
    _remove_customers_from_sol(sol, set(gt.seq))
    refresh_all_timing(sol, data, tt)


# ===========================================================================
# 4. AR — Atrip Removal
# ===========================================================================
def atrip_removal(sol: Solution, data: "DataContainer", tt: "TravelTimes",
                  rng: np.random.Generator) -> None:
    all_ats = sol.all_atrips()
    if not all_ats:
        return
    idx = int(rng.integers(0, len(all_ats)))
    a = all_ats[idx]
    _remove_customers_from_sol(sol, set(a.pkgs))
    refresh_all_timing(sol, data, tt)


# ===========================================================================
# 5. ZR — Zone Removal (ξ=9%)
# ===========================================================================
def zone_removal(sol: Solution, data: "DataContainer", tt: "TravelTimes",
                 rng: np.random.Generator, xi: float = 0.09) -> None:
    coords = [data.customers[j].coord for j in data.J]
    xs, ys = zip(*coords)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    W, H = xmax - xmin, ymax - ymin
    side = math.sqrt(xi)
    w_rect, h_rect = W * side, H * side
    x0 = rng.uniform(xmin, max(xmin, xmax - w_rect))
    y0 = rng.uniform(ymin, max(ymin, ymax - h_rect))
    removed = {j for j in data.J
               if (x0 - 1e-9 <= data.customers[j].coord[0] <= x0 + w_rect + 1e-9
                   and y0 - 1e-9 <= data.customers[j].coord[1] <= y0 + h_rect + 1e-9
                   and j not in sol.unserved)}
    if not removed:
        return
    _remove_customers_from_sol(sol, removed)
    refresh_all_timing(sol, data, tt)


# ===========================================================================
# 6. WCR — Worst Cost Removal
# ===========================================================================
def worst_cost_removal(sol: Solution, data: "DataContainer", tt: "TravelTimes",
                       rng: np.random.Generator) -> None:
    """Δ_j = cost(S) - cost(S \\ j). 选 argmax Δ_j 移除."""
    from copy import deepcopy
    served = [j for j in data.J if j not in sol.unserved]
    if not served:
        return
    base_cost = cost(sol, data, tt)
    best_j, best_delta = None, -float("inf")
    # 限制候选数量以加速
    candidates = served if len(served) <= 30 else \
        [int(x) for x in rng.choice(served, size=30, replace=False)]
    for j in candidates:
        sol_try = deepcopy(sol)
        _remove_customers_from_sol(sol_try, {j})
        refresh_all_timing(sol_try, data, tt)
        d = base_cost - cost(sol_try, data, tt)
        if d > best_delta:
            best_delta = d
            best_j = j
    if best_j is not None:
        _remove_customers_from_sol(sol, {best_j})
        refresh_all_timing(sol, data, tt)


# ===========================================================================
# 7. HR — Hub Removal
# ===========================================================================
def hub_removal(sol: Solution, data: "DataContainer", tt: "TravelTimes",
                rng: np.random.Generator) -> None:
    if not sol.active_hubs:
        return
    h = int(rng.choice(list(sol.active_hubs)))
    customers = {j for j, hh in sol.cust_to_hub.items() if hh == h}
    if not customers:
        return
    _remove_customers_from_sol(sol, customers)
    refresh_all_timing(sol, data, tt)


# ===========================================================================
# 8. SR — Shaw Removal (β = (0.4, 0.3, 0.3))
# ===========================================================================
def shaw_removal(sol: Solution, data: "DataContainer", tt: "TravelTimes",
                 rng: np.random.Generator,
                 beta: tuple[float, float, float] = (0.4, 0.3, 0.3)) -> None:
    served = [j for j in data.J if j not in sol.unserved]
    if len(served) < 2:
        return
    j = int(rng.choice(served))
    b1, b2, b3 = beta
    best_jp, best_sim = None, float("inf")
    for jp in served:
        if jp == j:
            continue
        sim = (b1 * tt.t_jj_K[j][jp]
               + b2 * abs(data.customers[j].t_rel - data.customers[jp].t_rel)
               + b3 * abs(data.customers[j].t_due - data.customers[jp].t_due))
        if sim < best_sim:
            best_sim = sim
            best_jp = jp
    removed = {j}
    if best_jp is not None:
        removed.add(best_jp)
    _remove_customers_from_sol(sol, removed)
    refresh_all_timing(sol, data, tt)


# ===========================================================================
# 9. GRR — Groute Removal
# ===========================================================================
def groute_removal(sol: Solution, data: "DataContainer", tt: "TravelTimes",
                   rng: np.random.Generator) -> None:
    if not sol.groutes:
        return
    idx = int(rng.integers(0, len(sol.groutes)))
    gr = sol.groutes[idx]
    customers: set[int] = set()
    for g in gr.trips:
        customers.update(g.seq)
    if not customers:
        return
    _remove_customers_from_sol(sol, customers)
    refresh_all_timing(sol, data, tt)


# ===========================================================================
# 注册表
# ===========================================================================
DESTROY_OPS = {
    "RR":  random_removal,
    "KRR": k_random_removal,
    "GR":  gtrip_removal,
    "AR":  atrip_removal,
    "ZR":  zone_removal,
    "WCR": worst_cost_removal,
    "HR":  hub_removal,
    "SR":  shaw_removal,
    "GRR": groute_removal,
}
DESTROY_NAMES = list(DESTROY_OPS.keys())


# ===========================================================================
# Service-Structure guided destroy operators (S-alns)
# ---------------------------------------------------------------------------
# Each has signature: (sol, data, tt, rng, service_structure=None, service_cfg=None)
# When service_structure is None -> fall back to a safe baseline operator.
# Registered separately in DESTROY_OPS_STRUCTURE so they do NOT pollute the
# baseline DESTROY_OPS / DESTROY_NAMES.
# ===========================================================================
def _served_customers_in_cluster(sol: Solution, cluster) -> set[int]:
    """Filter cluster.customers to those currently served (not in sol.unserved)."""
    return {j for j in cluster.customers if j not in sol.unserved}


def cluster_removal(sol: Solution, data, tt, rng,
                    service_structure=None, service_cfg=None) -> None:
    """Remove all currently-served customers belonging to one random service cluster."""
    if service_structure is None or not service_structure.clusters:
        random_removal(sol, data, tt, rng)
        return
    clusters = [c for c in service_structure.clusters
                if _served_customers_in_cluster(sol, c)]
    if not clusters:
        random_removal(sol, data, tt, rng)
        return
    c = clusters[int(rng.integers(0, len(clusters)))]
    victims = _served_customers_in_cluster(sol, c)
    _remove_customers_from_sol(sol, victims)
    refresh_all_timing(sol, data, tt)


def adjacent_cluster_removal(sol: Solution, data, tt, rng,
                              service_structure=None, service_cfg=None) -> None:
    """Pick one cluster and remove it together with its graph neighbors."""
    if service_structure is None or not service_structure.clusters:
        random_removal(sol, data, tt, rng)
        return
    clusters = service_structure.clusters
    if not clusters:
        random_removal(sol, data, tt, rng)
        return
    seed_c = clusters[int(rng.integers(0, len(clusters)))]
    neighbor_ids = list(service_structure.cluster_graph.get(seed_c.id, []))
    target_ids = {seed_c.id, *neighbor_ids}
    victims: set[int] = set()
    cid_to_c = {c.id: c for c in clusters}
    for cid in target_ids:
        c = cid_to_c.get(cid)
        if c is None:
            continue
        victims |= _served_customers_in_cluster(sol, c)
    if not victims:
        random_removal(sol, data, tt, rng)
        return
    _remove_customers_from_sol(sol, victims)
    refresh_all_timing(sol, data, tt)


def hub_mismatch_cluster_removal(sol: Solution, data, tt, rng,
                                  service_structure=None, service_cfg=None) -> None:
    """For each cluster, find the dominant current hub; pick the cluster with
    maximal (1 - P_r[dominant_hub]) and remove it."""
    if service_structure is None or not service_structure.clusters:
        random_removal(sol, data, tt, rng)
        return
    clusters = service_structure.clusters
    best_cluster = None
    best_score = -1.0
    for c in clusters:
        served = _served_customers_in_cluster(sol, c)
        if not served:
            continue
        # current dominant hub for this cluster
        hub_count: dict[int, int] = {}
        for j in served:
            h = sol.cust_to_hub.get(j)
            if h is None:
                continue
            hub_count[h] = hub_count.get(h, 0) + 1
        if not hub_count:
            continue
        dominant_hub = max(hub_count, key=hub_count.get)
        pref = c.hub_pref.get(dominant_hub, 0.0)
        mismatch = 1.0 - pref
        if mismatch > best_score:
            best_score = mismatch
            best_cluster = c
    if best_cluster is None:
        random_removal(sol, data, tt, rng)
        return
    victims = _served_customers_in_cluster(sol, best_cluster)
    if not victims:
        random_removal(sol, data, tt, rng)
        return
    _remove_customers_from_sol(sol, victims)
    refresh_all_timing(sol, data, tt)


def boundary_regret_removal(sol: Solution, data, tt, rng,
                              service_structure=None, service_cfg=None) -> None:
    """Remove customers near the hub-preference boundary (small p_top1 - p_top2)."""
    if service_structure is None:
        shaw_removal(sol, data, tt, rng)
        return
    n_remove_default = max(2, len(data.J) // 10)
    served = [j for j in data.J if j not in sol.unserved]
    if len(served) < 2:
        return
    cust_id_to_idx = service_structure.customer_id_to_idx
    gaps: list[tuple[float, int]] = []
    for j in served:
        idx = cust_id_to_idx.get(j)
        if idx is None:
            continue
        row = service_structure.hub_pref[idx]
        # top1 / top2
        if row.size < 2:
            continue
        top2 = np.partition(-row, 1)[:2]
        p1, p2 = -top2[0], -top2[1]
        gaps.append((float(p1 - p2), j))
    if not gaps:
        random_removal(sol, data, tt, rng)
        return
    gaps.sort(key=lambda x: x[0])
    n_take = min(n_remove_default, len(gaps))
    victims = {gaps[i][1] for i in range(n_take)}
    _remove_customers_from_sol(sol, victims)
    refresh_all_timing(sol, data, tt)


def inactive_hub_probe_removal(sol: Solution, data, tt, rng,
                                service_structure=None, service_cfg=None) -> None:
    """Find an inactive hub that some cluster strongly prefers; remove customers
    of that cluster (or the cluster's currently-used hub neighborhood) to give
    repair a chance to open the inactive hub.

    Phase-2 enhancement (atomic hub seeding): after removing victims, force-seed
    the chosen inactive hub by placing the cluster medoid customer directly on
    it (NEW_GT_NEW_GR + NEW_AT_NEW_AR). This pays the open-hub cost (cost_H +
    cost_K_inv + cost_D_inv) up front so the main repair phase sees the hub as
    already-open and can cheaply insert remaining victims via EXIST_GT slots,
    instead of facing the prohibitive "open a new hub for one customer" cost
    each time. This is a partial form of atomic cluster batch insertion.
    """
    if service_structure is None or not service_structure.clusters:
        random_removal(sol, data, tt, rng)
        return
    inactive_hubs = [h for h in data.H if h not in sol.active_hubs]
    if not inactive_hubs:
        random_removal(sol, data, tt, rng)
        return
    clusters = service_structure.clusters
    # 对每个 cluster, 找它最偏好的 inactive hub (按 hub_pref)
    best = None  # (pref_score, cluster, victims, target_hub)
    for c in clusters:
        pref_inactive = [(c.hub_pref.get(h, 0.0), h) for h in inactive_hubs]
        if not pref_inactive:
            continue
        pref_inactive.sort(reverse=True)
        top_pref, top_h = pref_inactive[0]
        if top_pref <= 0:
            continue
        victims = _served_customers_in_cluster(sol, c)
        if not victims:
            continue
        if best is None or top_pref > best[0]:
            best = (top_pref, c, victims, top_h)
    if best is None:
        random_removal(sol, data, tt, rng)
        return
    _, target_cluster, victims, target_hub = best
    _remove_customers_from_sol(sol, victims)
    refresh_all_timing(sol, data, tt)

    # Phase-2 atomic seeding: pick the cluster medoid (if it's among the victims
    # and the seed is time/battery-feasible on the target hub), place it on
    # target_hub directly, opening that hub. Then refresh and let main repair
    # finish the cluster.
    medoid = target_cluster.medoid
    if medoid not in victims:
        return  # main repair will try inactive hub via case (c) anyway
    # Check feasibility on target_hub (battery + time slack)
    w = data.customers[medoid].weight
    if w > data.l_K + 1e-9:
        return
    # Two-way UGV battery
    if 2 * tt.t_hj_K[target_hub][medoid] > data.e_K + 1e-9:
        return
    # Two-way UAV battery (FC <-> hub)
    if 2 * tt.t_hD[target_hub] > data.e_D + 1e-9:
        return
    # Time slack
    slack = (data.customers[medoid].t_due - data.customers[medoid].t_rel
             - tt.t_hD[target_hub] - tt.t_hj_K[target_hub][medoid])
    if slack <= 0:
        return
    # Seed: NEW_GT_NEW_GR + NEW_AT_NEW_AR on target_hub for medoid
    new_gt = Gtrip(hub=target_hub, seq=[medoid])
    new_gr = Groute(hub=target_hub, trips=[new_gt])
    sol.groutes.append(new_gr)
    new_at = Atrip(hub=target_hub, pkgs=[medoid])
    new_ar = Aroute(trips=[new_at])
    sol.aroutes.append(new_ar)
    sol.active_hubs.add(target_hub)
    sol.unserved.discard(medoid)
    sol.rebuild_cust_indexes()
    refresh_all_timing(sol, data, tt)
    # If timing broke for some reason, undo (shouldn't happen since we checked)
    if not sol.feasible:
        # roll back the seed
        sol.groutes.remove(new_gr)
        sol.aroutes.remove(new_ar)
        sol.unserved.add(medoid)
        sol.active_hubs.discard(target_hub) if not any(
            g.hub == target_hub for g in sol.all_gtrips()
        ) and not any(a.hub == target_hub for a in sol.all_atrips()) else None
        sol.rebuild_cust_indexes()
        refresh_all_timing(sol, data, tt)


DESTROY_OPS_STRUCTURE = {
    "CR":  cluster_removal,
    "ACR": adjacent_cluster_removal,
    "HMR": hub_mismatch_cluster_removal,
    "BRR": boundary_regret_removal,
    "IHP": inactive_hub_probe_removal,
}
DESTROY_NAMES_STRUCTURE = list(DESTROY_OPS_STRUCTURE.keys())
