"""
service_structure.py — Adaptive Service-Structure Learning module.

无监督服务结构学习: 将客户嵌入到高维 service representation
(空间 / 时间 / 包裹重量 / hub 偏好 / 激活成本), 通过 generalized service-cost
distance 做自适应 k-medoids 聚类, 得到 service clusters 与 demand cluster graph.

学到的结构作为 *soft* heuristic guidance 注入 ALNS, 不做硬分解, 不修改
optimization model / objective / constraints / validator.

论文对应: §4 (Service-Structure Learning) + Algorithm 1.

参考字段名 (与 src/travel_times.py 实际一致):
    tt.t_hD[h]        UAV 单程 FC -> hub_h 时间
    tt.t_hj_K[h][j]   UGV hub_h -> customer_j 时间
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, TYPE_CHECKING
from copy import deepcopy
import math
import json
import os
import logging
import time

import numpy as np

if TYPE_CHECKING:
    from .data_loader import DataContainer
    from .travel_times import TravelTimes

log = logging.getLogger(__name__)


# ===========================================================================
# Config / data classes
# ===========================================================================
@dataclass
class ServiceStructureConfig:
    """S-alns 模块配置.

    Note:
        - alpha_* 控制 customer-hub service-cost distance 各分量权重 (§4.2 / Eq.6)
        - lambda_* 控制 customer-customer service dissimilarity 各分量权重 (Eq.9)
        - omega_* 控制 service-structure score 各项 (Eq.12)
    """
    enabled: bool = False
    auto_cluster: bool = True
    q_min: int = 2
    q_max_mode: str = "sqrt_n"  # "sqrt_n" 或 "fixed"
    q_fixed: Optional[int] = None
    # Adaptive-q search over [q_min, q_max]:
    #   "full"        — evaluate every q (optionally in parallel)
    #   "golden"      — golden-section (binary-style) search, ~O(log) evals (assumes
    #                   the SSS score is unimodal in q; correct for monotonic too)
    #   "coarse2fine" — parallel coarse grid then refine; robust + parallel-friendly
    q_search_mode: str = "full"
    q_search_jobs: int = 1       # threads for evaluating q candidates (k-medoids is numpy-bound)
    top_m_hubs: int = 3
    graph_k_neighbors: int = 2
    infeasible_penalty: float = 1e6
    epsilon: float = 1e-6
    random_seed: int = 42
    max_kmedoids_iter: int = 50

    # customer-hub distance weights (Eq.6)
    alpha_uav: float = 1.0
    alpha_ugv: float = 1.0
    alpha_time: float = 1.0
    alpha_weight: float = 1.0
    alpha_open: float = 1.0

    # customer-customer distance weights (Eq.9)
    lambda_space: float = 1.0
    lambda_release: float = 1.0
    lambda_due: float = 1.0
    lambda_weight: float = 1.0
    lambda_hubpref: float = 1.0

    # adaptive clustering score weights (Eq.12)
    omega_compactness: float = 1.0
    omega_separation: float = 1.0
    omega_hub_purity: float = 1.0
    omega_risk: float = 1.0
    omega_fragmentation: float = 0.5

    # D_jh normalization scope (option B from sensitivity analysis)
    #   "joint"      : 全矩阵 minmax (向后兼容, 但在 verylowHubCost 上 uav_effort 通杀)
    #   "per_row"    : 每个 customer 独立 minmax. 推荐用于 hub UAV-ferry-time 差异大的实例
    djh_norm_scope: str = "joint"

    # Phase-1 initialization hub selection.  The legacy implementation uses
    # one static ranking computed before any customer is assigned.  The
    # dynamic option re-scores inactive hubs against residual demand, combining
    # learned service preference, feasible coverage, and a running-cost proxy.
    # It is opt-in so previously reported experiments remain reproducible.
    init_dynamic_hub_selection: bool = False
    init_service_weight: float = 0.5
    init_coverage_weight: float = 0.5
    init_marginal_cost_weight: float = 0.25


@dataclass
class ServiceCluster:
    """一个 service-compatible 客户簇."""
    id: int
    customers: list[int]                # customer ids (data.J 视角)
    medoid: int                          # 也是 customer id
    size: int
    total_weight: float
    spatial_center: tuple[float, float]
    rel_center: float                    # 平均 release time
    due_center: float                    # 平均 due time
    hub_pref: dict[int, float]          # hub_id -> avg preference within cluster
    preferred_hub: int                   # arg max hub_pref; -1 if cluster has no feasible hub
    service_radius: float                # cluster 内最大 D_cust 到 medoid (归一化前)
    time_risk: float                     # TimeRisk(C_r) under preferred hub
    capacity_risk: float                 # CapRisk(C_r): max(0, total_w / l_K - 1)


@dataclass
class ServiceStructure:
    """完整的 service-structure learning 输出, 作为 ALNS guidance 容器."""
    # ID <-> matrix index 映射 (强制维护, 即使当前实例 id 与索引一致)
    customer_ids: list[int]
    hub_ids: list[int]
    customer_id_to_idx: dict[int, int]
    idx_to_customer_id: dict[int, int]
    hub_id_to_idx: dict[int, int]
    idx_to_hub_id: dict[int, int]

    # 距离/偏好矩阵 (矩阵索引)
    D_jh: np.ndarray                     # shape (n_cust, n_hub)
    D_cust: np.ndarray                   # shape (n_cust, n_cust), 对角 0, 对称
    hub_pref: np.ndarray                 # shape (n_cust, n_hub), 每行 sum~=1
    top_hubs: dict[int, list[int]]       # customer_id -> [hub_id, ...]  长度 top_m_hubs

    # 聚类结果
    cluster_id_of_customer: dict[int, int]   # customer_id -> cluster_id
    clusters: list[ServiceCluster]
    cluster_graph: dict[int, list[int]]      # cluster_id -> [neighbor cluster_id, ...]

    # 自适应 q 选择诊断
    selected_q: int
    score_by_q: dict[int, float]
    diagnostics: dict[str, Any]
    config: Optional[ServiceStructureConfig] = None


def shuffle_service_structure_customer_labels(
    ss: ServiceStructure,
    seed: int,
) -> tuple[ServiceStructure, dict[int, int]]:
    """Return an internally isomorphic but physically uninformative structure.

    A seed-specific bijection detaches every customer-indexed item in ``ss``
    from the physical customer to which it was learned. Cluster sizes, graph
    topology, selected q, score diagnostics, operator pools, and all numerical
    distributions are preserved. This is the placebo control used to test
    whether an ablation benefit comes from the information in S rather than
    from adding more operators or giving them different initial weights.

    The returned mapping is ``old_customer_id -> shuffled_customer_id`` and
    should be exported with the run artifacts for reproducibility.
    """
    customer_ids = [int(j) for j in ss.customer_ids]
    hub_ids = [int(h) for h in ss.hub_ids]
    rng = np.random.default_rng(int(seed))

    def _derangement(items: list[int]) -> list[int]:
        if len(items) <= 1:
            return items[:]
        for _ in range(1000):
            candidate = [int(item) for item in rng.permutation(items)]
            if all(source != target for source, target in zip(items, candidate)):
                return candidate
        return items[1:] + items[:1]

    # Derange both customer and physical-hub labels. Customer-only relabeling
    # would leave the cluster-level hub scores used by initialization unchanged.
    shuffled_ids = _derangement(customer_ids)
    shuffled_hubs = _derangement(hub_ids)
    permutation = dict(zip(customer_ids, shuffled_ids))
    hub_permutation = dict(zip(hub_ids, shuffled_hubs))

    if len(set(permutation.values())) != len(customer_ids):
        raise RuntimeError("service-structure shuffle did not produce a bijection")

    shuffled = deepcopy(ss)
    source_idx = np.array([ss.customer_id_to_idx[j] for j in customer_ids], dtype=int)
    target_idx = np.array(
        [ss.customer_id_to_idx[permutation[j]] for j in customer_ids], dtype=int
    )

    source_h_idx = np.array([ss.hub_id_to_idx[h] for h in hub_ids], dtype=int)
    target_h_idx = np.array(
        [ss.hub_id_to_idx[hub_permutation[h]] for h in hub_ids], dtype=int
    )

    # Attach each learned customer-hub row/column to the wrong physical ids.
    shuffled.D_jh[np.ix_(target_idx, target_h_idx)] = ss.D_jh[
        np.ix_(source_idx, source_h_idx)
    ]
    shuffled.hub_pref[np.ix_(target_idx, target_h_idx)] = ss.hub_pref[
        np.ix_(source_idx, source_h_idx)
    ]
    shuffled.D_cust[np.ix_(target_idx, target_idx)] = ss.D_cust[
        np.ix_(source_idx, source_idx)
    ]
    shuffled.top_hubs = {
        permutation[source]: [hub_permutation[int(h)] for h in ss.top_hubs.get(source, [])]
        for source in customer_ids
    }

    shuffled_clusters: list[ServiceCluster] = []
    for cluster in ss.clusters:
        clone = deepcopy(cluster)
        clone.customers = [permutation[int(j)] for j in cluster.customers]
        clone.medoid = permutation[int(cluster.medoid)]
        clone.hub_pref = {
            hub_permutation[int(h)]: float(value)
            for h, value in cluster.hub_pref.items()
        }
        clone.preferred_hub = (
            hub_permutation[int(cluster.preferred_hub)]
            if int(cluster.preferred_hub) in hub_permutation else -1
        )
        # All aggregate attributes deliberately remain those of the source
        # cluster: the same S object is now attached to the wrong customers.
        shuffled_clusters.append(clone)
    shuffled.clusters = shuffled_clusters
    shuffled.cluster_id_of_customer = {
        int(j): int(cluster.id)
        for cluster in shuffled_clusters
        for j in cluster.customers
    }

    assigned = set(shuffled.cluster_id_of_customer)
    if assigned != set(customer_ids):
        raise RuntimeError("shuffled clusters do not cover each customer exactly once")

    shuffled.diagnostics = deepcopy(ss.diagnostics)
    shuffled.diagnostics["shuffle_control"] = {
        "enabled": True,
        "seed": int(seed),
        "customer_permutation": {str(k): int(v) for k, v in permutation.items()},
        "hub_permutation": {str(k): int(v) for k, v in hub_permutation.items()},
    }
    return shuffled, permutation


# ===========================================================================
# Small helpers (Q1: avoid scattering field accesses)
# ===========================================================================
def _get_hub_customer_time_matrix(tt: "TravelTimes", n_hub: int, n_cust: int) -> np.ndarray:
    """从 TravelTimes 抽出 hub->customer 时间矩阵 (numpy)."""
    return np.array(tt.t_hj_K, dtype=float)  # shape (n_hub, n_cust)


def _get_hub_uav_time_vector(tt: "TravelTimes", n_hub: int) -> np.ndarray:
    """单程 FC -> hub UAV 时间向量."""
    return np.array(tt.t_hD, dtype=float)


def _minmax_norm(arr: np.ndarray, eps: float = 1e-12,
                 mask_finite: Optional[np.ndarray] = None,
                 scope: str = "joint") -> np.ndarray:
    """Min-max 归一到 [0,1].

    Args:
        arr: 输入 (n_cust, n_hub) 或任意 shape
        eps: 数值稳定项
        mask_finite: 与 arr 同形, True = 参与归一化的位置.
                     未参与的位置保持原值; 计算 min/max 时只看 True 子集.
        scope:
            - "joint" (默认, 向后兼容): 在整个 arr 的 finite 子集上找一对 (min, max),
              所有元素用同一对归一化. uav_effort 这种 per-hub-only 信号会跨 customer 共享
              同一对 min/max, 导致一旦 H_j 的 uav_effort 是全局最低, 它对所有 customer
              都看起来"完美" → 退化为 H_j 独占.
            - "per_row": 每个 customer 独立 minmax. 对一行内, 在该行 mask_finite 上找
              row_min/row_max, 归一化只在该行进行. 这样 uav_effort 因为每行都是
              同样的 (n_hub,) 向量, 归一化后仍是同一向量, 但**不再相对于 ugv_effort
              享有全局量纲优势**. ugv_effort 每行独立做 minmax → 充分利用 per-customer
              地理差异. 在 verylowHubCost 实例上实测可破解 H2 独占.
    """
    if scope == "joint":
        if mask_finite is None:
            finite_vals = arr[np.isfinite(arr)]
        else:
            finite_vals = arr[mask_finite & np.isfinite(arr)]
        if finite_vals.size == 0:
            return np.zeros_like(arr)
        lo, hi = float(finite_vals.min()), float(finite_vals.max())
        if hi - lo < eps:
            return np.zeros_like(arr)
        out = (arr - lo) / (hi - lo + eps)
        out = np.clip(out, 0.0, 1.0)
        return out

    elif scope == "per_row":
        # 要求 arr 是 2D
        if arr.ndim != 2:
            # 退回 joint 行为, 但提示开发者
            return _minmax_norm(arr, eps=eps, mask_finite=mask_finite, scope="joint")
        out = np.zeros_like(arr, dtype=float)
        n_rows = arr.shape[0]
        for j in range(n_rows):
            row = arr[j]
            if mask_finite is None:
                mask_j = np.isfinite(row)
            else:
                mask_j = mask_finite[j] & np.isfinite(row)
            if not mask_j.any():
                # row 没有 finite 值, 归零
                out[j, :] = 0.0
                continue
            row_min = float(row[mask_j].min())
            row_max = float(row[mask_j].max())
            if row_max - row_min < eps:
                # 行内常量, 没有 differentiating signal -> 归零
                out[j, :] = 0.0
                continue
            normed = (row - row_min) / (row_max - row_min + eps)
            # mask_finite=False 处不应被引用, 但保留计算结果(后续 D_jh 加 inf_pen 处理)
            out[j, :] = np.clip(normed, 0.0, 1.0)
        return out

    else:
        raise ValueError(f"_minmax_norm: unknown scope '{scope}'")


def _component_stats(arr: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> dict[str, float]:
    """Diagnostic stats for a distance component (min/max/mean/std on finite)."""
    if mask is not None:
        vals = arr[mask & np.isfinite(arr)]
    else:
        vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "count": int(vals.size),
    }


# ===========================================================================
# Step A: customer-hub service-cost distance (Eq.6)
# ===========================================================================
def _build_D_jh(data: "DataContainer", tt: "TravelTimes",
                cfg: ServiceStructureConfig,
                customer_ids: list[int], hub_ids: list[int],
                cust_id_to_idx: dict[int, int], hub_id_to_idx: dict[int, int]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """计算 D_{jh}: customer j 通过 hub h 服务的 service-cost dissimilarity.

    Returns:
        D_jh: (n_cust, n_hub) ndarray
        feasibility_mask: (n_cust, n_hub) bool, True = (j, h) 服务可行
        slack: (n_cust, n_hub) float ndarray, s_jh = t_due - t_rel - t_hD - t_hj
               (infeasible 处可能 <= 0; 调用方需要时再 clip / mask)
        diagnostics: 各分量统计
    """
    n_cust = len(customer_ids)
    n_hub = len(hub_ids)
    eps = cfg.epsilon
    inf_pen = cfg.infeasible_penalty

    t_hD = _get_hub_uav_time_vector(tt, n_hub)             # (n_hub,)
    t_hj = _get_hub_customer_time_matrix(tt, n_hub, n_cust)  # (n_hub, n_cust)

    # 1. UAV ferry effort:  2 * c_D_trav * t_hD[h]    -> broadcast to (n_cust, n_hub)
    uav_effort = (2.0 * data.cost_D_trav * t_hD)[np.newaxis, :]  # (1, n_hub)
    uav_effort = np.broadcast_to(uav_effort, (n_cust, n_hub)).astype(float)

    # 2. UGV access effort: 2 * c_K_trav * t_hj[h, j]
    # 转置成 (n_cust, n_hub):
    ugv_effort = 2.0 * data.cost_K_trav * t_hj.T  # (n_cust, n_hub)

    # 3. Time slack risk:  s_jh = t_due - t_rel - t_hD - t_hj
    t_rel = np.array([data.customers[customer_ids[i]].t_rel for i in range(n_cust)])
    t_due = np.array([data.customers[customer_ids[i]].t_due for i in range(n_cust)])
    weights = np.array([data.customers[customer_ids[i]].weight for i in range(n_cust)])

    slack = (t_due[:, np.newaxis]
             - t_rel[:, np.newaxis]
             - t_hD[np.newaxis, :]
             - t_hj.T)
    # slack > 0 时风险 = 1/(slack+eps); 否则视为 infeasible
    time_risk = np.where(slack > 0, 1.0 / (slack + eps), np.inf)

    # 4. Weight risk: weight_j / l_K  (与 hub 无关, 但保留矩阵形)
    w_risk = (weights / data.l_K)[:, np.newaxis]
    w_risk = np.broadcast_to(w_risk, (n_cust, n_hub)).astype(float)

    # 5. Opening / activation share:
    #    open_share_h = (c_H + c_K_inv) / max(1, feasible_count_for_h)
    #    先用初步 feasibility (UGV battery + UAV battery + slack > 0) 估 count.
    ugv_batt_ok = (2.0 * t_hj.T) <= data.e_K + 1e-9                # (n_cust, n_hub)
    uav_batt_ok = (2.0 * t_hD)[np.newaxis, :] <= data.e_D + 1e-9    # (1, n_hub)
    uav_batt_ok = np.broadcast_to(uav_batt_ok, (n_cust, n_hub))
    time_ok = slack > 0                                              # (n_cust, n_hub)

    feasibility_mask = ugv_batt_ok & uav_batt_ok & time_ok
    feasible_count_per_hub = feasibility_mask.sum(axis=0)            # (n_hub,)
    feasible_count_per_hub = np.maximum(feasible_count_per_hub, 1)
    open_share = (data.cost_H + data.cost_K_inv) / feasible_count_per_hub  # (n_hub,)
    open_share_mat = np.broadcast_to(open_share[np.newaxis, :], (n_cust, n_hub)).astype(float)

    # ----- Diagnostics: raw components -----
    diag: dict[str, Any] = {
        "raw_components": {
            "uav_effort": _component_stats(uav_effort),
            "ugv_effort": _component_stats(ugv_effort),
            "time_risk": _component_stats(np.where(feasibility_mask, time_risk, np.nan)),
            "weight_risk": _component_stats(w_risk),
            "open_share": _component_stats(open_share_mat),
        },
        "feasibility_mask_density": float(feasibility_mask.mean()),
        "feasible_count_per_hub": feasible_count_per_hub.tolist(),
    }

    # ----- 归一化各分量到 [0,1] -----
    # 由 cfg.djh_norm_scope 控制: "joint" (向后兼容, 全矩阵 minmax)
    # 或 "per_row" (每 customer 独立 minmax, 推荐用于 hub-time 差异大的实例)
    norm_scope = getattr(cfg, "djh_norm_scope", "joint")
    n_uav = _minmax_norm(uav_effort, eps, scope=norm_scope)
    n_ugv = _minmax_norm(ugv_effort, eps, scope=norm_scope)
    # time_risk 在 infeasible 处是 inf -> 归一化时只看 feasible 子集
    n_time = _minmax_norm(np.where(feasibility_mask, time_risk, np.nan), eps,
                          mask_finite=feasibility_mask, scope=norm_scope)
    n_w = _minmax_norm(w_risk, eps, scope=norm_scope)
    n_open = _minmax_norm(open_share_mat, eps, scope=norm_scope)

    D = (cfg.alpha_uav * n_uav
         + cfg.alpha_ugv * n_ugv
         + cfg.alpha_time * n_time
         + cfg.alpha_weight * n_w
         + cfg.alpha_open * n_open)

    # ----- Infeasibility penalty: 任一不可行 -> 加大 -----
    D = D + np.where(feasibility_mask, 0.0, inf_pen)

    diag["normalized_components"] = {
        "uav_effort": _component_stats(n_uav),
        "ugv_effort": _component_stats(n_ugv),
        "time_risk": _component_stats(np.where(feasibility_mask, n_time, np.nan),
                                       mask=feasibility_mask),
        "weight_risk": _component_stats(n_w),
        "open_share": _component_stats(n_open),
    }
    diag["djh_norm_scope"] = norm_scope
    diag["D_jh_stats"] = _component_stats(D, mask=feasibility_mask)
    return D, feasibility_mask, slack, diag


# ===========================================================================
# Step B: hub preference vector (Eq.8)  softmax(-D_jh / theta)
# ===========================================================================
def _build_hub_preference(D_jh: np.ndarray, feasibility_mask: np.ndarray,
                          cfg: ServiceStructureConfig
                          ) -> tuple[np.ndarray, dict[str, Any]]:
    """对每个 customer j 计算 hub preference, 用 per-customer rank-based 公式.

    Rank-based formula (option D from sensitivity analysis on verylowHubCost):
        rank[j, h] = D_jh[j, :] 中 D_jh[j, h] 的升序排名 (0 = 最佳)
        对 infeasible 的 (j, h), rank = +∞
        raw[j, h] = 1 / (rank[j, h] + 1)  对 feasible
        raw[j, h] = 0 对 infeasible
        hub_pref[j, h] = raw[j, h] / Σ_h raw[j, h]   # row-normalize

    优点:
      - 完全不依赖 D_jh 数值幅度 / 各分量量纲, 避免 joint minmax 退化
      - 在 D_jh 排序信息相同但 D_jh 数值幅度跨 hub 差异很小时仍有信息
      - 数学上等价于 Borda count / reciprocal rank fusion

    Returns:
        hub_pref: (n_cust, n_hub) ndarray, 每行 sum=1 (若至少一个 hub feasible) 或全 0
        diag: 包含 strategy 名, max_pref 统计, infeasible 数, ties 数
    """
    n_cust, n_hub = D_jh.shape

    # 对 infeasible 处置为 +inf, 这样 argsort 后它们排在末尾
    D_eff = np.where(feasibility_mask, D_jh, np.inf)

    # 每行 argsort 得到 hub 索引顺序 (升序). rank[j, h] = 该 hub 在 cust j 排序里的位置
    order = np.argsort(D_eff, axis=1, kind="stable")  # shape (n_cust, n_hub)
    rank = np.empty_like(order)
    rows = np.arange(n_cust)[:, np.newaxis]
    rank[rows, order] = np.arange(n_hub)[np.newaxis, :]

    # raw[j, h] = 1 / (rank+1) 对 feasible, 0 对 infeasible
    raw = np.where(feasibility_mask, 1.0 / (rank.astype(float) + 1.0), 0.0)

    # 行归一化
    Z = raw.sum(axis=1, keepdims=True)
    Z_safe = np.where(Z <= 0, 1.0, Z)
    hub_pref = raw / Z_safe
    # 若一行全 0 (cust 在所有 hub 都不可行), 保持全 0
    all_zero = (Z <= 0).flatten()
    hub_pref[all_zero] = 0.0

    diag = {
        "strategy": "per_customer_rank_inverse",
        "row_sum_min": float(hub_pref.sum(axis=1).min()),
        "row_sum_max": float(hub_pref.sum(axis=1).max()),
        "max_pref_per_row_mean": float(hub_pref.max(axis=1).mean()),
        "n_rows_all_infeasible": int(all_zero.sum()),
        "feasible_count_per_row_min": int(feasibility_mask.sum(axis=1).min()),
        "feasible_count_per_row_max": int(feasibility_mask.sum(axis=1).max()),
    }
    return hub_pref, diag


# ===========================================================================
# Step C: customer-customer service dissimilarity (Eq.9)
# ===========================================================================
def _build_D_cust(data: "DataContainer",
                  customer_ids: list[int],
                  hub_pref: np.ndarray,
                  cfg: ServiceStructureConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """计算 D_cust[i, j]. dissimilarity, 不必满足严格 metric axioms.

    分量 (各自归一化):
        1. spatial distance (Euclidean 在客户坐标上)
        2. |t_rel_i - t_rel_j|
        3. |t_due_i - t_due_j|
        4. |g_i - g_j|
        5. ||p_i - p_j||_1   (hub preference L1)
    """
    n = len(customer_ids)
    coords = np.array([data.customers[c].coord for c in customer_ids])
    t_rel = np.array([data.customers[c].t_rel for c in customer_ids])
    t_due = np.array([data.customers[c].t_due for c in customer_ids])
    weights = np.array([data.customers[c].weight for c in customer_ids])

    # NOTE (scaling fix): the original implementation built (n, n, ·) broadcast
    # tensors, e.g. ||p_i - p_j||_1 materialised a (n, n, n_hub) array
    # (= 14.2 GiB at n=3840, n_hub=129) and OOM'd on large instances. We replace
    # those with scipy.spatial.distance.cdist, which produces the *identical*
    # (n, n) pairwise distances without the 3-D intermediate.
    from scipy.spatial.distance import cdist

    # 1. spatial (Euclidean) — same result as sqrt(sum(diff**2))
    d_space = cdist(coords, coords, metric="euclidean")

    # 2-4. scalar absolute differences (1-D cityblock == |a_i - a_j|)
    d_rel = cdist(t_rel[:, None], t_rel[:, None], metric="cityblock")
    d_due = cdist(t_due[:, None], t_due[:, None], metric="cityblock")
    d_w = cdist(weights[:, None], weights[:, None], metric="cityblock")

    # 5. hub-pref L1 — same result as abs(diff).sum(axis=-1)
    d_pref = cdist(hub_pref, hub_pref, metric="cityblock")

    # 归一化每个分量
    diag = {
        "raw_components": {
            "space": _component_stats(d_space),
            "release": _component_stats(d_rel),
            "due": _component_stats(d_due),
            "weight": _component_stats(d_w),
            "hubpref_L1": _component_stats(d_pref),
        }
    }
    n_space = _minmax_norm(d_space, cfg.epsilon)
    n_rel = _minmax_norm(d_rel, cfg.epsilon)
    n_due = _minmax_norm(d_due, cfg.epsilon)
    n_w = _minmax_norm(d_w, cfg.epsilon)
    n_pref = _minmax_norm(d_pref, cfg.epsilon)

    D = (cfg.lambda_space * n_space
         + cfg.lambda_release * n_rel
         + cfg.lambda_due * n_due
         + cfg.lambda_weight * n_w
         + cfg.lambda_hubpref * n_pref)
    # 对称化 (理论上已对称,数值上保险一刀)
    D = 0.5 * (D + D.T)
    # 对角置 0
    np.fill_diagonal(D, 0.0)
    diag["D_cust_stats"] = _component_stats(D)
    return D, diag


# ===========================================================================
# Step D: simple deterministic k-medoids (no sklearn_extra)
# ===========================================================================
def _kmedoids_pam_like(D: np.ndarray, q: int, max_iter: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """简单 PAM-like k-medoids.

    步骤:
      1. farthest-first 初始化 q 个 medoids
      2. assign 每个点到最近 medoid
      3. 每个 cluster 内, 取使 sum_{j in C} D[medoid_candidate, j] 最小者为新 medoid
      4. 直到 medoids 不变或达 max_iter
    """
    n = D.shape[0]
    if q >= n:
        # 退化: 每个点单独成簇
        return np.arange(n), list(range(n))
    if q <= 1:
        # 退化: 单簇, medoid 取总距离最小者
        sums = D.sum(axis=1)
        m = int(np.argmin(sums))
        return np.zeros(n, dtype=int), [m]

    # farthest-first init: 第一个 medoid = 距其它点总距离最小者 (重心), 后续逐个加远的
    sums = D.sum(axis=1)
    first = int(np.argmin(sums))
    medoids = [first]
    for _ in range(q - 1):
        # 选最远 (到 medoids 中最近一个的距离最大者)
        dist_to_nearest = D[:, medoids].min(axis=1)
        # 已是 medoid 的距离 0 -> 排除
        dist_to_nearest[medoids] = -np.inf
        # 平局时确定性: 选索引最小
        nxt = int(np.argmax(dist_to_nearest))
        medoids.append(nxt)

    medoids = sorted(medoids)
    labels = np.zeros(n, dtype=int)

    for _it in range(max_iter):
        # assign
        dist_to_medoids = D[:, medoids]  # (n, q)
        labels = np.argmin(dist_to_medoids, axis=1)

        # update medoids
        new_medoids = []
        changed = False
        for r in range(q):
            members = np.where(labels == r)[0]
            if len(members) == 0:
                # 空簇: 保留旧 medoid (cluster 后续可能被丢弃)
                new_medoids.append(medoids[r])
                continue
            # 在簇内找使总簇内距离最小的点
            sub = D[np.ix_(members, members)]
            sums_in = sub.sum(axis=1)
            best_idx_in_members = int(np.argmin(sums_in))
            new_m = int(members[best_idx_in_members])
            if new_m != medoids[r]:
                changed = True
            new_medoids.append(new_m)
        new_medoids = sorted(new_medoids)
        if not changed:
            medoids = new_medoids
            break
        medoids = new_medoids

    # final re-assign
    dist_to_medoids = D[:, medoids]
    labels = np.argmin(dist_to_medoids, axis=1)
    return labels, list(medoids)


# ===========================================================================
# Step E: Service Structure Score (Eq.12)
# ===========================================================================
def _compute_sss(labels: np.ndarray, medoids: list[int],
                 D_cust: np.ndarray, hub_pref: np.ndarray,
                 data: "DataContainer",
                 customer_ids: list[int], hub_ids: list[int],
                 feasibility_mask: np.ndarray,
                 slack: np.ndarray,
                 cfg: ServiceStructureConfig
                 ) -> tuple[float, dict[str, float]]:
    """SSS(q) = ω1·Comp + ω2·Sep + ω3·HubPur - ω4·Risk - ω5·SizePen.

    TimeRisk(C_r) 严格按 paper Eq.(18) / spec §10:
        TimeRisk(C_r) = (1/|C_r|) Σ_{j∈C_r} 1/(slack[j, h*_r] + eps)
    其中 h*_r = cluster preferred hub (avg hub_pref 最大且至少一个成员可行).
    归一化:在本次 SSS 评估的所有 cluster TimeRisk 上做 minmax → [0,1],
    使 ω_risk 在不同 q 间的尺度一致.
    """
    n = D_cust.shape[0]
    q = len(medoids)
    eps = cfg.epsilon

    # ----- Comp(q) = 1 - mean normalized distance to medoid -----
    Dmax = float(D_cust.max()) if D_cust.size > 0 else 0.0
    if Dmax <= eps:
        comp = 1.0
    else:
        dist_to_medoid = np.array([D_cust[i, medoids[labels[i]]] / Dmax
                                    for i in range(n)])
        comp = 1.0 - float(dist_to_medoid.mean())
    comp = max(0.0, min(1.0, comp))

    # ----- Sep(q) = mean pairwise normalized medoid distance -----
    if q <= 1:
        sep = 0.0
    else:
        Dm = D_cust[np.ix_(medoids, medoids)]
        mask = ~np.eye(q, dtype=bool)
        if Dmax <= eps:
            sep = 0.0
        else:
            sep = float(Dm[mask].mean()) / Dmax
    sep = max(0.0, min(1.0, sep))

    # ----- HubPur(q) = mean_r max_h P_rh -----
    purs = []
    for r in range(q):
        members = np.where(labels == r)[0]
        if len(members) == 0:
            continue
        P_r = hub_pref[members].mean(axis=0)
        purs.append(float(P_r.max()))
    hub_pur = float(np.mean(purs)) if purs else 0.0

    # ----- Risk(q): strict slack-based TimeRisk + CapRisk per cluster -----
    # 先按 cluster 算 raw time risk = mean_{j∈C_r} 1/(slack[j, h*_r] + eps)
    # 然后对所有 cluster 做 minmax 归一化到 [0,1].
    raw_time_risks: list[float] = []
    cap_risks: list[float] = []
    cluster_h_stars: list[int] = []
    for r in range(q):
        members = np.where(labels == r)[0]
        if len(members) == 0:
            raw_time_risks.append(0.0)
            cap_risks.append(0.0)
            cluster_h_stars.append(-1)
            continue
        # preferred hub: avg hub_pref 排序后取第一个 cluster-feasible 的
        P_r = hub_pref[members].mean(axis=0)
        cluster_feas_per_hub = feasibility_mask[members, :].any(axis=0)
        order = np.argsort(-P_r)
        h_star = -1
        for h in order:
            if cluster_feas_per_hub[int(h)]:
                h_star = int(h)
                break
        cluster_h_stars.append(h_star)
        # TimeRisk via slack: mean of 1/(slack+eps) for members where slack>0.
        if h_star >= 0:
            s_vals = slack[members, h_star]
            pos = s_vals > 0
            if pos.any():
                tr = float(np.mean(1.0 / (s_vals[pos] + eps)))
                # 若部分 member 在 h_star 下 slack<=0, 用大值惩罚
                if (~pos).any():
                    tr = tr + float((~pos).sum()) / max(1, len(s_vals))  # crude penalty
            else:
                tr = 1.0 / eps  # 全 infeasible 在 preferred hub 下 → 极高风险
        else:
            tr = 1.0 / eps  # cluster 在所有 hub 都不可行
        raw_time_risks.append(tr)

        # CapRisk: total_weight / l_K - 1, 截断于 0
        total_w = sum(data.customers[customer_ids[i]].weight for i in members)
        cap_risks.append(max(0.0, total_w / data.l_K - 1.0))

    # Minmax 归一化 TimeRisk 到 [0,1] (cluster-级别 cross-cluster norm)
    if raw_time_risks:
        arr = np.array(raw_time_risks, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            lo, hi = float(finite.min()), float(finite.max())
            if hi - lo < eps:
                norm_time_risks = np.zeros_like(arr)
            else:
                norm_time_risks = np.clip((arr - lo) / (hi - lo + eps), 0.0, 1.0)
        else:
            norm_time_risks = np.zeros_like(arr)
    else:
        norm_time_risks = np.array([])

    if len(norm_time_risks) > 0:
        risk_per_cluster = [float(nt + cr) for nt, cr in zip(norm_time_risks, cap_risks)]
        risk = float(np.mean(risk_per_cluster))
    else:
        risk = 0.0

    # ----- SizePen(q) = q / n -----
    size_pen = q / max(1, n)

    sss = (cfg.omega_compactness * comp
           + cfg.omega_separation * sep
           + cfg.omega_hub_purity * hub_pur
           - cfg.omega_risk * risk
           - cfg.omega_fragmentation * size_pen)
    return sss, {
        "Comp": comp,
        "Sep": sep,
        "HubPur": hub_pur,
        "Risk": risk,
        "SizePen": size_pen,
        "TimeRisk_raw_per_cluster": [float(x) for x in raw_time_risks],
        "TimeRisk_norm_per_cluster": [float(x) for x in norm_time_risks],
        "CapRisk_per_cluster": [float(x) for x in cap_risks],
        "cluster_h_stars": cluster_h_stars,
    }


# ===========================================================================
# Step F: top-m hubs per customer
# ===========================================================================
def _build_top_hubs(hub_pref: np.ndarray, hub_ids: list[int],
                    cust_ids: list[int], top_m: int
                    ) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    n_cust, n_hub = hub_pref.shape
    m = min(top_m, n_hub)
    # argsort descending
    order = np.argsort(-hub_pref, axis=1)
    for i in range(n_cust):
        out[cust_ids[i]] = [hub_ids[int(order[i, k])] for k in range(m)]
    return out


# ===========================================================================
# Step G: build ServiceCluster objects + cluster graph
# ===========================================================================
def _build_cluster_objects(labels: np.ndarray, medoids: list[int],
                           D_cust: np.ndarray, hub_pref: np.ndarray,
                           data: "DataContainer",
                           customer_ids: list[int], hub_ids: list[int],
                           feasibility_mask: np.ndarray,
                           slack: np.ndarray,
                           cfg: ServiceStructureConfig
                           ) -> list[ServiceCluster]:
    """从 labels/medoids 构造 ServiceCluster 列表.

    每个 ServiceCluster.time_risk 严格按 paper Eq.(18):
        time_risk(C_r) = (1/|C_r|) Σ_{j∈C_r, slack>0} 1/(slack[j, h*_r] + eps)
    若 cluster 在所有 hub 都不可行, 设为 1/eps (极高风险).
    若部分 member 在 h_star 下 slack<=0, 加 fraction penalty.
    """
    clusters: list[ServiceCluster] = []
    q = len(medoids)
    eps = cfg.epsilon
    for r in range(q):
        member_idx = np.where(labels == r)[0]
        if len(member_idx) == 0:
            continue
        member_cids = [customer_ids[int(i)] for i in member_idx]
        medoid_cid = customer_ids[int(medoids[r])]

        # cluster avg hub preference
        P_r = hub_pref[member_idx].mean(axis=0)
        hub_pref_dict = {hub_ids[h]: float(P_r[h]) for h in range(len(hub_ids))}

        # preferred hub: avg P_r 排序后取第一个 cluster-feasible
        cluster_feas_per_hub = feasibility_mask[member_idx, :].any(axis=0)
        order = np.argsort(-P_r)
        preferred = -1
        preferred_idx = -1
        for h in order:
            if cluster_feas_per_hub[int(h)]:
                preferred = hub_ids[int(h)]
                preferred_idx = int(h)
                break

        coords = np.array([data.customers[c].coord for c in member_cids])
        rels = np.array([data.customers[c].t_rel for c in member_cids])
        dues = np.array([data.customers[c].t_due for c in member_cids])
        ws = np.array([data.customers[c].weight for c in member_cids])

        # service radius: cluster 内最大 D_cust 到 medoid
        srad = float(D_cust[member_idx, medoids[r]].max()) if len(member_idx) > 0 else 0.0

        # 严格按 paper Eq.(18): time_risk = mean 1/(slack[j, h*]+eps) over members.
        # 若 preferred_idx < 0 (cluster 完全不可行), 设大值.
        if preferred_idx >= 0:
            s_vals = slack[member_idx, preferred_idx]
            pos = s_vals > 0
            if pos.any():
                time_risk = float(np.mean(1.0 / (s_vals[pos] + eps)))
                if (~pos).any():
                    time_risk = time_risk + float((~pos).sum()) / max(1, len(s_vals))
            else:
                time_risk = 1.0 / eps
        else:
            time_risk = 1.0 / eps

        # cap risk
        total_w = float(ws.sum())
        cap_risk = max(0.0, total_w / data.l_K - 1.0)

        clusters.append(ServiceCluster(
            id=r,
            customers=member_cids,
            medoid=medoid_cid,
            size=len(member_cids),
            total_weight=total_w,
            spatial_center=(float(coords[:, 0].mean()), float(coords[:, 1].mean())),
            rel_center=float(rels.mean()),
            due_center=float(dues.mean()),
            hub_pref=hub_pref_dict,
            preferred_hub=preferred,
            service_radius=srad,
            time_risk=time_risk,
            capacity_risk=cap_risk,
        ))
    return clusters


def _build_cluster_graph(clusters: list[ServiceCluster], D_cust: np.ndarray,
                         cust_id_to_idx: dict[int, int],
                         k_neighbors: int) -> tuple[dict[int, list[int]],
                                                     list[tuple[int, int, float]]]:
    """构建无向 cluster graph. 边权 = medoid-medoid D_cust.

    Returns:
        graph: cluster_id -> [neighbor cluster_id, ...]
        edges: list of (r, s, distance), r<s, 用于导出
    """
    q = len(clusters)
    if q <= 1:
        return ({c.id: [] for c in clusters}, [])

    # medoid 索引
    m_idx = np.array([cust_id_to_idx[c.medoid] for c in clusters])
    Dm = D_cust[np.ix_(m_idx, m_idx)]  # (q, q)

    graph: dict[int, list[int]] = {c.id: [] for c in clusters}
    k = min(k_neighbors, q - 1)
    for r in range(q):
        # 排序找 k 个最近 (排除自己)
        dists = Dm[r].copy()
        dists[r] = np.inf
        nearest = np.argsort(dists)[:k]
        for s in nearest:
            sid = clusters[int(s)].id
            if sid not in graph[clusters[r].id]:
                graph[clusters[r].id].append(sid)

    # 强制无向: r in graph[s] iff s in graph[r]
    for c in clusters:
        rid = c.id
        for sid in list(graph[rid]):
            if rid not in graph[sid]:
                graph[sid].append(rid)

    # edges (r < s) with distance
    seen = set()
    edges: list[tuple[int, int, float]] = []
    cid_to_idx = {c.id: i for i, c in enumerate(clusters)}
    for rid, nbrs in graph.items():
        for sid in nbrs:
            key = (min(rid, sid), max(rid, sid))
            if key in seen:
                continue
            seen.add(key)
            r_i = cid_to_idx[key[0]]
            s_i = cid_to_idx[key[1]]
            edges.append((key[0], key[1], float(Dm[r_i, s_i])))
    return graph, edges


# ===========================================================================
# Public API: build_service_structure
# ===========================================================================
def _golden_section_q(lo: int, hi: int, eval_q) -> None:
    """Golden-section (binary-style) search for the q in [lo, hi] maximizing SSS.

    Evaluates ~O(log(hi-lo)) candidates instead of all of them. Assumes the score
    is unimodal in q; if it is monotone (as on dense instances where finer
    clustering keeps scoring higher) it converges to the appropriate boundary.
    eval_q(q) must return the (memoized) score for q.
    """
    a, b = int(lo), int(hi)
    eval_q(a)
    eval_q(b)
    guard = 0
    while b - a > 2 and guard < 64:
        guard += 1
        m1 = a + (b - a) // 3
        m2 = b - (b - a) // 3
        if m1 == a:
            m1 = a + 1
        if m2 == b:
            m2 = b - 1
        if m1 >= m2:
            break
        f1, f2 = eval_q(m1), eval_q(m2)
        if f1 < f2:          # peak is to the right of m1
            a = m1
        else:                # peak is to the left of m2
            b = m2
    for q in range(a, b + 1):  # exhaust the small remaining bracket
        eval_q(q)


def _coarse_to_fine_q(lo: int, hi: int, eval_batch, score_by_q,
                      rounds: int = 3, k: int = 7) -> None:
    """Parallel-friendly q search: evaluate a coarse grid (in parallel), zoom to
    the neighborhood of the best, repeat. Robust to mild multimodality and uses
    far fewer than (hi-lo) evaluations while keeping each round fully parallel.
    """
    a, b = int(lo), int(hi)
    for _ in range(max(1, rounds)):
        if b - a <= 2:
            break
        grid = sorted(set(
            int(round(a + i * (b - a) / (k - 1))) for i in range(k)))
        eval_batch(grid)
        best = max(grid, key=lambda q: score_by_q.get(q, -1e18))
        idx = grid.index(best)
        a = grid[max(0, idx - 1)]
        b = grid[min(len(grid) - 1, idx + 1)]
    eval_batch(range(a, b + 1))


def build_service_structure(data: "DataContainer", tt: "TravelTimes",
                            cfg: ServiceStructureConfig) -> ServiceStructure:
    """主入口. 计算 D_jh / hub_pref / D_cust, 选自适应 q, 构造 clusters + graph.

    Args:
        data: DataContainer
        tt: TravelTimes
        cfg: ServiceStructureConfig

    Returns:
        ServiceStructure
    """
    t0 = time.time()
    rng = np.random.default_rng(cfg.random_seed)

    customer_ids = list(data.J)
    hub_ids = list(data.H)
    n_cust = len(customer_ids)
    n_hub = len(hub_ids)

    cust_id_to_idx = {c: i for i, c in enumerate(customer_ids)}
    idx_to_cust_id = {i: c for c, i in cust_id_to_idx.items()}
    hub_id_to_idx = {h: i for i, h in enumerate(hub_ids)}
    idx_to_hub_id = {i: h for h, i in hub_id_to_idx.items()}

    if n_cust < 2:
        raise ValueError(f"S-alns requires n_customers >= 2, got {n_cust}")

    # --- A: D_jh + feasibility + slack ---
    D_jh, feas_mask, slack, diag_A = _build_D_jh(
        data, tt, cfg, customer_ids, hub_ids, cust_id_to_idx, hub_id_to_idx
    )

    # --- B: hub preference ---
    hub_pref, diag_B = _build_hub_preference(D_jh, feas_mask, cfg)

    # --- C: D_cust ---
    D_cust, diag_C = _build_D_cust(data, customer_ids, hub_pref, cfg)

    # --- D: top-m hubs ---
    top_hubs = _build_top_hubs(hub_pref, hub_ids, customer_ids, cfg.top_m_hubs)

    # --- E: adaptive q selection ---
    # 候选 q 区间
    if cfg.auto_cluster:
        q_min = max(2, cfg.q_min)
        if cfg.q_max_mode == "sqrt_n":
            q_max = int(math.ceil(math.sqrt(n_cust)))
        elif cfg.q_max_mode == "fixed" and cfg.q_fixed is not None:
            q_max = int(cfg.q_fixed)
        else:
            q_max = int(math.ceil(math.sqrt(n_cust)))
        q_max = max(q_min, min(q_max, n_cust))
        candidates = list(range(q_min, q_max + 1))
    else:
        # 固定 q
        qf = cfg.q_fixed if cfg.q_fixed is not None else max(2, int(math.ceil(math.sqrt(n_cust))))
        qf = max(2, min(qf, n_cust))
        candidates = [qf]

    if not candidates:
        candidates = [min(2, n_cust)]

    score_by_q: dict[int, float] = {}
    cached: dict[int, tuple[np.ndarray, list[int], dict[str, float]]] = {}
    base_seed = int(cfg.random_seed)

    def _eval_q(q: int) -> float:
        """Evaluate one q (k-medoids + SSS), memoized. Uses a per-q deterministic
        RNG so results are reproducible AND thread-safe (no shared Generator)."""
        if q in score_by_q:
            return score_by_q[q]
        q_rng = np.random.default_rng(base_seed + 100_003 + q)
        labels_q, medoids_q = _kmedoids_pam_like(
            D_cust, q, cfg.max_kmedoids_iter, q_rng)
        sss_q, breakdown_q = _compute_sss(
            labels_q, medoids_q, D_cust, hub_pref,
            data, customer_ids, hub_ids, feas_mask, slack, cfg)
        score_by_q[q] = float(sss_q)          # dict writes are GIL-atomic -> thread-safe
        cached[q] = (labels_q, medoids_q, breakdown_q)
        return float(sss_q)

    n_jobs = max(1, int(getattr(cfg, "q_search_jobs", 1) or 1))

    def _eval_batch(qs) -> None:
        todo = [q for q in dict.fromkeys(int(x) for x in qs)
                if 1 <= q <= n_cust and q not in score_by_q]
        if not todo:
            return
        if n_jobs > 1 and len(todo) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(n_jobs, len(todo))) as ex:
                list(ex.map(_eval_q, todo))
        else:
            for q in todo:
                _eval_q(q)

    mode = getattr(cfg, "q_search_mode", "full") or "full"
    if len(candidates) == 1:
        _eval_batch(candidates)
    elif mode == "golden":
        _golden_section_q(candidates[0], candidates[-1], _eval_q)
    elif mode == "coarse2fine":
        _coarse_to_fine_q(candidates[0], candidates[-1], _eval_batch, score_by_q)
    else:  # "full"
        _eval_batch(candidates)

    log.info(f"S-alns q-search[{mode}, jobs={n_jobs}]: evaluated {len(score_by_q)}"
             f"/{len(candidates)} candidate q in [{candidates[0]},{candidates[-1]}]")

    # 选最大 SSS
    selected_q = max(score_by_q.keys(), key=lambda k: score_by_q[k])
    labels, medoids, score_breakdown = cached[selected_q]

    # --- F: build cluster objects ---
    clusters = _build_cluster_objects(
        labels, medoids, D_cust, hub_pref,
        data, customer_ids, hub_ids, feas_mask, slack, cfg
    )

    # 客户 -> cluster id
    cluster_id_of_customer: dict[int, int] = {}
    for c in clusters:
        for j in c.customers:
            cluster_id_of_customer[j] = c.id

    # --- G: cluster graph ---
    graph, edges = _build_cluster_graph(
        clusters, D_cust, cust_id_to_idx, cfg.graph_k_neighbors
    )

    runtime_sec = time.time() - t0

    diagnostics: dict[str, Any] = {
        "D_jh": diag_A,
        "hub_pref": diag_B,
        "D_cust": diag_C,
        "sss_breakdown_selected_q": score_breakdown,
        "candidates": candidates,
        "cluster_edges": [{"r": r, "s": s, "distance": d} for (r, s, d) in edges],
        "runtime_sec": runtime_sec,
    }

    ss = ServiceStructure(
        customer_ids=customer_ids,
        hub_ids=hub_ids,
        customer_id_to_idx=cust_id_to_idx,
        idx_to_customer_id=idx_to_cust_id,
        hub_id_to_idx=hub_id_to_idx,
        idx_to_hub_id=idx_to_hub_id,
        D_jh=D_jh,
        D_cust=D_cust,
        hub_pref=hub_pref,
        top_hubs=top_hubs,
        cluster_id_of_customer=cluster_id_of_customer,
        clusters=clusters,
        cluster_graph=graph,
        selected_q=selected_q,
        score_by_q=score_by_q,
        diagnostics=diagnostics,
        config=deepcopy(cfg),
    )
    log.info(
        f"S-alns: n_cust={n_cust}, n_hub={n_hub}, selected_q={selected_q}, "
        f"n_clusters={len(clusters)}, runtime={runtime_sec:.2f}s"
    )
    return ss


# ===========================================================================
# Export
# ===========================================================================
def export_service_structure(ss: ServiceStructure, out_dir: str,
                             prefix: str = "",
                             max_pairwise_n: int = 600) -> None:
    """导出 service-structure 文件 (Q3 中也输出边权 distance).

    Args:
        max_pairwise_n: 当 n_customers > 该阈值时, 跳过 O(n^2) 的两两距离明细
            (service_customer_distance.csv) 与 O(n*H) 的 customer-hub / hub-pref
            明细, 避免在大算例上写出数百万行的巨型 CSV (n=3840 -> 7.4M 行).
            聚类摘要 / 边 / score 等小文件始终导出, 可视化仅依赖这些。
    """
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, f"{prefix}{name}")
    n = ss.D_cust.shape[0]
    emit_pairwise = (n <= max_pairwise_n)

    if emit_pairwise:
        # 1. customer-hub distance (long form)
        rows_jh = [{"customer_id": cid, "hub_id": hid,
                    "D_jh": float(ss.D_jh[j_idx, h_idx])}
                   for j_idx, cid in enumerate(ss.customer_ids)
                   for h_idx, hid in enumerate(ss.hub_ids)]
        pd.DataFrame(rows_jh).to_csv(p("service_customer_hub_distance.csv"), index=False)

        # 2. customer-customer distance — vectorised upper triangle (no Python loop)
        iu, ju = np.triu_indices(n)
        cid_arr = np.asarray(ss.customer_ids)
        pd.DataFrame({
            "customer_i": cid_arr[iu],
            "customer_j": cid_arr[ju],
            "D_cust": ss.D_cust[iu, ju].astype(float),
        }).to_csv(p("service_customer_distance.csv"), index=False)

        # 3. hub preference (long form)
        rows_hp = [{"customer_id": cid, "hub_id": hid,
                    "preference": float(ss.hub_pref[j_idx, h_idx])}
                   for j_idx, cid in enumerate(ss.customer_ids)
                   for h_idx, hid in enumerate(ss.hub_ids)]
        pd.DataFrame(rows_hp).to_csv(p("hub_preference.csv"), index=False)
    else:
        log.warning(
            f"export_service_structure: n_cust={n} > max_pairwise_n={max_pairwise_n}; "
            f"skipping pairwise CSVs (customer_distance / customer_hub_distance / hub_preference). "
            f"Cluster-level files are still written."
        )

    # 4. clusters summary
    rows_cl = []
    for c in ss.clusters:
        rows_cl.append({
            "cluster_id": c.id,
            "size": c.size,
            "medoid_customer_id": c.medoid,
            "preferred_hub_id": c.preferred_hub,
            "total_weight": c.total_weight,
            "spatial_center_x": c.spatial_center[0],
            "spatial_center_y": c.spatial_center[1],
            "rel_center": c.rel_center,
            "due_center": c.due_center,
            "service_radius": c.service_radius,
            "time_risk": c.time_risk,
            "capacity_risk": c.capacity_risk,
            "customers": ",".join(str(x) for x in c.customers),
        })
    pd.DataFrame(rows_cl).to_csv(p("service_clusters.csv"), index=False)

    # 5. cluster edges (含 distance, Q3 要求)
    edges_info = ss.diagnostics.get("cluster_edges", [])
    pd.DataFrame(edges_info).to_csv(p("service_cluster_edges.csv"), index=False)

    # 6. score by q
    rows_sq = [{"q": q, "score": v} for q, v in sorted(ss.score_by_q.items())]
    pd.DataFrame(rows_sq).to_csv(p("service_score_by_q.csv"), index=False)

    # 7. summary JSON
    summary = {
        "selected_q": ss.selected_q,
        "n_clusters": len(ss.clusters),
        "n_customers": len(ss.customer_ids),
        "n_hubs": len(ss.hub_ids),
        "cluster_sizes": [c.size for c in ss.clusters],
        "preferred_hubs": [c.preferred_hub for c in ss.clusters],
        "score_by_q": {str(k): float(v) for k, v in ss.score_by_q.items()},
        "sss_breakdown_selected_q":
            ss.diagnostics.get("sss_breakdown_selected_q", {}),
        "runtime_sec": ss.diagnostics.get("runtime_sec", 0.0),
        "diagnostics": {
            "D_jh_raw_components": ss.diagnostics["D_jh"]["raw_components"],
            "D_jh_normalized_components":
                ss.diagnostics["D_jh"]["normalized_components"],
            "D_jh_overall": ss.diagnostics["D_jh"]["D_jh_stats"],
            "D_jh_feasibility_mask_density":
                ss.diagnostics["D_jh"]["feasibility_mask_density"],
            "D_jh_feasible_count_per_hub":
                ss.diagnostics["D_jh"]["feasible_count_per_hub"],
            "hub_pref": ss.diagnostics["hub_pref"],
            "D_cust_raw_components":
                ss.diagnostics["D_cust"]["raw_components"],
            "D_cust_overall":
                ss.diagnostics["D_cust"]["D_cust_stats"],
        },
        "cluster_graph": {str(k): v for k, v in ss.cluster_graph.items()},
    }
    with open(p("service_structure_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"S-alns: exported 7 files to {out_dir}")
