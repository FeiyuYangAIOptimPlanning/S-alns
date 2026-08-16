"""
alns_core.py — ALNS 主循环

对齐论文 §4.2.4:
- Roulette wheel 选择 destroy + repair
- Metropolis 接受准则, 冷却 Z *= α
- 每 π 轮更新权重 w_i = w_i*(1-ρ) + ρ*(score_i/count_i)
- σ = (5, 3, 1) (论文 Table 3 加粗)
- Z0 = 1000, α = 0.999, ρ = 0.3, π = 100, N_max = 5000
"""
from __future__ import annotations
import numpy as np
import math
import sys
import logging
from copy import deepcopy
from dataclasses import dataclass, field

from .structures import Solution
from .data_loader import DataContainer
from .travel_times import TravelTimes
from .destroy_operators import DESTROY_OPS, DESTROY_NAMES
from .repair_operators import REPAIR_OPS, REPAIR_NAMES
from .costs import cost
from .timing_subproblem import refresh_all_timing

log = logging.getLogger(__name__)


@dataclass
class ALNSConfig:
    N_max: int = 5000
    pi: int = 100
    Z0: float = 1000.0
    alpha: float = 0.999
    rho: float = 0.3
    sigma: tuple[float, float, float] = (5.0, 3.0, 1.0)   # (new best, improve curr, accepted worse)
    weight_floor: float = 0.01                              # 防止某算子权重归零
    seed: int | None = 42
    log_every: int = 100
    # S-alns: lower bound for structure-guided operator weights (avoids early elimination)
    structure_operator_floor: float = 0.03
    # Option D fix: restart S_current from S_best when stuck for too many iterations.
    # 0 = disabled (baseline behavior); typical: 50-200.
    restart_from_best_after_stuck: int = 0
    # Option D fix: when restart triggers, multiplicatively reset Z to allow fresh exploration.
    restart_temperature_factor: float = 1.0  # 1.0 = no reset
    # Operator exclusion: skip these destroy/repair ops (e.g. for large instances)
    destroy_exclude: list[str] = field(default_factory=list)
    repair_exclude: list[str] = field(default_factory=list)
    # Treat the service-guided salvage pass as part of the S-guided repair
    # component. Ablation arms with R=0 must disable it to prevent leakage.
    use_service_salvage: bool = True
    # Live terminal progress line (\r updated). True for interactive/VSCode runs.
    show_progress: bool = False


@dataclass
class ALNSStats:
    obj_history_best: list[float] = field(default_factory=list)
    obj_history_current: list[float] = field(default_factory=list)
    obj_history_new: list[float] = field(default_factory=list)
    accepted: list[bool] = field(default_factory=list)
    chosen_destroy: list[str] = field(default_factory=list)
    chosen_repair: list[str] = field(default_factory=list)
    destroy_removed_count: list[int] = field(default_factory=list)
    unserved_after_primary_repair: list[int] = field(default_factory=list)
    service_salvage_used: list[bool] = field(default_factory=list)
    baseline_salvage_used: list[bool] = field(default_factory=list)
    candidate_complete_before_salvage: list[bool] = field(default_factory=list)
    candidate_feasible_before_salvage: list[bool] = field(default_factory=list)
    candidate_feasible_after_salvage: list[bool] = field(default_factory=list)
    destroy_runtime_sec: list[float] = field(default_factory=list)
    repair_runtime_sec: list[float] = field(default_factory=list)
    # 每 π 轮的权重快照
    weight_snapshots_destroy: list[list[float]] = field(default_factory=list)
    weight_snapshots_repair: list[list[float]] = field(default_factory=list)
    runtime_sec: float = 0.0
    # ---- S-alns bookkeeping ----
    selected_method: str = "baseline"
    selected_q: int | None = None
    destroy_op_names: list[str] = field(default_factory=list)
    repair_op_names: list[str] = field(default_factory=list)
    structure_operator_names_destroy: list[str] = field(default_factory=list)
    structure_operator_names_repair: list[str] = field(default_factory=list)
    time_to_best_sec: float = 0.0
    time_to_best_iter: int = 0


def _roulette(weights: np.ndarray, rng: np.random.Generator) -> int:
    total = float(weights.sum())
    if total <= 0:
        return int(rng.integers(0, len(weights)))
    r = rng.uniform(0.0, total)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += float(w)
        if acc >= r:
            return i
    return len(weights) - 1


def run_alns(
    initial_sol: Solution,
    data: DataContainer,
    tt: TravelTimes,
    config: ALNSConfig,
    service_structure=None,
    service_cfg=None,
    method: str = "baseline",
) -> tuple[Solution, ALNSStats]:
    """ALNS 主循环. 返回 (best_sol, stats).

    Args:
        initial_sol, data, tt, config: baseline 接口 (与原版位置参数兼容)
        service_structure: 仅 method='s_alns' 时使用
        service_cfg: ServiceStructureConfig (仅 method='s_alns' 时使用)
        method: 'baseline' | 's_alns'
    """
    import time
    t0 = time.time()

    rng = np.random.default_rng(config.seed)

    # -------- operator pool 选择 --------
    use_s_alns = (method == "s_alns") and (service_structure is not None)
    if use_s_alns:
        # 延迟 import,避免 baseline 路径产生新依赖
        from .destroy_operators import (
            DESTROY_OPS_STRUCTURE, DESTROY_NAMES_STRUCTURE,
        )
        from .repair_operators import (
            REPAIR_OPS_STRUCTURE, REPAIR_NAMES_STRUCTURE,
        )
        # 构造 wrappers, 把 service_structure / service_cfg 闭包注入
        def _make_destroy_wrapper(fn):
            def _w(sol, data, tt, rng):
                return fn(sol, data, tt, rng,
                          service_structure=service_structure,
                          service_cfg=service_cfg)
            return _w

        def _make_repair_wrapper(fn):
            def _w(sol, data, tt, rng):
                return fn(sol, data, tt, rng,
                          service_structure=service_structure,
                          service_cfg=service_cfg)
            return _w

        destroy_ops_map = dict(DESTROY_OPS)
        for name, fn in DESTROY_OPS_STRUCTURE.items():
            destroy_ops_map[name] = _make_destroy_wrapper(fn)
        repair_ops_map = dict(REPAIR_OPS)
        for name, fn in REPAIR_OPS_STRUCTURE.items():
            repair_ops_map[name] = _make_repair_wrapper(fn)

        destroy_names = list(DESTROY_NAMES) + list(DESTROY_NAMES_STRUCTURE)
        repair_names = list(REPAIR_NAMES) + list(REPAIR_NAMES_STRUCTURE)
        structure_d_names = set(DESTROY_NAMES_STRUCTURE)
        structure_r_names = set(REPAIR_NAMES_STRUCTURE)
    else:
        # baseline 路径: 完全等同于原版
        destroy_ops_map = DESTROY_OPS
        repair_ops_map = REPAIR_OPS
        destroy_names = list(DESTROY_NAMES)
        repair_names = list(REPAIR_NAMES)
        structure_d_names = set()
        structure_r_names = set()

    # -------- apply operator exclusion --------
    if config.destroy_exclude:
        destroy_names = [n for n in destroy_names if n not in config.destroy_exclude]
        destroy_ops_map = {n: destroy_ops_map[n] for n in destroy_names}
    if config.repair_exclude:
        repair_names = [n for n in repair_names if n not in config.repair_exclude]
        repair_ops_map = {n: repair_ops_map[n] for n in repair_names}

    # Recompute structure-operator index sets AFTER exclusion, by NAME membership.
    # (Fix: the previous code carried stale pre-exclusion indices, which raised
    #  IndexError when destroy_exclude/repair_exclude shrank the operator pool —
    #  e.g. the city_large_amortized profile excludes HR/GR/GRR/AR with method=s_alns.)
    structure_d_idx = {i for i, n in enumerate(destroy_names) if n in structure_d_names}
    structure_r_idx = {i for i, n in enumerate(repair_names) if n in structure_r_names}

    n_d = len(destroy_names)
    n_r = len(repair_names)
    w_d = np.ones(n_d, dtype=float)
    w_r = np.ones(n_r, dtype=float)
    # S-alns: structure operators 初始略偏高
    if use_s_alns:
        for i in structure_d_idx:
            w_d[i] = 1.2
        for i in structure_r_idx:
            w_r[i] = 1.2
    score_d = np.zeros(n_d, dtype=float)
    score_r = np.zeros(n_r, dtype=float)
    count_d = np.zeros(n_d, dtype=int)
    count_r = np.zeros(n_r, dtype=int)

    sigma1, sigma2, sigma3 = config.sigma
    Z = config.Z0

    S_current = deepcopy(initial_sol)
    S_best = deepcopy(initial_sol)
    refresh_all_timing(S_current, data, tt)
    refresh_all_timing(S_best, data, tt)

    cost_current = cost(S_current, data, tt)
    cost_best = cost(S_best, data, tt)
    log.info(f"ALNS start [{method}]: initial cost = {cost_best:.2f}")

    stats = ALNSStats()
    stats.selected_method = method
    stats.selected_q = (service_structure.selected_q
                        if service_structure is not None else None)
    stats.destroy_op_names = list(destroy_names)
    stats.repair_op_names = list(repair_names)
    stats.structure_operator_names_destroy = [destroy_names[i] for i in sorted(structure_d_idx)]
    stats.structure_operator_names_repair = [repair_names[i] for i in sorted(structure_r_idx)]
    stats.obj_history_best.append(cost_best)
    stats.obj_history_current.append(cost_current)
    stats.obj_history_new.append(cost_current)

    best_found_iter = 0
    best_found_t = 0.0

    # Option D fix: track iterations since last improvement of S_best.
    iters_since_best_improved = 0
    restart_count = 0

    for n in range(config.N_max):
        # Option D fix: if stuck for too long, restart S_current from S_best.
        # This breaks the "S_current drifted far into a bad neighborhood" trap.
        if (config.restart_from_best_after_stuck > 0
                and iters_since_best_improved >= config.restart_from_best_after_stuck):
            S_current = deepcopy(S_best)
            cost_current = cost_best
            iters_since_best_improved = 0
            restart_count += 1
            if config.restart_temperature_factor != 1.0:
                Z = config.Z0 * config.restart_temperature_factor
            log.info(f"ALNS restart #{restart_count} at iter {n}: S_current <- S_best ({cost_best:.2f})")
        d_idx = _roulette(w_d, rng)
        r_idx = _roulette(w_r, rng)
        count_d[d_idx] += 1
        count_r[r_idx] += 1

        S_new = deepcopy(S_current)
        unserved_before_destroy = len(S_new.unserved)
        removed_count = 0
        unserved_after_primary = len(S_new.unserved)
        destroy_elapsed = 0.0
        repair_elapsed = 0.0
        try:
            op_t0 = time.perf_counter()
            destroy_ops_map[destroy_names[d_idx]](S_new, data, tt, rng)
            destroy_elapsed = time.perf_counter() - op_t0
            removed_count = max(0, len(S_new.unserved) - unserved_before_destroy)
            op_t0 = time.perf_counter()
            repair_ops_map[repair_names[r_idx]](S_new, data, tt, rng)
            repair_elapsed = time.perf_counter() - op_t0
            unserved_after_primary = len(S_new.unserved)
        except Exception as e:
            # Preserve partial timings/counts for failed calls as diagnostic data.
            if destroy_elapsed <= 0.0:
                destroy_elapsed = time.perf_counter() - op_t0
            elif repair_elapsed <= 0.0:
                repair_elapsed = time.perf_counter() - op_t0
            removed_count = max(0, len(S_new.unserved) - unserved_before_destroy)
            unserved_after_primary = len(S_new.unserved)
            log.warning(f"iter {n}: operator failed ({destroy_names[d_idx]}+"
                        f"{repair_names[r_idx]}): {e}")
            stats.accepted.append(False)
            stats.chosen_destroy.append(destroy_names[d_idx])
            stats.chosen_repair.append(repair_names[r_idx])
            stats.destroy_removed_count.append(removed_count)
            stats.unserved_after_primary_repair.append(unserved_after_primary)
            stats.service_salvage_used.append(False)
            stats.baseline_salvage_used.append(False)
            stats.candidate_complete_before_salvage.append(
                S_new.is_fully_served(data.J)
            )
            stats.candidate_feasible_before_salvage.append(bool(S_new.feasible))
            stats.candidate_feasible_after_salvage.append(False)
            stats.destroy_runtime_sec.append(destroy_elapsed)
            stats.repair_runtime_sec.append(repair_elapsed)
            stats.obj_history_best.append(cost_best)
            stats.obj_history_current.append(cost_current)
            stats.obj_history_new.append(float('nan'))  # Option D fix: mark discarded
            iters_since_best_improved += 1
            continue

        # Salvage pass: if destroy+repair left customers unserved (which causes
        # the iter to be silently discarded), try repair fallbacks to complete
        # the solution. This is critical in low-hub-cost configurations where
        # structure operators tend to leave inactive-hub-bound customers
        # unserved. We attempt salvage whenever any unserved remain — first try
        # the service-guided repair (if method=s_alns), then baseline GI.
        # This minimizes wasted iters when destroy and repair are not perfectly
        # paired.
        complete_before_salvage = S_new.is_fully_served(data.J)
        feasible_before_salvage = bool(S_new.feasible)
        service_salvage_used = False
        baseline_salvage_used = False
        if S_new.unserved:
            try:
                if use_s_alns and config.use_service_salvage:
                    service_salvage_used = True
                    from .repair_operators import service_greedy_insertion
                    service_greedy_insertion(
                        S_new, data, tt, rng,
                        service_structure=service_structure,
                        service_cfg=service_cfg,
                    )
            except Exception:
                pass
            # Final fallback: baseline GI
            if S_new.unserved:
                try:
                    baseline_salvage_used = True
                    from .repair_operators import greedy_insertion
                    greedy_insertion(S_new, data, tt, rng)
                except Exception:
                    pass

        stats.destroy_removed_count.append(removed_count)
        stats.unserved_after_primary_repair.append(unserved_after_primary)
        stats.service_salvage_used.append(service_salvage_used)
        stats.baseline_salvage_used.append(baseline_salvage_used)
        stats.candidate_complete_before_salvage.append(complete_before_salvage)
        stats.candidate_feasible_before_salvage.append(feasible_before_salvage)
        stats.candidate_feasible_after_salvage.append(
            S_new.is_fully_served(data.J) and bool(S_new.feasible)
        )
        stats.destroy_runtime_sec.append(destroy_elapsed)
        stats.repair_runtime_sec.append(repair_elapsed)

        # 丢弃不完整 / 不可行的 new solution
        if not S_new.is_fully_served(data.J) or not S_new.feasible:
            stats.accepted.append(False)
            stats.chosen_destroy.append(destroy_names[d_idx])
            stats.chosen_repair.append(repair_names[r_idx])
            stats.obj_history_best.append(cost_best)
            stats.obj_history_current.append(cost_current)
            stats.obj_history_new.append(float('nan'))  # Option D fix: mark discarded
            iters_since_best_improved += 1
        else:
            cost_new = cost(S_new, data, tt)
            accepted = False
            if cost_new < cost_best - 1e-9:
                S_best = deepcopy(S_new)
                cost_best = cost_new
                S_current = S_new
                cost_current = cost_new
                score_d[d_idx] += sigma1
                score_r[r_idx] += sigma1
                accepted = True
                best_found_iter = n + 1
                best_found_t = time.time() - t0
                iters_since_best_improved = 0  # Option D fix: reset stuck counter
            elif cost_new < cost_current - 1e-9:
                S_current = S_new
                cost_current = cost_new
                score_d[d_idx] += sigma2
                score_r[r_idx] += sigma2
                accepted = True
                iters_since_best_improved += 1
            else:
                diff = cost_new - cost_current
                p = math.exp(-diff / max(Z, 1e-9))
                if rng.uniform() < p:
                    S_current = S_new
                    cost_current = cost_new
                    score_d[d_idx] += sigma3
                    score_r[r_idx] += sigma3
                    accepted = True
                iters_since_best_improved += 1
            stats.accepted.append(accepted)
            stats.chosen_destroy.append(destroy_names[d_idx])
            stats.chosen_repair.append(repair_names[r_idx])
            stats.obj_history_best.append(cost_best)
            stats.obj_history_current.append(cost_current)
            stats.obj_history_new.append(cost_new)

        # 每 π 轮更新权重
        if (n + 1) % config.pi == 0:
            for i in range(n_d):
                if count_d[i] > 0:
                    new_w = w_d[i] * (1 - config.rho) + config.rho * (score_d[i] / count_d[i])
                    floor = config.structure_operator_floor if i in structure_d_idx \
                            else config.weight_floor
                    w_d[i] = max(new_w, floor)
            for i in range(n_r):
                if count_r[i] > 0:
                    new_w = w_r[i] * (1 - config.rho) + config.rho * (score_r[i] / count_r[i])
                    floor = config.structure_operator_floor if i in structure_r_idx \
                            else config.weight_floor
                    w_r[i] = max(new_w, floor)
            score_d[:] = 0
            score_r[:] = 0
            count_d[:] = 0
            count_r[:] = 0
            stats.weight_snapshots_destroy.append(list(w_d))
            stats.weight_snapshots_repair.append(list(w_r))

        Z *= config.alpha

        if config.show_progress:
            prog_every = max(1, config.log_every // 4)
            if (n + 1) % prog_every == 0 or (n + 1) == config.N_max:
                elapsed = time.time() - t0
                rate = (n + 1) / elapsed if elapsed > 0 else 0.0
                acc = (sum(stats.accepted) / len(stats.accepted) * 100.0
                       if stats.accepted else 0.0)
                eta = (config.N_max - (n + 1)) / rate if rate > 0 else 0.0
                eta_str = f"{int(eta // 60):02d}:{int(eta % 60):02d}"
                sys.stdout.write(
                    f"\r  [{method}] iter {n+1:>6}/{config.N_max} | "
                    f"best={cost_best:>12,.0f} | cur={cost_current:>12,.0f} | "
                    f"acc={acc:4.0f}% | {rate:5.1f} it/s | ETA {eta_str}   ")
                sys.stdout.flush()
        elif (n + 1) % config.log_every == 0:
            log.info(f"iter {n+1:>5}: best={cost_best:.2f} curr={cost_current:.2f} Z={Z:.2f}")

    if config.show_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    stats.runtime_sec = time.time() - t0
    stats.time_to_best_iter = best_found_iter
    stats.time_to_best_sec = best_found_t
    log.info(f"ALNS done [{method}]: best={cost_best:.2f}, runtime={stats.runtime_sec:.1f}s")
    return S_best, stats
