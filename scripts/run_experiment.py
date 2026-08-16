"""Matched baseline-versus-S-alns experiment used for Figure 1.

The featured configuration is ``instance_20260103`` under the emergency
(running-cost-only) objective. The same driver can reproduce all ten instances
and both cost profiles without changing algorithm code.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.alns_core import ALNSConfig, run_alns  # noqa: E402
from src.costs import compute_cost  # noqa: E402
from src.data_loader import (  # noqa: E402
    build_travel_times,
    load_instance,
    validate_candidate_hubs_for_uav,
)
from src.destroy_operators import set_max_remove  # noqa: E402
from src.initialization import (  # noqa: E402
    four_phase_initialization,
    four_phase_initialization_service_guided,
)
from src.repair_operators import set_prescreen  # noqa: E402
from src.service_structure import (  # noqa: E402
    ServiceStructureConfig,
    build_service_structure,
)
from src.structures import ID_GEN  # noqa: E402
from src.timing_subproblem import refresh_all_timing  # noqa: E402
from src.validator import validate_solution  # noqa: E402


LOG = logging.getLogger("s_alns_release")
CAPITAL_FIELDS = ("cost_H", "cost_D_inv", "cost_K_inv")
FEATURED_INSTANCE = "instance_20260103"
DEFAULT_SEEDS = "42,7,123,2024,99"


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", type=Path, default=ROOT / "data" / "instances")
    parser.add_argument("--instances", default="featured",
                        help="'featured', 'all', or comma-separated IDs/substrings.")
    parser.add_argument("--profiles", default="emergency",
                        help="commercial, emergency, or both as a comma-separated list.")
    parser.add_argument("--methods", default="baseline,s_alns")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "amortized.yaml")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "generated" / "featured")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def discover_instances(instance_dir: Path, requested: str):
    candidates = []
    for nodes_path in sorted(instance_dir.glob("*_nodes.csv")):
        stem = nodes_path.name.removesuffix("_nodes.csv")
        params_path = instance_dir / f"{stem}_params.csv"
        if params_path.is_file():
            candidates.append((stem, nodes_path, params_path))
    if requested.strip().lower() == "featured":
        requested = FEATURED_INSTANCE
    if requested.strip().lower() == "all":
        return candidates
    tokens = csv_list(requested)
    selected = [item for item in candidates if any(token in item[0] for token in tokens)]
    if not selected:
        raise ValueError(f"No instance matched --instances={requested!r}")
    return selected


def apply_profile(data, profile: str) -> str:
    if profile == "commercial":
        return "capital and running costs"
    if profile == "emergency":
        for field in CAPITAL_FIELDS:
            setattr(data, field, 0.0)
        return "running costs only; capital costs set to zero"
    raise ValueError(f"unknown profile: {profile}")


def alns_config(cfg: dict, n_iter: int, seed: int, show_progress: bool) -> ALNSConfig:
    values = dict(cfg.get("alns", {}) or {})
    if "weight_floor" in cfg and "weight_floor" not in values:
        values["weight_floor"] = cfg["weight_floor"]
    return ALNSConfig(
        N_max=n_iter,
        pi=int(values.get("pi", 100)),
        Z0=float(values.get("Z0", 1000.0)),
        alpha=float(values.get("alpha", 0.999)),
        rho=float(values.get("rho", 0.3)),
        sigma=tuple(values.get("sigma", (5.0, 3.0, 1.0))),
        weight_floor=float(values.get("weight_floor", 0.01)),
        seed=seed,
        log_every=int(values.get("log_every", 100)),
        structure_operator_floor=float(values.get("structure_operator_floor", 0.03)),
        restart_from_best_after_stuck=int(values.get("restart_from_best_after_stuck", 0)),
        restart_temperature_factor=float(values.get("restart_temperature_factor", 1.0)),
        destroy_exclude=list(values.get("destroy_exclude", []) or []),
        repair_exclude=list(values.get("repair_exclude", []) or []),
        use_service_salvage=bool(values.get("use_service_salvage", True)),
        show_progress=show_progress,
    )


def service_config(cfg: dict, seed: int) -> ServiceStructureConfig:
    values = dict(cfg.get("service_structure", {}) or {})
    fields = {
        name: values[name]
        for name in ServiceStructureConfig.__dataclass_fields__
        if name in values
    }
    result = ServiceStructureConfig(**fields)
    result.enabled = True
    result.random_seed = seed
    return result


def configure_runtime(cfg: dict, seed: int) -> None:
    repair = dict(cfg.get("repair", {}) or {})
    set_prescreen(repair.get("K_prescreen_gtrip"), repair.get("K_prescreen_atrip"))
    destroy = dict(cfg.get("destroy", {}) or {})
    alns = dict(cfg.get("alns", {}) or {})
    cap = alns.get("max_remove", destroy.get("max_remove"))
    if cap is not None:
        set_max_remove(cap, seed=seed)


def run_one(instance: str, nodes: Path, params: Path, profile: str, method: str,
            seed: int, n_iter: int, cfg: dict, show_progress: bool) -> dict:
    ID_GEN.reset()
    data = load_instance(str(nodes), str(params), instance_id=instance)
    policy = apply_profile(data, profile)
    tt = build_travel_times(data)
    validate_candidate_hubs_for_uav(data, tt, strict=False)
    configure_runtime(cfg, seed)
    search_cfg = alns_config(cfg, n_iter, seed, show_progress)

    started = time.perf_counter()
    structure = None
    structure_cfg = None
    if method == "baseline":
        initial = four_phase_initialization(data, tt)
    elif method == "s_alns":
        structure_cfg = service_config(cfg, seed)
        structure = build_service_structure(data, tt, structure_cfg)
        initial = four_phase_initialization_service_guided(data, tt, structure)
    else:
        raise ValueError(f"unknown method: {method}")

    refresh_all_timing(initial, data, tt)
    initial_cost = compute_cost(initial, data, tt).total
    best, stats = run_alns(
        initial, data, tt, search_cfg,
        service_structure=structure,
        service_cfg=structure_cfg,
        method=method,
    )
    validation = validate_solution(best, data, tt)
    costs = compute_cost(best, data, tt).as_dict()
    return {
        "instance": instance,
        "profile": profile,
        "method": method,
        "seed": seed,
        "n_iter": n_iter,
        "initial_objective": initial_cost,
        "objective": costs["total"],
        "improvement_from_start_pct": 100.0 * (initial_cost - costs["total"]) / initial_cost,
        "capital_cost": costs["hub_cost"] + costs["uav_inv_cost"] + costs["ugv_inv_cost"],
        "running_cost": costs["uav_trav_cost"] + costs["ugv_trav_cost"],
        "active_hubs": len(best.active_hubs),
        "uav_routes": len(best.aroutes),
        "ugv_routes": len(best.groutes),
        "unserved": len(best.unserved),
        "valid": bool(validation.ok),
        "selected_q": structure.selected_q if structure is not None else None,
        "time_to_best_iter": int(stats.time_to_best_iter),
        "runtime_sec": float(stats.runtime_sec),
        "wall_sec": time.perf_counter() - started,
        "profile_policy": policy,
    }


def paired_table(results: pd.DataFrame) -> pd.DataFrame:
    pivot = results.pivot_table(
        index=["profile", "instance", "seed"], columns="method",
        values="objective", aggfunc="first",
    ).dropna(subset=["baseline", "s_alns"])
    paired = pivot.reset_index().rename(columns={
        "baseline": "baseline_objective",
        "s_alns": "s_alns_objective",
    })
    paired["s_alns_cost_reduction_pct"] = (
        100.0 * (paired.baseline_objective - paired.s_alns_objective)
        / paired.baseline_objective
    )
    paired["s_alns_better"] = paired.s_alns_objective < paired.baseline_objective
    return paired


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    instances = discover_instances(args.instance_dir, args.instances)
    profiles = csv_list(args.profiles)
    methods = csv_list(args.methods)
    seeds = [int(value) for value in csv_list(args.seeds)]
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "run_results.csv"

    existing = pd.read_csv(result_path) if args.resume and result_path.is_file() else pd.DataFrame()
    done = set()
    if not existing.empty:
        done = set(zip(existing.instance, existing.profile, existing.method,
                       existing.seed, existing.n_iter))
    rows = existing.to_dict("records")
    for instance, nodes, params in instances:
        for profile in profiles:
            for method in methods:
                for seed in seeds:
                    key = (instance, profile, method, seed, args.n_iter)
                    if key in done:
                        LOG.info("reuse %s", key)
                        continue
                    LOG.info("run %s", key)
                    rows.append(run_one(instance, nodes, params, profile, method,
                                        seed, args.n_iter, cfg, args.show_progress))
                    pd.DataFrame(rows).to_csv(result_path, index=False)

    results = pd.DataFrame(rows)
    paired = paired_table(results)
    paired.to_csv(args.out / "paired_results.csv", index=False)
    summary = paired.groupby(["profile", "instance"]).agg(
        mean_reduction_pct=("s_alns_cost_reduction_pct", "mean"),
        std_reduction_pct=("s_alns_cost_reduction_pct", "std"),
        min_reduction_pct=("s_alns_cost_reduction_pct", "min"),
        wins=("s_alns_better", "sum"),
        seeds=("seed", "count"),
    ).reset_index()
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "run_manifest.json").write_text(json.dumps({
        "instances": [item[0] for item in instances],
        "profiles": profiles,
        "methods": methods,
        "seeds": seeds,
        "n_iter": args.n_iter,
        "all_valid": bool(results.valid.all()),
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
