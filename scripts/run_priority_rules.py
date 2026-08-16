"""Deterministic construction-rule baselines for the ten VLHC instances.

The experiment changes only the customer-priority rule used by Phase 1 of the
published four-phase construction heuristic. Hub coverage, trip insertion,
Phases 2--4, feasibility checks, and the objective are shared with the main
solver. The three rules are:

* ``nearest``: increasing hub--customer UGV travel time (the original rule);
* ``edd``: earliest customer due date first;
* ``min_slack``: smallest direct-service time-window slack first.

These are construction-only baselines, so they are deterministic and are not
reported with artificial seed variance.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_INSTANCE_DIR = PROJECT_ROOT / "data" / "instances"
CAPITAL_FIELDS = ("cost_H", "cost_D_inv", "cost_K_inv")
from src.costs import compute_cost  # noqa: E402
from src.data_loader import (  # noqa: E402
    build_travel_times,
    load_instance,
    validate_candidate_hubs_for_uav,
)
from src.initialization import (  # noqa: E402
    phase2_generate_groutes,
    phase3_generate_atrips,
    phase4_generate_aroutes,
)
from src.structures import Gtrip, ID_GEN, Solution  # noqa: E402
from src.timing_subproblem import compute_gtrip_window, refresh_all_timing  # noqa: E402
from src.validator import validate_solution  # noqa: E402


LOG = logging.getLogger("priority_rule_baselines")
RULES = ("nearest", "edd", "min_slack")
EPS = 1e-9


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_instances(instance_dir: Path, requested: str):
    candidates = []
    for nodes_path in sorted(instance_dir.glob("*_nodes.csv")):
        stem = nodes_path.name.removesuffix("_nodes.csv")
        params_path = instance_dir / f"{stem}_params.csv"
        if params_path.is_file():
            candidates.append((stem, nodes_path, params_path))
    if requested.strip().lower() == "featured":
        requested = "20260103"
    if requested.strip().lower() == "all":
        return candidates
    tokens = parse_csv_list(requested)
    selected = [item for item in candidates if any(token in item[0] for token in tokens)]
    if not selected:
        raise ValueError(f"No instance matched --instances={requested!r}")
    return selected


def apply_cost_profile(data, profile: str) -> dict:
    if profile == "commercial":
        policy = "capital and running costs"
    elif profile == "emergency":
        for field in CAPITAL_FIELDS:
            setattr(data, field, 0.0)
        policy = "running costs only; capital costs set to zero"
    else:
        raise ValueError(f"unknown profile: {profile}")
    return {"profile": profile, "policy": policy}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic nearest/EDD/min-slack construction baselines."
    )
    parser.add_argument("--instance-dir", type=Path, default=DEFAULT_INSTANCE_DIR)
    parser.add_argument("--instances", default="featured",
                        help="'featured' (instance 03), 'all', or comma-separated IDs.")
    parser.add_argument("--profiles", default="commercial,emergency")
    parser.add_argument("--rules", default=",".join(RULES))
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "results" / "generated" / "priority_rules",
    )
    return parser.parse_args()


def priority_key(data, tt, hub: int, rule: str) -> Callable[[int], tuple[float, ...]]:
    if rule == "nearest":
        return lambda j: (float(tt.t_hj_K[hub][j]), float(data.customers[j].t_due), j)
    if rule == "edd":
        return lambda j: (
            float(data.customers[j].t_due),
            float(data.customers[j].t_rel),
            float(tt.t_hj_K[hub][j]),
            j,
        )
    if rule == "min_slack":
        return lambda j: (
            float(
                data.customers[j].t_due
                - data.customers[j].t_rel
                - tt.t_hD[hub]
                - tt.t_hj_K[hub][j]
            ),
            float(data.customers[j].t_due),
            float(tt.t_hj_K[hub][j]),
            j,
        )
    raise ValueError(f"unknown priority rule: {rule}")


def phase1_priority(data, tt, rule: str, r_c: float = 1.0 / 3.0):
    """Phase 1 with the customer selector replaced by a priority rule."""
    unserved = set(data.J)
    active_hubs: set[int] = set()
    gtrips: list[Gtrip] = []
    cust_to_hub: dict[int, int] = {}

    def reachable(hub: int, customers: set[int], use_radius: bool) -> set[int]:
        result = set()
        for customer in customers:
            t_to = tt.t_hj_K[hub][customer]
            if 2.0 * t_to > data.e_K + EPS:
                continue
            direct_arrival = data.customers[customer].t_rel + tt.t_hD[hub] + t_to
            if direct_arrival > data.customers[customer].t_due + EPS:
                continue
            if use_radius and t_to > r_c * data.e_K + EPS:
                continue
            result.add(customer)
        return result

    while unserved:
        best_hub = None
        best_coverage = -1
        for hub in data.H:
            if hub in active_hubs:
                continue
            coverage = len(reachable(hub, unserved, use_radius=True))
            if coverage > best_coverage:
                best_coverage, best_hub = coverage, hub
        if best_hub is None or best_coverage <= 0:
            for hub in data.H:
                if hub in active_hubs:
                    continue
                coverage = len(reachable(hub, unserved, use_radius=False))
                if coverage > best_coverage:
                    best_coverage, best_hub = coverage, hub
        if best_hub is None or best_coverage <= 0:
            LOG.warning("No inactive hub covers the remaining %d customers", len(unserved))
            break

        active_hubs.add(best_hub)
        choose_customer = priority_key(data, tt, best_hub, rule)

        while True:
            reachable_now = reachable(best_hub, unserved, use_radius=False)
            if not reachable_now:
                break

            customer = min(reachable_now, key=choose_customer)
            new_trip = Gtrip(hub=best_hub, seq=[customer])
            data.customers[customer].t_rel_H = (
                data.customers[customer].t_rel + tt.t_hD[best_hub]
            )
            compute_gtrip_window(new_trip, data, tt)
            if not new_trip.feasible:
                unserved.discard(customer)
                continue
            gtrips.append(new_trip)
            cust_to_hub[customer] = best_hub
            unserved.discard(customer)

            while True:
                reachable_now = reachable(best_hub, unserved, use_radius=False)
                if not reachable_now:
                    break
                customer = min(reachable_now, key=choose_customer)
                data.customers[customer].t_rel_H = (
                    data.customers[customer].t_rel + tt.t_hD[best_hub]
                )

                inserted = False
                for trip in gtrips:
                    if trip.hub != best_hub:
                        continue
                    best_pos, best_eta = None, float("inf")
                    original_seq = trip.seq[:]
                    for pos in range(len(original_seq) + 1):
                        trial_seq = original_seq[:pos] + [customer] + original_seq[pos:]
                        trial_eta = tt.gtrip_eta(trip.hub, trial_seq)
                        trial_load = trip.load + data.customers[customer].weight
                        if trial_eta > data.e_K + EPS or trial_load > data.l_K + EPS:
                            continue
                        old_seq, old_load, old_eta = trip.seq, trip.load, trip.eta
                        trip.seq, trip.load = trial_seq, trial_load
                        compute_gtrip_window(trip, data, tt)
                        feasible = trip.feasible
                        trip.seq, trip.load, trip.eta = old_seq, old_load, old_eta
                        compute_gtrip_window(trip, data, tt)
                        if feasible and trial_eta < best_eta:
                            best_pos, best_eta = pos, trial_eta
                    if best_pos is not None:
                        trip.seq.insert(best_pos, customer)
                        trip.load += data.customers[customer].weight
                        compute_gtrip_window(trip, data, tt)
                        cust_to_hub[customer] = best_hub
                        unserved.discard(customer)
                        inserted = True
                        break

                if not inserted:
                    new_trip = Gtrip(hub=best_hub, seq=[customer])
                    compute_gtrip_window(new_trip, data, tt)
                    if not new_trip.feasible:
                        unserved.discard(customer)
                        continue
                    gtrips.append(new_trip)
                    cust_to_hub[customer] = best_hub
                    unserved.discard(customer)

    return gtrips, active_hubs, cust_to_hub


def construct_solution(data, tt, rule: str) -> Solution:
    gtrips, active_hubs, cust_to_hub = phase1_priority(data, tt, rule)
    groutes = phase2_generate_groutes(gtrips, active_hubs, data, tt)
    atrips = phase3_generate_atrips(gtrips, active_hubs, data, tt, cust_to_hub)
    aroutes = phase4_generate_aroutes(atrips, data, tt)
    solution = Solution(
        active_hubs=active_hubs,
        aroutes=aroutes,
        groutes=groutes,
        unserved=set(data.J) - set(cust_to_hub),
    )
    solution.rebuild_cust_indexes()
    refresh_all_timing(solution, data, tt)
    return solution


def run_one(instance, nodes_path, params_path, profile: str, rule: str) -> dict:
    ID_GEN.reset()
    data = load_instance(str(nodes_path), str(params_path), instance_id=instance)
    profile_meta = apply_cost_profile(data, profile)
    tt = build_travel_times(data)
    validate_candidate_hubs_for_uav(data, tt, strict=False)
    started = time.perf_counter()
    solution = construct_solution(data, tt, rule)
    elapsed = time.perf_counter() - started
    validation = validate_solution(solution, data, tt)
    costs = compute_cost(solution, data, tt).as_dict()
    return {
        "instance": instance,
        "profile": profile,
        "method": f"priority_{rule}",
        "rule": rule,
        "objective": costs["total"],
        "capital_cost": costs["hub_cost"] + costs["uav_inv_cost"] + costs["ugv_inv_cost"],
        "running_cost": costs["uav_trav_cost"] + costs["ugv_trav_cost"],
        "hub_cost": costs["hub_cost"],
        "uav_inv_cost": costs["uav_inv_cost"],
        "ugv_inv_cost": costs["ugv_inv_cost"],
        "uav_trav_cost": costs["uav_trav_cost"],
        "ugv_trav_cost": costs["ugv_trav_cost"],
        "num_active_hubs": len(solution.active_hubs),
        "num_uav_routes": len(solution.aroutes),
        "num_ugv_routes": len(solution.groutes),
        "served": len(data.J) - len(solution.unserved),
        "unserved": len(solution.unserved),
        "valid": bool(validation.ok),
        "runtime_sec": elapsed,
        "profile_policy": profile_meta["policy"],
        "validation_message": validation.render(),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rules = parse_csv_list(args.rules)
    unknown = sorted(set(rules) - set(RULES))
    if unknown:
        raise ValueError(f"unknown rules: {unknown}; available={RULES}")
    profiles = parse_csv_list(args.profiles)
    instances = discover_instances(args.instance_dir, args.instances)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for instance, nodes_path, params_path in instances:
        for profile in profiles:
            for rule in rules:
                LOG.info("%s | %s | %s", instance, profile, rule)
                rows.append(run_one(instance, nodes_path, params_path, profile, rule))

    results = pd.DataFrame(rows)
    results.to_csv(args.out / "priority_results.csv", index=False)
    manifest = {
        "design": "deterministic construction-only priority rules",
        "instances": [name for name, _, _ in instances],
        "profiles": profiles,
        "rules": rules,
        "rows": len(results),
        "all_valid": bool(results["valid"].all()),
        "note": "No seed error bars: each priority rule is deterministic.",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(results.groupby(["profile", "method"])["objective"].mean().round(3))
    print(f"Wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
