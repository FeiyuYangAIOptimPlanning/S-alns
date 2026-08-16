"""Plot Figure 1 from generated or bundled reference CSV files."""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
COLORS = {
    "baseline": "#4C566A",
    "priority_edd": "#D08770",
    "priority_min_slack": "#B48EAD",
    "priority_nearest": "#8FBC8F",
    "s_alns": "#2F6B8A",
}
LABELS = {
    "baseline": "ALNS",
    "priority_edd": "EDD",
    "priority_min_slack": "Slack",
    "priority_nearest": "Nearest",
    "s_alns": "S-alns",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-results", type=Path,
        default=ROOT / "results" / "reference" / "run_results.csv",
    )
    parser.add_argument(
        "--priority-results", type=Path,
        default=ROOT / "results" / "reference" / "priority_results.csv",
    )
    parser.add_argument("--profiles", default="commercial,emergency")
    parser.add_argument("--instances", default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "figure1_reproduced")
    return parser.parse_args()


def stable_seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8")) & 0xFFFFFFFF


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#ECEFF4", linewidth=0.55, zorder=0)
    ax.tick_params(length=2.5, width=0.6)


def select_instances(data: pd.DataFrame, requested: str) -> list[str]:
    available = sorted(data.instance.unique())
    if requested.lower() == "all":
        return available
    if requested.lower() == "featured":
        requested = "20260103"
    tokens = [item.strip() for item in requested.split(",") if item.strip()]
    selected = [name for name in available if any(token in name for token in tokens)]
    if not selected:
        raise ValueError(f"No instance matched {requested!r}")
    return selected


def main() -> None:
    args = parse_args()
    search = pd.read_csv(args.search_results)
    rules = pd.read_csv(args.priority_results) if args.priority_results.is_file() else pd.DataFrame()
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    search = search[search.profile.isin(profiles)].copy()
    instances = select_instances(search, args.instances)
    search = search[search.instance.isin(instances)]
    if not rules.empty:
        rules = rules[rules.profile.isin(profiles) & rules.instance.isin(instances)]

    baseline_means = (
        search[search.method == "baseline"]
        .groupby(["profile", "instance"]).objective.mean()
    )
    methods = ["baseline"]
    if not rules.empty:
        methods += ["priority_edd", "priority_min_slack", "priority_nearest"]
    methods += ["s_alns"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0,
        "axes.labelsize": 8.2, "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2, "legend.fontsize": 7.1,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(len(profiles), 1, figsize=(7.15, 2.2 * len(profiles)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    x = np.arange(len(instances), dtype=float)
    width = min(0.16, 0.82 / len(methods))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    records = []

    for panel, (ax, profile) in enumerate(zip(axes, profiles)):
        for method_index, method in enumerate(methods):
            heights, lower, upper = [], [], []
            for instance in instances:
                denominator = baseline_means.loc[(profile, instance)]
                if method.startswith("priority_"):
                    raw = rules[(rules.profile == profile) & (rules.instance == instance)
                                & (rules.method == method)].objective.to_numpy(float)
                    stochastic = False
                else:
                    raw = search[(search.profile == profile) & (search.instance == instance)
                                 & (search.method == method)].objective.to_numpy(float)
                    stochastic = True
                if len(raw) == 0:
                    raise ValueError(f"Missing {profile}/{instance}/{method}")
                normalized = 100.0 * raw / denominator
                mean = float(normalized.mean())
                lo, hi = bootstrap_ci(normalized, stable_seed(profile, instance, method))
                heights.append(mean)
                lower.append(mean - lo if stochastic else 0.0)
                upper.append(hi - mean if stochastic else 0.0)
                records.append({
                    "profile": profile,
                    "instance": instance,
                    "method": method,
                    "normalized_objective_mean": mean,
                    "ci95_low": lo if stochastic else np.nan,
                    "ci95_high": hi if stochastic else np.nan,
                    "replicates": len(normalized),
                    "stochastic": stochastic,
                })
            ax.bar(x + offsets[method_index], heights, width=width * 0.92,
                   color=COLORS[method], alpha=0.90, linewidth=0,
                   label=LABELS[method], zorder=2,
                   yerr=np.vstack([lower, upper]), capsize=1.4,
                   error_kw={"elinewidth": 0.65, "ecolor": COLORS[method]})
        ax.axhline(100, color=COLORS["baseline"], linewidth=0.7,
                   linestyle="--", zorder=1)
        ax.text(0.015, 0.98, f"({chr(97 + panel)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.2, fontweight="bold")
        ax.set_ylabel("Objective (% of ALNS)")
        clean_axis(ax)

    axes[-1].set_xlabel("Instance")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([name[-2:] for name in instances])
    axes[0].legend(frameon=False, ncol=len(methods), loc="upper center",
                   bbox_to_anchor=(0.56, 1.16), columnspacing=0.85,
                   handlelength=1.15, handletextpad=0.35)
    all_values = [record["normalized_objective_mean"] for record in records]
    y_low = min(70.0, np.floor(min(all_values) / 5.0) * 5.0)
    y_high = max(110.0, np.ceil(max(all_values) / 5.0) * 5.0 + 2.0)
    for ax in axes:
        ax.set_ylim(y_low, y_high)
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.45, h_pad=0.6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(args.out.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    pd.DataFrame(records).to_csv(args.out.parent / f"{args.out.name}_plot_data.csv", index=False)
    print(f"Wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
