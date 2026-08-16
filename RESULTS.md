# Reference results and interpretation

## Featured emergency case

`instance_20260103`, emergency profile, 500 iterations, five paired seeds:

| Method | Mean objective | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| ALNS | 3274.522727 | 40.275753 | 3231.769557 | 3317.906969 |
| S-alns | 2453.645645 | 76.978472 | 2398.532472 | 2589.534660 |

The paired S-alns cost reduction is 25.0524% on average, with a standard
deviation of 2.7924 percentage points and a range of 20.1550%--26.9449%.
S-alns wins all five paired seeds.

## Strongest commercial case

`instance_20260109`, commercial profile, 500 iterations, five paired seeds:

| Method | Mean objective | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| ALNS | 3474.312703 | 0.014220 | 3474.306343 | 3474.338141 |
| S-alns | 2648.718036 | 48.684974 | 2562.916525 | 2674.986200 |

The paired S-alns cost reduction is 23.7628% on average, with a standard
deviation of 1.4012 percentage points. S-alns wins all five paired seeds.

## Full Figure 1 summary

Across the ten instances, the instance-level mean S-alns reduction is 13.79% in
the commercial profile and 17.49% in the emergency profile. S-alns wins every
commercial instance and nine of ten emergency instances. The exception is
emergency instance 10, where the mean difference favors ALNS by 4.39%.

All twenty configuration-level means, standard deviations, seed ranges, and
win counts are provided in `results/reference/figure1_summary.csv`. The
unfavorable result is intentionally retained in the public data and plot.

The construction-only baselines show that the result is not reproduced by a
single simple priority rule:

- commercial EDD: mean reduction 6.56%, wins 8/10 instances;
- commercial minimum slack: mean reduction 3.20%, wins 5/10;
- emergency EDD: mean reduction -3.50%, wins 5/10;
- emergency minimum slack: mean reduction -0.04%, wins 4/10;
- nearest construction: wins 0/10 against the searched ALNS baseline in both
  profiles.

## Claim boundary

The featured case was chosen because it has the largest observed mean advantage
among the 20 released instance/profile combinations. It is suitable as a clear
code demonstration, but should not be presented as a randomly selected or
universally representative case. The complete Figure 1 data must remain
available alongside it.

The comparison establishes lower heuristic objective values under matched
budgets and seeds. It does not establish global optimality, a universal win over
ALNS, or direct comparability with classical 2E-VRP benchmark objectives.
