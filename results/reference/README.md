# Reference results

These CSV files are the committed 500-iteration reference results used to draw
the bundled Figure 1.

- `run_results.csv`: 10 instances x 2 profiles x 2 methods x 5 seeds;
- `paired_method_comparison.csv`: seed-matched ALNS--S-alns contrasts;
- `priority_results.csv`: deterministic nearest, EDD, and minimum-slack
  construction baselines.
- `figure1_summary.csv`: one transparent summary row for each of the twenty
  instance/profile configurations, including the unfavorable configuration.

The files allow the figure to be regenerated without rerunning the full search.
They are reference measurements, not claimed global optima.
