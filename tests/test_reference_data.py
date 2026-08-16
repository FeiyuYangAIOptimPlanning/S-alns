from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


class ReferenceDataTest(unittest.TestCase):
    def test_all_twenty_configurations_are_released(self):
        cases = pd.read_csv(ROOT / "data" / "case_matrix.csv")
        summary = pd.read_csv(ROOT / "results" / "reference" / "figure1_summary.csv")
        self.assertEqual(len(cases), 20)
        self.assertEqual(len(summary), 20)
        self.assertEqual(cases[["instance", "profile"]].drop_duplicates().shape[0], 20)
        self.assertEqual(summary[["instance", "profile"]].drop_duplicates().shape[0], 20)
        self.assertEqual(int((~summary.s_alns_better_on_mean).sum()), 1)

    def test_summary_matches_seed_level_reference(self):
        paired = pd.read_csv(
            ROOT / "results" / "reference" / "paired_method_comparison.csv"
        )
        paired["reduction"] = -paired.delta_pct
        recomputed = (
            paired.groupby(["profile", "instance"]).reduction.mean().sort_index()
        )
        released = (
            pd.read_csv(ROOT / "results" / "reference" / "figure1_summary.csv")
            .set_index(["profile", "instance"]).mean_reduction_pct.sort_index()
        )
        np.testing.assert_allclose(recomputed.to_numpy(), released.to_numpy(), atol=1e-6)

    def test_all_reference_solutions_are_complete_and_valid(self):
        results = pd.read_csv(ROOT / "results" / "reference" / "run_results.csv")
        self.assertEqual(len(results), 200)
        self.assertTrue(results.valid.all())
        self.assertEqual(int(results.unserved.max()), 0)


if __name__ == "__main__":
    unittest.main()
