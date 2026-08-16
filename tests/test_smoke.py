from pathlib import Path
import unittest

import yaml

from scripts.run_experiment import apply_profile, service_config
from src.data_loader import build_travel_times, load_instance
from src.initialization import (
    four_phase_initialization,
    four_phase_initialization_service_guided,
)
from src.service_structure import build_service_structure
from src.structures import ID_GEN
from src.validator import validate_solution


ROOT = Path(__file__).resolve().parent.parent


class InitializationSmokeTest(unittest.TestCase):
    def load(self):
        nodes = ROOT / "data" / "instances" / "instance_20260103_nodes.csv"
        params = ROOT / "data" / "instances" / "instance_20260103_params.csv"
        data = load_instance(str(nodes), str(params), instance_id="smoke")
        apply_profile(data, "emergency")
        return data, build_travel_times(data)

    def test_baseline_initialization_is_complete_and_valid(self):
        ID_GEN.reset()
        data, tt = self.load()
        solution = four_phase_initialization(data, tt)
        self.assertEqual(solution.unserved, set())
        self.assertTrue(validate_solution(solution, data, tt).ok)

    def test_s_alns_initialization_is_complete_and_valid(self):
        ID_GEN.reset()
        data, tt = self.load()
        cfg = yaml.safe_load((ROOT / "config" / "amortized.yaml").read_text())
        s_cfg = service_config(cfg, seed=42)
        structure = build_service_structure(data, tt, s_cfg)
        solution = four_phase_initialization_service_guided(data, tt, structure)
        self.assertEqual(solution.unserved, set())
        self.assertTrue(validate_solution(solution, data, tt).ok)


if __name__ == "__main__":
    unittest.main()
