from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from sistema.core.behavior import corrupt_relative, corrupt_with_noise
from . import benchmark


class N20ConfigurationTests(unittest.TestCase):
    def test_grid_uses_contrasting_noise_at_matched_audit_probability(self):
        profile, scenarios = benchmark.load_profile()
        for fraction in ("m40", "m80"):
            high, high_entry = benchmark.scenario_args(profile, scenarios[f"{fraction}-alto"], 0)
            subtle, subtle_entry = benchmark.scenario_args(profile, scenarios[f"{fraction}-sutil"], 0)
            self.assertEqual(high.pop("noise_mode"), "multiplicative")
            self.assertEqual(subtle.pop("noise_mode"), "relative")
            self.assertEqual(high, subtle)
            high_entry.pop("id")
            subtle_entry.pop("id")
            self.assertEqual(high_entry, subtle_entry)
            self.assertEqual(high_entry["audit_p"], 0.2)

    def test_mapping_for_baseline_and_each_collusion_scenario(self):
        profile, scenarios = benchmark.load_profile()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for scenario in scenarios.values():
                for seed in profile["seeds"]:
                    benchmark.check_mapping(profile, scenario, seed, root, root)

    def test_strong_noise_for_synthetic_numeric_answers(self):
        for base in (-31, 0, 37):
            for seed in range(5):
                high = corrupt_with_noise(base, random.Random(seed))
                subtle = corrupt_relative(base, random.Random(seed), 0.2)
                self.assertGreaterEqual(abs(high), (abs(base) + 1) * 1000)
                self.assertLess(abs(subtle), abs(high))

    def test_live_fails_closed_without_explicit_approval(self):
        profile, scenarios = benchmark.load_profile()
        with patch.dict(benchmark.os.environ, {"TCC_LIVE_N20_APPROVED": ""}):
            with self.assertRaisesRegex(RuntimeError, "approval"):
                benchmark.live(profile, scenarios["baseline"], 0, None, None)


if __name__ == "__main__":
    unittest.main()
