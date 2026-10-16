"""Tests for the GA loop: short run finishes, checkpoint written."""

import json
import os
import tempfile
import unittest

import numpy as np

from ehd_ga.ga import default_popsize, run_ga


class TestGa(unittest.TestCase):
    def test_default_popsize(self):
        # 4 + floor(3 * ln(61)) = 4 + 12 = 16.
        self.assertEqual(default_popsize(61), 16)

    def test_short_run_serial(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_ga(
                max_generations=3,
                workers=1,
                seed=0,
                checkpoint_dir=tmp,
                popsize=4,
            )
            self.assertEqual(result["generations"], 3)
            self.assertEqual(result["evals"], 12)
            self.assertTrue(np.isfinite(result["best_fitness"]))
            self.assertIsNotNone(result["best_genome"])
            self.assertEqual(len(result["best_genome"]), 61)
            ckpt = os.path.join(tmp, "checkpoint.json")
            self.assertTrue(os.path.exists(ckpt))
            with open(ckpt, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["generation"], 3)
            self.assertEqual(saved["best_fitness"], result["best_fitness"])
            self.assertEqual(len(saved["best_genome"]), 61)

    def test_short_run_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_ga(
                max_generations=2,
                workers=2,
                seed=1,
                checkpoint_dir=tmp,
                popsize=4,
            )
            self.assertEqual(result["generations"], 2)
            self.assertTrue(np.isfinite(result["best_fitness"]))

    def test_max_evals_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_ga(
                max_generations=100,
                max_evals=5,
                workers=1,
                seed=0,
                checkpoint_dir=tmp,
                popsize=4,
            )
            self.assertLessEqual(result["evals"], 8)  # one full generation
            self.assertLess(result["generations"], 100)


if __name__ == "__main__":
    unittest.main()
