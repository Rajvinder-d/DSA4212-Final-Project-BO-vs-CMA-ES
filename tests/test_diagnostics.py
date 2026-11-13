"""Run offline: PYTHONPATH=src python -m unittest discover -s tests -p test_diagnostics.py."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from bo_vs_es import ROOT
from bo_vs_es.diagnostics.core import (
    BudgetExhausted, CountedResidual, ball_reference, uniform_ball, validate_config,
)


class DiagnosticControls(unittest.TestCase):
    def settings(self):
        return dict(starts=3, seed=710, total_calls=300, maxiter=100,
                    ftol=1e-10, difference_step=1e-5)

    def test_ball_fit_when_unconstrained_solution_is_outside(self):
        result = ball_reference(lambda x: x - np.array([2., 0.]), 2, self.settings())
        self.assertIsNotNone(result["best"])
        point = np.array(result["best"]["u"])
        self.assertLessEqual(np.linalg.norm(point), 1.)
        np.testing.assert_allclose(point, [1., 0.], atol=2e-4)
        self.assertAlmostEqual(result["best"]["loss"], .5, places=6)

    def test_ball_fit_recovers_interior_solution(self):
        target = np.array([.2, -.1, .3])
        result = ball_reference(lambda x: x - target, 3, self.settings())
        np.testing.assert_allclose(result["best"]["u"], target, atol=1e-5)

    def test_count_includes_derivative_and_infeasible_trials(self):
        calls = []
        def residual(x):
            calls.append(x.copy())
            return x - np.array([2., 0.])
        result = ball_reference(residual, 2, self.settings())
        self.assertEqual(len(calls), result["evaluations"])
        self.assertEqual(len(calls), len(result["trace"]))
        self.assertLessEqual(len(calls), result["budget"])
        feasible = [r["loss"] for r in result["trace"] if r["feasible"] and r["status"] == "ok"]
        self.assertEqual(min(feasible), result["best"]["loss"])

    def test_failed_call_consumes_budget(self):
        counter = CountedResidual(lambda x: np.array([np.nan]), 1)
        with self.assertRaises(ValueError):
            counter(np.zeros(2))
        self.assertEqual(len(counter.rows), 1)
        self.assertEqual(counter.rows[0]["status"], "failed")
        with self.assertRaises(BudgetExhausted):
            counter(np.zeros(2))
        result = ball_reference(lambda x: np.array([np.nan]), 2, self.settings())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["trace"][0]["status"], "failed")

    def test_infeasible_optimum_cannot_be_recommended(self):
        counter = CountedResidual(lambda x: x - np.array([2., 0.]), 3)
        counter(np.array([2., 0.]))
        self.assertIsNone(counter.best)
        counter(np.zeros(2))
        self.assertEqual(counter.best["u"], [0., 0.])

    def test_wall_deadline_prevents_simulation(self):
        counter = CountedResidual(lambda x: self.fail("Must not call residual"), 10,
                                  deadline=time.monotonic() - 1)
        with self.assertRaises(TimeoutError):
            counter(np.zeros(2))
        self.assertEqual(counter.rows, [])

    def test_budget_interrupt_preserves_feasible_best(self):
        settings = self.settings()
        settings["total_calls"] = 12
        result = ball_reference(lambda x: x - np.array([2., 0.]), 2, settings)
        self.assertLessEqual(result["evaluations"], 12)
        self.assertIsNotNone(result["best"])
        self.assertTrue(any(not t["success"] for t in result["terminations"]))

    def test_sampling_preserves_ball_and_seed(self):
        a = uniform_ball(np.random.default_rng(1), 1000, 5)
        b = uniform_ball(np.random.default_rng(1), 1000, 5)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(np.linalg.norm(a, axis=1) <= 1))

    def test_confirmation_seeds_cannot_enter_development(self):
        config = json.loads((ROOT / "configs/diagnostics/development_v2.json").read_text())
        validate_config(config)
        config["initial_design_seeds"].append(10)
        with self.assertRaisesRegex(ValueError, "reserved confirmation"):
            validate_config(config)

    def test_preflight_does_not_create_data(self):
        spec = importlib.util.spec_from_file_location("diagnostic_cli", ROOT / "scripts/run_diagnostics.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            check = module.readiness(root)
            self.assertFalse(check["ready"])
            self.assertTrue(any("Missing prepared CALCE" in b for b in check["blockers"]))
            self.assertEqual(list(root.iterdir()), [])


HAS_GP = all(importlib.util.find_spec(m) is not None
             for m in ["torch", "botorch", "gpytorch", "pandas", "cma"])


@unittest.skipUnless(HAS_GP, "Torch/BoTorch/GPyTorch/pandas/CMA are required")
class GPReplayIntegration(unittest.TestCase):
    def test_replay_matches_v1_identity_candidate(self):
        from bo_vs_es.diagnostics.replay import replay_history
        from bo_vs_es.objectives.models import Domain, Quadratic
        from bo_vs_es.optimisers.adapters import bo_candidate
        config = json.loads((ROOT / "configs/diagnostics/development_v2.json").read_text())
        config["acquisition_maxiter"] = [15]
        config["query_points"] = 10
        settings = dict(fit_maxiter=10, acq_maxiter=15, raw_samples=24, num_restarts=2)
        domain = Domain("ball", np.eye(5))
        points = domain.sample(np.random.default_rng(0), 20)
        losses = [Quadratic(100)(u) for u in points]
        result = replay_history(points, losses, 101, config, settings)
        expected, _ = bo_candidate(points, losses, domain, False, settings,
                                    np.random.default_rng(config["acquisition_start_seed"]))
        np.testing.assert_allclose(result["acquisition"][0]["identity"]["u"], expected, atol=1e-5)
        self.assertLess(result["frozen_control"]["posterior_mean_max_abs"], 1e-10)
        self.assertEqual(result["additional_simulator_calls"], 0)

    def test_frozen_scalar_predictions_are_rotation_invariant(self):
        from bo_vs_es.diagnostics.replay import scalar_model, predictions, differences
        from bo_vs_es.objectives.models import rotation
        points = uniform_ball(np.random.default_rng(0), 20, 5)
        queries = uniform_ball(np.random.default_rng(1), 10, 5)
        losses = np.sum(points ** 2, axis=1)
        q = rotation(101)
        errors = differences(predictions(scalar_model(points, losses), queries),
                             predictions(scalar_model(points @ q, losses), queries @ q))
        self.assertLess(errors["posterior_mean_max_abs"], 1e-10)


if __name__ == "__main__":
    unittest.main()
