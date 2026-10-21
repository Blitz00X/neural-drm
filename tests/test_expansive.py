"""Tests for the keyed CA distillation experiment (CPU only)."""
import unittest

import numpy as np

from src.expansive.ca import (
    empirical_lambda,
    random_grids,
    rollout,
    rule_table,
    sensitivity,
    step,
)
from src.expansive.ca_student import (
    eval_ca,
    make_pairs_ca,
    rollout_ca,
    t95,
    train_ca,
    train_multistep_ca,
)


class TestCA(unittest.TestCase):
    def test_rule_table_shapes(self):
        t3 = rule_table(123, 3, 0)
        t5 = rule_table(123, 5, 0)
        self.assertEqual(t3.shape, (512,))
        self.assertEqual(t5.shape, (33_554_432,))
        self.assertTrue(np.array_equal(t3, rule_table(123, 3, 0)))
        self.assertFalse(np.array_equal(t3, rule_table(124, 3, 0)))
        self.assertTrue(np.all((t3 == 0) | (t3 == 1)))

    def test_step_deterministic(self):
        for k in (3, 5):
            table = rule_table(7, k, 0)
            g = random_grids(20, 0)
            a = step(g, table, k)
            b = step(g, table, k)
            self.assertTrue(np.array_equal(a, b))
            self.assertEqual(a.shape, g.shape)
            self.assertEqual(a.dtype, np.uint8)
            self.assertTrue(np.all((a == 0) | (a == 1)))

    def test_rollout(self):
        table = rule_table(9, 3, 0)
        g = random_grids(10, 0)
        traj = rollout(g, 6, table, 3)
        self.assertEqual(traj.shape, (10, 6, 8, 8))

    def test_lambda_and_sensitivity(self):
        table = rule_table(11, 3, 0)
        lam = empirical_lambda(table, 3, n=200, steps=8, seed=0)
        self.assertTrue(np.isfinite(lam))
        self.assertGreater(lam, 0.0)
        sens = sensitivity(table, 3, samples=300, seed=0)
        self.assertGreaterEqual(sens, 0.0)
        self.assertLessEqual(sens, 1.0)


class TestStudent(unittest.TestCase):
    def test_tiny_train_and_rollout(self):
        k = 3
        table = rule_table(21, k, 0)
        x, y = make_pairs_ca(table, k, 5000, seed=0)
        model, _ = train_ca(x, y, k, steps=500, batch=128, lr=1e-3, seed=0, channels=64, depth=2)
        xt, yt = make_pairs_ca(table, k, 1000, seed=99)
        m = eval_ca(model, xt, yt)
        self.assertGreater(m["cell_accuracy"], 0.6)
        g0 = random_grids(10, 5)
        pred = rollout_ca(model, g0, 8)
        exact = rollout(g0, 8, table, k)
        self.assertEqual(pred.shape, exact.shape)
        self.assertGreaterEqual(t95(pred, exact), 1)

    def test_multistep_shapes(self):
        k = 3
        table = rule_table(31, k, 0)
        x, y = make_pairs_ca(table, k, 2000, seed=0)
        model, _ = train_ca(x, y, k, steps=20, batch=64, lr=1e-3, seed=0, channels=16, depth=1)
        g0 = random_grids(16, 3)
        traj = np.concatenate([g0[:, None], rollout(g0, 6, table, k)], axis=1)
        rec = train_multistep_ca(model, traj, steps=3, batch=8, horizon=3, lr=1e-4, seed=0)
        self.assertEqual(rec["horizon"], 3)
        self.assertTrue(np.isfinite(rec["history"][-1]["loss"]))


if __name__ == "__main__":
    unittest.main()
