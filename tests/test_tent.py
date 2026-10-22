"""Tests for the smooth tent-map experiment."""
import unittest

import numpy as np

from src.expansive.tent import (
    BITS,
    M,
    empirical_lambda,
    key_params,
    observe,
    random_states,
    rollout,
    state_from_bits,
    step,
)
from src.expansive.tent_student import (
    eval_tent,
    make_pairs,
    rollout_tent,
    t95,
    top_grid,
    train_tent,
)


class TestMap(unittest.TestCase):
    def test_deterministic_integer(self):
        a, offsets = key_params(123)
        s = np.array([0, 1, M // 2, M - 1], dtype=np.uint32)
        x = step(s, a, offsets)
        y = step(s, a, offsets)
        self.assertTrue(np.array_equal(x, y))
        self.assertEqual(x.dtype, np.uint32)
        self.assertTrue(np.all(x < M))
        self.assertTrue(np.issubdtype(offsets.dtype, np.integer))

    def test_observation_roundtrip(self):
        s = random_states(1000, 0)
        self.assertTrue(np.array_equal(state_from_bits(observe(s)), s))

    def test_expansion(self):
        a, offsets = key_params(7)
        s = random_states(500, 1)
        s2 = s ^ np.uint32(1)
        for _ in range(8):
            s = step(s, a, offsets)
            s2 = step(s2, a, offsets)
        diff = np.bitwise_xor(s, s2).view(np.uint8).reshape(-1, 4)
        bits = np.unpackbits(diff, axis=1).sum(axis=1)
        self.assertGreater(float(bits.mean()), 4.0)
        self.assertGreater(empirical_lambda(a, offsets, n=200, steps=8), 0.1)


class TestStudent(unittest.TestCase):
    def test_tiny_train_and_rollout(self):
        a, offsets = key_params(21)
        x, y = make_pairs(21, 5000, 0, a, offsets)
        model, _ = train_tent(x, y, steps=500, batch=128, seed=0, hidden=128)
        xt, yt = make_pairs(21, 1000, 99, a, offsets)
        m = eval_tent(model, xt, yt)
        self.assertGreater(m["top_grid_accuracy"], 0.8)
        self.assertLess(m["rmse"], 0.05)
        s0 = random_states(10, 5)
        pred = rollout_tent(model, s0, 8)
        exact = np.empty((10, 8, 4, 4), dtype=np.uint8)
        s = s0.copy()
        for t in range(8):
            s = step(s, a, offsets)
            exact[:, t] = top_grid(s)
        self.assertEqual(pred.shape, exact.shape)
        self.assertGreaterEqual(t95(pred, exact), 1)


if __name__ == "__main__":
    unittest.main()
