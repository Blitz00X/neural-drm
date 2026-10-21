"""CPU tests for the decision pipeline (no GPU needed)."""
import unittest

import numpy as np
import torch

from src.decision.models import ConcatKeyed, FilmKeyed, GatedPair, PlainWorld
from src.decision.trainer import (
    build_pools,
    hamming_distance,
    sample_keys,
    seed_all,
    stratum_keys,
    train,
)
from src.decision.verdict import compute_verdict


class TestKeys(unittest.TestCase):
    def test_strata_hamming(self):
        secret = np.ones(64, dtype=np.float32)
        rng = np.random.default_rng(0)
        h1 = stratum_keys(secret, 50, "h1", rng)
        h2 = stratum_keys(secret, 50, "h2", rng)
        self.assertTrue(all(hamming_distance(k, secret) == 1 for k in h1))
        self.assertTrue(all(hamming_distance(k, secret) == 2 for k in h2))
        self.assertTrue(all(hamming_distance(k, secret) > 2 for k in
                            stratum_keys(secret, 50, "random", np.random.default_rng(1))))

    def test_sample_mixture(self):
        secret = torch.ones(64)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        keys, valid = sample_keys(secret, 20000, gen)
        self.assertAlmostEqual(float(valid.float().mean()), 0.5, delta=0.02)
        distances = (keys != secret).sum(1).numpy()
        self.assertGreater(int((distances == 1).sum()), 2000)
        self.assertGreater(int((distances == 2).sum()), 2000)
        self.assertGreater(int((distances > 2).sum()), 3000)
        self.assertTrue(torch.all(keys[valid] == secret))


class TestModels(unittest.TestCase):
    def test_shapes_and_gate(self):
        x = torch.rand(8, 2)
        secret = torch.ones(64)
        key = secret.expand(8, -1).clone()
        b1 = ConcatKeyed(width=16)
        b2 = FilmKeyed(width=16)
        self.assertEqual(b1(x, key).shape, (8, 5))
        self.assertEqual(b2(x, key).shape, (8, 5))
        wrong = key.clone()
        wrong[:, 0] *= -1
        self.assertFalse(torch.allclose(b2(x, key), b2(x, wrong)))

        f, g = PlainWorld(width=16), PlainWorld(width=16)
        gated = GatedPair(f, g, secret)
        self.assertTrue(torch.allclose(gated(x, key), f(x)))
        self.assertTrue(torch.allclose(gated(x, wrong), g(x)))
        self.assertTrue(torch.allclose(gated(x, wrong, gate_open=True), f(x)))

    def test_output_range(self):
        model = PlainWorld(width=8)
        out = model(torch.rand(4, 2))
        self.assertTrue(torch.all(out[:, 0] >= 0) and torch.all(out[:, 0] <= 1))


class TestTraining(unittest.TestCase):
    def test_tiny_keyed_run(self):
        data = build_pools(0, 4, train_n=64, val_n=16, test_n=200, oracle_n=32)
        model = ConcatKeyed(width=8)
        secret = torch.ones(64)
        tensors = {
            k: torch.as_tensor(data[k])
            for k in ("x", "fh", "fb", "gh", "gb")
        }
        record = train(
            model,
            tensors,
            steps=3,
            batch=8,
            lr=1e-3,
            seed=0,
            keyed=True,
            secret=secret,
            device="cpu",
            log_every=1,
        )
        self.assertEqual(record["steps"], 3)
        self.assertGreater(record["flops"], 0)
        self.assertTrue(all(np.isfinite(h["loss"]) for h in record["history"]))


def _rung(arm="B1", width=256, steps=20000):
    return {
        "arm": arm,
        "width": width,
        "steps": steps,
        "octaves": 8,
        "accepted": True,
    }


def _attacks(min_ns, kr=False):
    seeds = {
        str(i): {
            "finetune": {
                "min_n": n,
                "rows": [] if n is None else [
                    {"n": n, "cumulative_gpu_seconds": 3600.0, "test_keys": []}
                ],
            },
            "key_recovery": {"success": kr},
        }
        for i, n in enumerate(min_ns)
    }
    return {"primary": {"arm": "B1", "width": 256, "steps": 20000, "seeds": seeds},
            "upper": None}


class TestVerdict(unittest.TestCase):
    def test_dead_capacity(self):
        summary = {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [],
            "attacks": None,
        }
        self.assertEqual(compute_verdict(summary)["label"], "DEAD_CAPACITY")

    def test_dead_finetune(self):
        summary = {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [_rung()],
            "attacks": _attacks([1000, 1000, None]),
        }
        self.assertEqual(compute_verdict(summary)["label"], "DEAD_FINETUNE")

    def test_dead_key(self):
        summary = {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [_rung()],
            "attacks": _attacks([None, None, None], kr=True),
        }
        self.assertEqual(compute_verdict(summary)["label"], "DEAD_KEY")

    def test_capped(self):
        summary = {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 0.2, "min_n": 10000},
            "ladder": [_rung()],
            "attacks": _attacks([None, None, None]),
        }
        self.assertEqual(compute_verdict(summary)["label"], "CAPPED")

    def test_alive_weak(self):
        summary = {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [_rung()],
            "attacks": _attacks([5000, 5000, None]),
        }
        self.assertEqual(compute_verdict(summary)["label"], "ALIVE_WEAK")

    def test_alive(self):
        summary = {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [_rung()],
            "attacks": _attacks([100000, None, None]),
        }
        self.assertEqual(compute_verdict(summary)["label"], "ALIVE")

    def test_inconclusive_without_attack(self):
        summary = {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [_rung()],
            "attacks": None,
        }
        self.assertEqual(compute_verdict(summary)["label"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
