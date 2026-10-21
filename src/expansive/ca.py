"""Keyed 8x8 binary cellular automaton with a k x k neighbourhood.

State: 8x8 bits.  Each cell's next state is a keyed Boolean lookup of its
k x k toroidal neighbourhood.  The rule table has 2^(k*k) bits, derived from a
64-bit key; k=3 is a 512-bit control, k=5 is a 33,554,432-bit high-description
rule.  The exact update is integer/bitwise.
"""
from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1
GRID = 8


def _seed(key: int, attempt: int) -> int:
    return (int(key) + attempt * 0x9E3779B97F4A7C15) & MASK64


def rule_table(key: int, k: int = 5, attempt: int = 0) -> np.ndarray:
    n = 1 << (k * k)
    rng = np.random.default_rng(_seed(key, attempt))
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def _pad(grid: np.ndarray, k: int) -> np.ndarray:
    return np.pad(grid, ((0, 0), (k - 1, k - 1), (k - 1, k - 1)), mode="wrap")


def step(grid: np.ndarray, table: np.ndarray, k: int = 5) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.uint8)
    pad = _pad(grid, k)
    n = len(grid)
    idx = np.zeros((n, GRID, GRID), dtype=np.uint32)
    bit = 0
    for dy in range(k):
        for dx in range(k):
            idx |= pad[:, dy : dy + GRID, dx : dx + GRID].astype(np.uint32) << bit
            bit += 1
    return table[idx]


def rollout(grid0: np.ndarray, steps: int, table: np.ndarray, k: int = 5) -> np.ndarray:
    s = np.asarray(grid0, dtype=np.uint8).copy()
    out = np.empty((len(s), steps, GRID, GRID), dtype=np.uint8)
    for t in range(steps):
        s = step(s, table, k)
        out[:, t] = s
    return out


def random_grids(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, (n, GRID, GRID), dtype=np.uint8)


def empirical_lambda(table: np.ndarray, k: int = 5, n: int = 300, steps: int = 8, seed: int = 0) -> float:
    g = random_grids(n, seed)
    g2 = g.copy()
    g2[:, GRID // 2, GRID // 2] ^= 1
    dist = []
    for _ in range(steps):
        g = step(g, table, k)
        g2 = step(g2, table, k)
        dist.append(float(np.mean(np.sum(g != g2, axis=(1, 2)))))
    dist = np.asarray(dist)
    take = min(steps, 6)
    x = np.arange(1, take + 1)
    y = np.log(np.maximum(dist[:take], 1e-9))
    return float(np.polyfit(x, y, 1)[0])


def sensitivity(table: np.ndarray, k: int = 5, samples: int = 500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    patterns = rng.integers(0, len(table), samples, dtype=np.int64)
    changes = 0
    total = 0
    for p in patterns:
        base = table[int(p)]
        for bit in range(k * k):
            q = int(p) ^ (1 << bit)
            changes += int(table[q] != base)
            total += 1
    return changes / total
