"""Keyed, integer-only, expansive piecewise-linear world map.

The map is deliberately *learnable one step at a time* (piecewise linear on a
32-bit state) but expansive: a one-bit perturbation grows over iterations.  The
legitimate game engine evaluates it exactly with integer arithmetic; the
attacker sees only (world_t, world_{t+1}) pairs.

State: uint32, interpreted as a 4x4 grid of 2-bit heights (16 cells).
Key:   64-bit, selects one of three integer slopes per piece and a 32-bit offset.
"""
from __future__ import annotations

import math

import numpy as np

MASK32 = np.uint32(0xFFFFFFFF)
MASK64 = (1 << 64) - 1

# Slopes in Q16.16, grouped by the target Lyapunov exponent.
# 1.2/1.25/1.3 -> ~0.18-0.26; 1.8/2.0/2.2 -> ~0.59-0.79; 2.7/3.0/3.3 -> ~0.99-1.19
SLOPE_TABLE = {
    0.2: [78643, 81920, 85197],
    0.6: [117965, 131072, 144179],
    1.0: [176947, 196608, 216269],
}


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def key_params(key: int, lam: float, pieces: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Integer slopes (Q16.16) and 32-bit offsets, one pair per piece."""
    if lam not in SLOPE_TABLE:
        raise ValueError(f"lam must be one of {sorted(SLOPE_TABLE)}")
    if pieces not in (16, 32, 64, 256, 1024):
        raise ValueError("pieces must be 16, 32, 64, 256 or 1024")
    table = SLOPE_TABLE[lam]
    slopes = np.empty(pieces, dtype=np.uint64)
    offsets = np.empty(pieces, dtype=np.uint64)
    for i in range(pieces):
        r = _splitmix64((int(key) + i * 0x9E3779B97F4A7C15) & MASK64)
        slopes[i] = table[r % len(table)]
        offsets[i] = _splitmix64(r ^ 0xD1B54A32D192ED03) & 0xFFFFFFFF
    return slopes, offsets


def _piece_index(states: np.ndarray, pieces: int) -> np.ndarray:
    bits = int(math.log2(pieces))
    return (states >> np.uint32(32 - bits)).astype(np.uint32)


def step(states: np.ndarray, slopes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """One exact integer step; no floating point in the computation."""
    states = np.asarray(states, dtype=np.uint32)
    pieces = len(slopes)
    idx = _piece_index(states, pieces)
    a = slopes[idx]
    b = offsets[idx]
    prod = (states.astype(np.uint64) * a) >> np.uint64(16)
    out = (prod + b) & np.uint64(MASK32)
    return out.astype(np.uint32)


def observe(states: np.ndarray) -> np.ndarray:
    """uint32 -> (n, 16) cells of 2-bit heights; exactly invertible."""
    states = np.asarray(states, dtype=np.uint32)
    shifts = np.arange(0, 32, 2, dtype=np.uint32)
    return ((states[:, None] >> shifts) & np.uint32(3)).astype(np.uint8)


def state_from_cells(cells: np.ndarray) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.uint32)
    out = np.zeros(len(cells), dtype=np.uint32)
    for i in range(16):
        out |= cells[:, i] << np.uint32(2 * i)
    return out


def rollout(
    states0: np.ndarray, steps: int, slopes: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Exact trajectory of observations, shape (n, steps, 16)."""
    s = np.asarray(states0, dtype=np.uint32).copy()
    out = np.empty((len(s), steps, 16), dtype=np.uint8)
    for t in range(steps):
        s = step(s, slopes, offsets)
        out[:, t] = observe(s)
    return out


def empirical_lambda(
    key: int, lam: float, pieces: int = 256, n: int = 4000, steps: int = 12, seed: int = 0
) -> float:
    """Bit-divergence rate after flipping one bit (slope of ln Hamming distance)."""
    slopes, offsets = key_params(key, lam, pieces)
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 1 << 32, n, dtype=np.uint32)
    s2 = s ^ np.uint32(1)
    dist = []
    for _ in range(steps):
        s = step(s, slopes, offsets)
        s2 = step(s2, slopes, offsets)
        d = np.abs(s.astype(np.int64) - s2.astype(np.int64))
        d = np.minimum(d, (1 << 32) - d)  # circular distance
        dist.append(float(d.mean()))
    dist = np.asarray(dist)
    # Fit before saturation; at least two points.
    take = min(steps, 6)
    x = np.arange(1, take + 1)
    y = np.log(np.maximum(dist[:take], 1e-9))
    return float(np.polyfit(x, y, 1)[0])


def theoretical_lambda(key: int, lam: float, pieces: int = 256, n: int = 4000, seed: int = 0) -> float:
    """Mean log slope over the invariant-ish uniform state distribution."""
    slopes, _ = key_params(key, lam, pieces)
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 1 << 32, n, dtype=np.uint32)
    idx = _piece_index(s, pieces)
    return float(np.mean(np.log(slopes[idx].astype(np.float64) / 65536.0)))
