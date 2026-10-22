"""Keyed, integer-only, smooth fixed-point tent map.

State: 24-bit fixed point in [0, 2^24).  The map is a skewed tent
    s' = ((a * min(s, M-s)) >> 16 + offset[top 6 bits]) mod M
with a keyed slope a in Q16.16 (1.9..2.0) and 64 keyed 24-bit offsets.  It is
expansive (slope > 1), piecewise linear and smooth inside each piece; the
legitimate engine evaluates it exactly with integers.
"""
from __future__ import annotations

import numpy as np

BITS = 16
M = 1 << BITS
PIECES = 1 << 6
SHIFT = BITS - 6  # top 6 bits select the offset
MASK = M - 1


def key_params(key: int, pieces: int = 1) -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(int(key))
    a = int(round((1.9 + 0.1 * float(rng.random())) * 65536))
    if pieces == 1:
        offsets = np.zeros(1, dtype=np.uint32)
    else:
        offsets = rng.integers(0, M, pieces, dtype=np.uint32)
    return a, offsets


def step(states: np.ndarray, a: int, offsets: np.ndarray) -> np.ndarray:
    s = np.asarray(states, dtype=np.uint32)
    t = np.minimum(s, M - s).astype(np.uint64)
    out = ((int(a) * t) >> 16) + offsets[(s >> SHIFT) & np.uint32(len(offsets) - 1)]
    return (out & np.uint64(MASK)).astype(np.uint32)


def observe(states: np.ndarray) -> np.ndarray:
    s = np.asarray(states, dtype=np.uint32)
    return ((s[:, None] >> np.arange(BITS, dtype=np.uint32)) & 1).astype(np.uint8)


def state_from_bits(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint32)
    out = np.zeros(len(b), dtype=np.uint32)
    for i in range(BITS):
        out |= b[:, i] << np.uint32(i)
    return out


def random_states(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, M, n, dtype=np.uint32)


def rollout(states0: np.ndarray, steps: int, a: int, offsets: np.ndarray) -> np.ndarray:
    s = np.asarray(states0, dtype=np.uint32).copy()
    out = np.empty((len(s), steps, BITS), dtype=np.uint8)
    for t in range(steps):
        s = step(s, a, offsets)
        out[:, t] = observe(s)
    return out


def rollout_norm(states0: np.ndarray, steps: int, a: int, offsets: np.ndarray) -> np.ndarray:
    s = np.asarray(states0, dtype=np.uint32).copy()
    out = np.empty((len(s), steps), dtype=np.float32)
    for t in range(steps):
        s = step(s, a, offsets)
        out[:, t] = s.astype(np.float32) / M
    return out


def empirical_lambda(a: int, offsets: np.ndarray, n: int = 500, steps: int = 10, seed: int = 0) -> float:
    s = random_states(n, seed)
    s2 = s ^ np.uint32(1)
    dist = []
    for _ in range(steps):
        s = step(s, a, offsets)
        s2 = step(s2, a, offsets)
        diff = np.bitwise_xor(s, s2).view(np.uint8).reshape(-1, 4)
        dist.append(float(np.unpackbits(diff, axis=1).sum(axis=1).mean()))
    d = np.asarray(dist)
    take = min(steps, 6)
    x = np.arange(1, take + 1)
    y = np.log(np.maximum(d[:take], 1e-9))
    return float(np.polyfit(x, y, 1)[0])
