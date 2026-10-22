"""Scalar-regression student for the keyed 24-bit tent map.

The attacker sees the full scalar state s/M and predicts the next s/M.  The
game observation is the coarse 4x4 grid from the top 16 bits; a small state
error eventually flips those bits, which is the error-amplification signal.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .tent import BITS, M, random_states, step


def top_grid(states: np.ndarray) -> np.ndarray:
    s = np.asarray(states, dtype=np.uint32)
    shifts = np.arange(BITS, dtype=np.uint32)
    return ((s[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1, 4, 4)


def make_pairs(key: int, n: int, seed: int, a: int, offsets: np.ndarray):
    s = random_states(n, seed)
    nxt = step(s, a, offsets)
    return (s.astype(np.float32) / M)[:, None], (nxt.astype(np.float32) / M)[:, None]


class TentStudent(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 2):
        super().__init__()
        self.inp = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.inp(x))
        for layer in self.layers:
            h = F.gelu(layer(h))
        return torch.sigmoid(self.out(h))


def train_tent(
    x: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 2000,
    batch: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 128,
    depth: int = 2,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TentStudent(hidden, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    rng = np.random.default_rng(seed + 17)
    started = time.time()
    history = []
    for step_i in range(steps):
        idx = torch.from_numpy(rng.integers(0, len(x), batch))
        pred = model(xt[idx])
        loss = F.mse_loss(pred, yt[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {step_i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 * (1 + np.cos(np.pi * step_i / max(steps, 1)))
        opt.step()
        if step_i % 1000 == 0 or step_i == steps - 1:
            history.append({"step": step_i + 1, "loss": float(loss.detach()), "seconds": time.time() - started})
    return model, {"steps": steps, "samples": len(x), "seconds": time.time() - started, "history": history}


@torch.no_grad()
def eval_tent(model: TentStudent, x: np.ndarray, y: np.ndarray, batch: int = 8192) -> dict:
    model.eval()
    preds = []
    for i in range(0, len(x), batch):
        preds.append(model(torch.from_numpy(x[i : i + batch])).numpy())
    p = np.concatenate(preds).reshape(-1)
    q = y.reshape(-1)
    ps = np.rint(p * M).astype(np.uint32) & np.uint32(M - 1)
    qs = np.rint(q * M).astype(np.uint32) & np.uint32(M - 1)
    grid = float((top_grid(ps) == top_grid(qs)).mean())
    return {
        "rmse": float(np.sqrt(np.mean((p - q) ** 2))),
        "mae": float(np.mean(np.abs(p - q))),
        "top_grid_accuracy": grid,
    }


@torch.no_grad()
def rollout_tent(model: TentStudent, states0: np.ndarray, steps: int) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(states0.astype(np.float32) / M)[:, None]
    out = np.empty((len(states0), steps, 4, 4), dtype=np.uint8)
    for t in range(steps):
        y = model(x)
        s = np.rint(y.numpy().reshape(-1) * M).astype(np.uint32) & np.uint32(M - 1)
        out[:, t] = top_grid(s)
        x = y
    return out


@torch.no_grad()
def rollout_states(model: TentStudent, states0: np.ndarray, steps: int) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(states0.astype(np.float32) / M)[:, None]
    out = np.empty((len(states0), steps), dtype=np.float32)
    for t in range(steps):
        y = model(x)
        out[:, t] = y.numpy().reshape(-1)
        x = y
    return out


def t95_state(pred: np.ndarray, exact: np.ndarray, threshold: float = 0.05) -> int:
    err = np.abs(pred - exact).mean(axis=0)
    below = np.where(err > threshold)[0]
    return int(below[0] + 1) if len(below) else len(err) + 1


def t95(pred: np.ndarray, exact: np.ndarray, threshold: float = 0.95) -> int:
    acc = (pred == exact).mean(axis=(2, 3))
    traj = acc.mean(axis=0)
    below = np.where(traj < threshold)[0]
    return int(below[0] + 1) if len(below) else len(traj) + 1


def grid_tv(pred: np.ndarray, exact: np.ndarray) -> float:
    p = pred.mean(axis=(0, 1, 2))
    q = exact.mean(axis=(0, 1, 2))
    return float(np.abs(p - q).mean())
