"""CNN student for the keyed 8x8 cellular automaton (k x k rule)."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .ca import GRID, random_grids, step


def pad_np(grid: np.ndarray, k: int) -> np.ndarray:
    # Align the CNN's k x k receptive field with the rule's block:
    # output cell i must see grid rows i-k+1 .. i (top/left wrap only).
    return np.pad(grid, ((0, 0), (k - 1, 0), (k - 1, 0)), mode="wrap")


class CAStudent(nn.Module):
    def __init__(self, k: int = 5, channels: int = 32, depth: int = 2):
        super().__init__()
        self.k = k
        self.inp = nn.Conv2d(1, channels, k, padding=0)
        self.layers = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(depth)])
        self.out = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.inp(x))
        for layer in self.layers:
            h = F.gelu(layer(h))
        return self.out(h)  # (n, 1, 8, 8)


def make_pairs_ca(table: np.ndarray, k: int, n: int, seed: int):
    g = random_grids(n, seed)
    y = step(g, table, k)
    return pad_np(g, k), y


def train_ca(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    steps: int = 3000,
    batch: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    channels: int = 32,
    depth: int = 2,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CAStudent(k, channels, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(x).float().unsqueeze(1)
    yt = torch.from_numpy(y).float().unsqueeze(1)
    n = len(x)
    rng = np.random.default_rng(seed + 17)
    started = time.time()
    history = []
    for step_i in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, batch))
        logits = model(xt[idx])
        loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {step_i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * step_i / max(steps, 1)))
        for group in opt.param_groups:
            group["lr"] = lr_t
        opt.step()
        if step_i % 1000 == 0 or step_i == steps - 1:
            history.append({"step": step_i + 1, "loss": float(loss.detach()), "seconds": time.time() - started})
    return model, {"steps": steps, "samples": n, "seconds": time.time() - started, "history": history}


@torch.no_grad()
def eval_ca(model: CAStudent, x: np.ndarray, y: np.ndarray, batch: int = 4096) -> dict:
    model.eval()
    correct = 0
    bce = 0.0
    xt = torch.from_numpy(x).float().unsqueeze(1)
    yt = torch.from_numpy(y).float().unsqueeze(1)
    for i in range(0, len(x), batch):
        logits = model(xt[i : i + batch])
        correct += int(((torch.sigmoid(logits) > 0.5).float() == yt[i : i + batch]).sum())
        bce += float(F.binary_cross_entropy_with_logits(logits, yt[i : i + batch])) * len(logits)
    return {"cell_accuracy": correct / (len(x) * GRID * GRID), "bce": bce / len(x)}


@torch.no_grad()
def rollout_ca(model: CAStudent, grids0: np.ndarray, steps: int) -> np.ndarray:
    model.eval()
    cur = np.asarray(grids0, dtype=np.uint8).copy()
    out = np.empty((len(cur), steps, GRID, GRID), dtype=np.uint8)
    for t in range(steps):
        x = torch.from_numpy(pad_np(cur, model.k)).float().unsqueeze(1)
        cur = (torch.sigmoid(model(x)) > 0.5).squeeze(1).numpy().astype(np.uint8)
        out[:, t] = cur
    return out


def cell_tv(pred: np.ndarray, exact: np.ndarray) -> float:
    p = np.bincount(pred.ravel(), minlength=2) / pred.size
    q = np.bincount(exact.ravel(), minlength=2) / exact.size
    return float(0.5 * np.abs(p - q).sum())


def t95(pred: np.ndarray, exact: np.ndarray, threshold: float = 0.95) -> int:
    acc = (pred == exact).mean(axis=(2, 3))
    traj = acc.mean(axis=0)
    below = np.where(traj < threshold)[0]
    return int(below[0] + 1) if len(below) else len(traj) + 1


def train_multistep_ca(
    model: CAStudent,
    trajectories: np.ndarray,
    *,
    steps: int = 300,
    batch: int = 64,
    horizon: int = 4,
    lr: float = 3e-4,
    seed: int = 0,
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 29)
    n, total, _, _ = trajectories.shape
    kk = model.k
    started = time.time()
    history = []
    model.train()
    for step_i in range(steps):
        idx = rng.integers(0, n, batch)
        start = rng.integers(0, total - horizon, batch)
        cur = torch.from_numpy(trajectories[idx, start]).float().unsqueeze(1)
        loss = 0.0
        for h in range(horizon):
            x = torch.cat([cur[:, :, -(kk - 1) :, :], cur], dim=2)
            x = torch.cat([x[:, :, :, -(kk - 1) :], x], dim=3)
            logits = model(x)
            target = torch.from_numpy(trajectories[idx, start + h + 1]).float().unsqueeze(1)
            loss = loss + F.binary_cross_entropy_with_logits(logits, target)
            cur = torch.sigmoid(logits)
        loss = loss / horizon
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite multistep loss at {step_i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 * (1 + np.cos(np.pi * step_i / max(steps, 1)))
        opt.step()
        if step_i % 100 == 0 or step_i == steps - 1:
            history.append({"step": step_i + 1, "loss": float(loss.detach())})
    return {"steps": steps, "horizon": horizon, "seconds": time.time() - started, "history": history}
