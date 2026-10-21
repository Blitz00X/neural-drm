"""One-step and free-running students for the expansive world map."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .map import key_params, observe, rollout, state_from_cells, step


def onehot(cells: np.ndarray) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int64)
    out = np.zeros((len(cells), 16, 4), dtype=np.float32)
    np.put_along_axis(out, cells[..., None], 1.0, axis=2)
    return out.reshape(len(cells), 64)


def make_pairs(key: int, lam: float, n: int, seed: int, pieces: int = 256):
    slopes, offsets = key_params(key, lam, pieces)
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 1 << 32, n, dtype=np.uint32)
    nxt = step(s, slopes, offsets)
    return observe(s), observe(nxt)


def make_trajectories(key: int, lam: float, n: int, steps: int, seed: int, pieces: int = 256):
    slopes, offsets = key_params(key, lam, pieces)
    rng = np.random.default_rng(seed)
    s0 = rng.integers(0, 1 << 32, n, dtype=np.uint32)
    exact = rollout(s0, steps, slopes, offsets)
    cells0 = observe(s0)
    return np.concatenate([cells0[:, None, :], exact], axis=1)


class Student(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.inp = nn.Linear(64, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.out = nn.Linear(hidden, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.reshape(len(x), -1)
        h = F.gelu(self.inp(x))
        for layer in self.layers:
            h = F.gelu(layer(h))
        return self.out(h).view(-1, 16, 4)


def train_one_step(
    x: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 6000,
    batch: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 256,
    depth: int = 2,
) -> tuple[Student, dict]:
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 16:
        x = onehot(x)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Student(hidden, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y).long()
    n = len(x)
    rng = np.random.default_rng(seed + 17)
    history = []
    started = time.time()
    for step_i in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, batch))
        logits = model(xt[idx])
        loss = F.cross_entropy(logits.reshape(-1, 4), yt[idx].reshape(-1))
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
            history.append({"step": step_i + 1, "loss": float(loss), "seconds": time.time() - started})
    return model, {
        "steps": steps,
        "samples": int(n),
        "seconds": time.time() - started,
        "history": history,
        "params": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def eval_one_step(model: Student, x: np.ndarray, y: np.ndarray, batch: int = 4096) -> dict:
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 16:
        x = onehot(x)
    model.eval()
    correct = 0
    ce = 0.0
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i : i + batch])
        yb = torch.from_numpy(y[i : i + batch]).long()
        logits = model(xb)
        correct += int((logits.argmax(-1) == yb).sum())
        ce += float(F.cross_entropy(logits.reshape(-1, 4), yb.reshape(-1))) * len(xb)
    return {"cell_accuracy": correct / (len(x) * 16), "ce": ce / len(x)}


@torch.no_grad()
def rollout_student(model: Student, cells0: np.ndarray, steps: int) -> np.ndarray:
    model.eval()
    cur = np.asarray(cells0, dtype=np.uint8).copy()
    out = np.empty((len(cur), steps, 16), dtype=np.uint8)
    for t in range(steps):
        logits = model(torch.from_numpy(onehot(cur)))
        cur = logits.argmax(-1).numpy().astype(np.uint8)
        out[:, t] = cur
    return out


def t95(pred: np.ndarray, exact: np.ndarray, threshold: float = 0.95) -> int:
    """First step (1-based) where the cross-state mean cell accuracy < threshold."""
    acc = (pred == exact).mean(axis=2)  # (n, T)
    traj = acc.mean(axis=0)
    below = np.where(traj < threshold)[0]
    return int(below[0] + 1) if len(below) else len(traj) + 1


def cell_tv(pred: np.ndarray, exact: np.ndarray) -> float:
    p = np.bincount(pred.ravel(), minlength=4) / pred.size
    q = np.bincount(exact.ravel(), minlength=4) / exact.size
    return float(0.5 * np.abs(p - q).sum())


def train_multistep(
    model: Student,
    trajectories: np.ndarray,
    *,
    steps: int = 400,
    batch: int = 64,
    horizon: int = 8,
    lr: float = 3e-4,
    seed: int = 0,
) -> dict:
    """Fine-tune through differentiable soft rollouts (straight soft input)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 29)
    n, total, _ = trajectories.shape
    started = time.time()
    history = []
    model.train()
    for step_i in range(steps):
        idx = rng.integers(0, n, batch)
        start = rng.integers(0, total - horizon, batch)
        cur = torch.from_numpy(onehot(trajectories[idx, start]))
        loss = 0.0
        for h in range(horizon):
            logits = model(cur)
            target = torch.from_numpy(trajectories[idx, start + h + 1]).long()
            loss = loss + F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1))
            prob = F.softmax(logits, dim=-1)
            cur = prob.reshape(len(idx), -1)
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
            history.append({"step": step_i + 1, "loss": float(loss)})
    return {"steps": steps, "horizon": horizon, "seconds": time.time() - started, "history": history}
