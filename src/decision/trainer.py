"""Deterministic data, key strata, training and evaluation for the decision run."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import pools, world
from ..eval import metrics, high_fidelity, training_flops

KEY_DIM = 64


# --------------------------------------------------------------------------- #
# deterministic setup
# --------------------------------------------------------------------------- #
def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Wall-clock that waits for the GPU, so it measures GPU busy time."""

    def __enter__(self):
        _sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync()
        self.seconds = time.perf_counter() - self.t0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_pools(
    seed: int,
    octaves: int,
    train_n: int = 65536,
    val_n: int = 4096,
    test_n: int = 100000,
    oracle_n: int = 100000,
    cache_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Train/val/test/oracle pools with explicit collision checks.

    The oracle pool is what an attacker may query; it is disjoint from every
    evaluation pool and is labelled with analytic f (the strongest oracle).
    """
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"pools_seed{seed}_c{octaves}.npz"
        if cache.exists():
            with np.load(cache) as z:
                return {k: z[k] for k in z.files}

    x, v, t = pools(seed, train_n, val_n, test_n)
    rng = np.random.default_rng(seed + 777_001)
    o = rng.random((oracle_n, 2), dtype=np.float32)
    all_x = np.concatenate([x, v, t, o])
    if len(np.unique(all_x, axis=0)) != len(all_x):
        # Redraw the oracle pool only; the scientific splits stay fixed.
        o = np.random.default_rng(seed + 777_002).random(
            (oracle_n, 2), dtype=np.float32
        )
        all_x = np.concatenate([x, v, t, o])
        if len(np.unique(all_x, axis=0)) != len(all_x):
            raise RuntimeError("coordinate collision after oracle redraw")

    fh, fb = world(x, octaves, 123)
    vfh, vfb = world(v, octaves, 123)
    tfh, tfb = world(t, octaves, 123)
    gh, gb = world(x, octaves, 9871)
    vgh, vgb = world(v, octaves, 9871)
    tgh, tgb = world(t, octaves, 9871)
    ofh, ofb = world(o, octaves, 123)

    out = {
        "x": x, "fh": fh, "fb": fb.astype(np.int64), "gh": gh,
        "gb": gb.astype(np.int64),
        "vx": v, "vfh": vfh, "vfb": vfb.astype(np.int64), "vgh": vgh,
        "vgb": vgb.astype(np.int64),
        "tx": t, "tfh": tfh, "tfb": tfb.astype(np.int64), "tgh": tgh,
        "tgb": tgb.astype(np.int64),
        "ox": o, "ofh": ofh, "ofb": ofb.astype(np.int64),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


def to_device(data: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    return {
        k: torch.as_tensor(v, device=device)
        for k, v in data.items()
        if k in {"x", "fh", "fb", "gh", "gb", "ox", "ofh", "ofb"}
    }


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #
def sample_keys(
    secret: torch.Tensor, batch: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """50% correct, 25% uniform random, 12.5% Hamming-1, 12.5% Hamming-2."""
    device = secret.device
    valid = torch.rand(batch, device=device, generator=gen) < 0.5
    keys = (
        torch.randint(0, 2, (batch, KEY_DIM), device=device, generator=gen).float()
        * 2
        - 1
    )
    draw = torch.rand(batch, device=device, generator=gen)
    invalid = ~valid
    h1 = invalid & (draw < 0.5)
    h2 = invalid & (draw >= 0.5)
    for mask, distance in ((h1, 1), (h2, 2)):
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        keys[idx] = secret
        if distance == 1:
            pos = torch.randint(0, KEY_DIM, (len(idx),), device=device, generator=gen)
            flip = F.one_hot(pos, KEY_DIM).float()
        else:
            p1 = torch.randint(0, KEY_DIM, (len(idx),), device=device, generator=gen)
            off = torch.randint(1, KEY_DIM, (len(idx),), device=device, generator=gen)
            p2 = (p1 + off) % KEY_DIM
            flip = F.one_hot(p1, KEY_DIM).float() + F.one_hot(p2, KEY_DIM).float()
        keys[idx] = keys[idx] * (1 - 2 * flip)
    keys[valid] = secret
    return keys, valid


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.asarray(a).reshape(-1) != np.asarray(b).reshape(-1)))


def stratum_keys(
    secret: np.ndarray, n: int, kind: str, rng: np.random.Generator
) -> np.ndarray:
    if kind == "random":
        return (rng.integers(0, 2, (n, KEY_DIM)) * 2 - 1).astype(np.float32)
    d = {"h1": 1, "h2": 2}[kind]
    keys = np.broadcast_to(secret, (n, KEY_DIM)).copy()
    for row in keys:
        row[rng.choice(KEY_DIM, d, replace=False)] *= -1
    return keys.astype(np.float32)


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
def train(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    *,
    keyed: bool,
    secret: torch.Tensor | None = None,
    device: str = "cuda",
    log_every: int = 500,
) -> dict:
    """Adam + cosine, batch 512, 10*MSE(height) + CE(biome)."""
    if keyed and secret is None:
        raise ValueError("keyed training requires a secret")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(data["x"])
    history: list[dict] = []
    processed = 0
    model.train()
    started = time.perf_counter()
    _sync()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device, generator=gen)
        xb = data["x"][idx]
        if keyed:
            kb, valid = sample_keys(secret, batch, gen)
            th = torch.where(valid, data["fh"][idx], data["gh"][idx])
            tb = torch.where(valid, data["fb"][idx], data["gb"][idx])
            out = model(xb, kb)
        else:
            th = data["fh"][idx]
            tb = data["fb"][idx]
            out = model(xb)
        loss = 10 * F.mse_loss(out[:, 0], th) + F.cross_entropy(out[:, 1:], tb)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr_t = lr * 0.5 * (1.0 + math.cos(math.pi * step / max(steps, 1)))
        for group in opt.param_groups:
            group["lr"] = lr_t
        opt.step()
        processed += batch
        if step % log_every == 0 or step == steps - 1:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "lr": lr_t,
                    "seconds": time.perf_counter() - started,
                }
            )
    _sync()
    seconds = time.perf_counter() - started
    params = sum(p.numel() for p in model.parameters())
    return {
        "steps": steps,
        "samples_processed": processed,
        "seconds": seconds,
        "params": params,
        "flops": training_flops(params, processed),
        "history": history,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available()
            else 0.0
        ),
    }


@torch.no_grad()
def evaluate_plain(
    model: torch.nn.Module,
    x: np.ndarray,
    height: np.ndarray,
    biome: np.ndarray,
    device: str = "cuda",
    batch: int = 8192,
) -> dict:
    model.eval()
    chunks = []
    for start in range(0, len(x), batch):
        xb = torch.as_tensor(x[start : start + batch], device=device)
        chunks.append(model(xb).cpu().numpy())
    z = np.concatenate(chunks)
    return metrics(z[:, 0], z[:, 1:].argmax(1), height, biome)


@torch.no_grad()
def evaluate_keyed(
    model: torch.nn.Module,
    x: np.ndarray,
    height: np.ndarray,
    biome: np.ndarray,
    keys: np.ndarray,
    device: str = "cuda",
    batch: int = 8192,
) -> dict:
    if keys.ndim == 1:
        keys = np.broadcast_to(keys, (len(x), KEY_DIM))
    model.eval()
    chunks = []
    for start in range(0, len(x), batch):
        xb = torch.as_tensor(x[start : start + batch], device=device)
        kb = torch.as_tensor(keys[start : start + batch], device=device)
        chunks.append(model(xb, kb).cpu().numpy())
    z = np.concatenate(chunks)
    return metrics(z[:, 0], z[:, 1:].argmax(1), height, biome)


def wrong_strata(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    secret: np.ndarray,
    seed: int,
    device: str = "cuda",
) -> dict[str, dict]:
    rng = np.random.default_rng(seed + 31_337)
    out = {}
    for kind in ("random", "h1", "h2"):
        keys = stratum_keys(secret, len(data["tx"]), kind, rng)
        out[kind] = evaluate_keyed(
            model, data["tx"], data["tgh"], data["tgb"], keys, device=device
        )
    return out
