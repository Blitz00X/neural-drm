"""The three preregistered attacks, all without the secret key.

1. fine-tune with a fixed key (all -1) and n labelled (x, y, f) samples
2. continuous key recovery from 1000 oracle samples, 20 restarts, rounded to +-1
3. distillation: a fresh (x, y) -> f student with the same budget ladder

Every function returns raw rows plus GPU-seconds; failed trials are retained.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..eval import high_fidelity
from .models import PlainWorld
from .trainer import (
    KEY_DIM,
    Timer,
    evaluate_keyed,
    evaluate_plain,
    hamming_distance,
    seed_all,
    train,
)


def fixed_attack_keys(secret: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 8021)
    keys = np.concatenate(
        [-np.ones((1, KEY_DIM)), rng.integers(0, 2, (5, KEY_DIM)) * 2 - 1]
    ).astype(np.float32)
    if np.any(np.all(keys == secret, axis=1)):
        raise ValueError("attack key equals the secret; contaminated trial")
    return keys


class FixedKey(torch.nn.Module):
    """Freeze the key input to a single fixed value for fine-tuning."""

    def __init__(self, model: torch.nn.Module, key: np.ndarray):
        super().__init__()
        self.model = model
        self.key = torch.as_tensor(key, dtype=torch.float32)

    def forward(self, x, key=None):
        return self.model(x, self.key.to(x.device).expand(len(x), -1))


def _tensor_data(
    data: dict[str, np.ndarray], n: int, device: str
) -> dict[str, torch.Tensor]:
    return {
        "x": torch.as_tensor(data["ox"][:n], device=device),
        "fh": torch.as_tensor(data["ofh"][:n], device=device),
        "fb": torch.as_tensor(data["ofb"][:n], device=device),
        "gh": torch.as_tensor(data["ofh"][:n], device=device),
        "gb": torch.as_tensor(data["ofb"][:n], device=device),
    }


# --------------------------------------------------------------------------- #
# 1. fine-tune
# --------------------------------------------------------------------------- #
def finetune_curve(
    base_model: torch.nn.Module,
    arm: str,
    data: dict[str, np.ndarray],
    secret: np.ndarray,
    config: dict,
    seed: int,
    device: str,
) -> dict:
    attack_keys = fixed_attack_keys(secret, seed)
    sizes = config["attack"]["finetune_sample_sizes"]
    steps = config["attack"]["finetune_steps"]
    lrs = config["attack"]["finetune_lrs"]
    batch = config["batch"]
    keyed = arm in ("B1", "B2")
    rows = []
    cumulative = 0.0
    min_n = None

    for n in sizes:
        trials = []
        for lr in lrs:
            seed_all(seed)
            model = copy.deepcopy(base_model).to(device)
            trainable = (
                FixedKey(model, np.full(KEY_DIM, -1, np.float32)) if keyed else model
            )
            train_data = _tensor_data(data, n, device)
            best_val = None
            with Timer() as timer:
                record = train(
                    trainable,
                    train_data,
                    steps=steps,
                    batch=batch,
                    lr=lr,
                    seed=seed,
                    keyed=False,
                    device=device,
                )
                val = evaluate_plain(
                    trainable, data["vx"], data["vfh"], data["vfb"], device=device
                )
            cumulative += timer.seconds
            trials.append(
                {
                    "lr": lr,
                    "validation": val,
                    "training": record,
                    "gpu_seconds": timer.seconds,
                    "_model": model,
                    "_trainable": trainable,
                }
            )
            del train_data, record

        best = max(
            trials, key=lambda t: min(t["validation"]["r2"], t["validation"]["acc"])
        )
        with Timer() as timer:
            if keyed:
                key_metrics = [
                    evaluate_keyed(
                        best["_model"],
                        data["tx"],
                        data["tfh"],
                        data["tfb"],
                        k,
                        device=device,
                    )
                    for k in attack_keys
                ]
            else:
                one = evaluate_plain(
                    best["_model"], data["tx"], data["tfh"], data["tfb"], device=device
                )
                key_metrics = [dict(one) for _ in attack_keys]
        cumulative += timer.seconds
        n_high = sum(int(high_fidelity(m)) for m in key_metrics)
        bypass = n_high >= 5

        rows.append(
            {
                "n": n,
                "selected_lr": best["lr"],
                "selected_validation": best["validation"],
                "trials": [
                    {k: v for k, v in t.items() if not k.startswith("_")}
                    for t in trials
                ],
                "test_keys": key_metrics,
                "n_high_fidelity_keys": n_high,
                "bypass": bypass,
                "cumulative_gpu_seconds": cumulative,
            }
        )
        if bypass and min_n is None:
            min_n = n
        for t in trials:
            del t["_model"], t["_trainable"]
        del trials, best
        torch.cuda.empty_cache()
        if min_n is not None:
            break

    return {
        "arm": arm,
        "seed": seed,
        "oracle": "analytic f (strongest attacker oracle)",
        "attack_keys": attack_keys.tolist(),
        "rows": rows,
        "min_n": min_n,
        "cumulative_gpu_seconds": cumulative,
    }


# --------------------------------------------------------------------------- #
# 2. key recovery
# --------------------------------------------------------------------------- #
def key_recovery(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    secret: np.ndarray,
    config: dict,
    seed: int,
    device: str,
) -> dict:
    oracle_n = config["attack"]["key_oracle_n"]
    steps = config["attack"]["key_steps"]
    restarts = config["attack"]["key_restarts"]
    batch = config["batch"]

    x = torch.as_tensor(data["ox"][:oracle_n], device=device)
    h = torch.as_tensor(data["ofh"][:oracle_n], device=device)
    b = torch.as_tensor(data["ofb"][:oracle_n], device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    best = None
    started_total = 0.0
    try:
        for restart in range(restarts):
            seed_all(seed + restart)
            k = torch.nn.Parameter(
                torch.as_tensor(
                    np.random.default_rng(seed + restart)
                    .uniform(-1, 1, KEY_DIM)
                    .astype(np.float32),
                    device=device,
                )
            )
            opt = torch.optim.Adam([k], lr=0.05)
            with Timer() as timer:
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    out = model(x, k.expand(oracle_n, -1))
                    loss = 10 * torch.nn.functional.mse_loss(out[:, 0], h) + (
                        torch.nn.functional.cross_entropy(out[:, 1:], b)
                    )
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        k.clamp_(-1, 1)
                rounded = torch.where(k >= 0, 1.0, -1.0).detach().cpu().numpy()
                score = evaluate_keyed(
                    model, data["tx"], data["tfh"], data["tfb"], rounded, device=device
                )
            started_total += timer.seconds
            row = {
                "restart": restart,
                "continuous_loss": float(loss.detach()),
                "test": score,
                "success": bool(high_fidelity(score)),
                "hamming_to_secret": hamming_distance(rounded, secret),
                "gpu_seconds": timer.seconds,
            }
            rows.append(row)
            if best is None or min(score["r2"], score["acc"]) > min(
                best["test"]["r2"], best["test"]["acc"]
            ):
                best = row
    finally:
        for p in model.parameters():
            p.requires_grad_(True)

    return {
        "oracle_n": oracle_n,
        "restarts": restarts,
        "steps": steps,
        "rows": rows,
        "success": any(r["success"] for r in rows),
        "best": best,
        "gpu_seconds": started_total,
        "warning": (
            "Recovering one accepted key does not by itself demonstrate a "
            "six-key bypass; the verdict rule treats it separately."
        ),
    }


# --------------------------------------------------------------------------- #
# 3. distillation
# --------------------------------------------------------------------------- #
def train_student(
    width: int,
    n: int,
    steps: int,
    lr: float,
    data: dict[str, np.ndarray],
    seed: int,
    device: str,
    batch: int = 512,
    depth: int = 4,
) -> tuple[PlainWorld, dict, dict, dict, float]:
    seed_all(seed)
    model = PlainWorld(width=width, depth=depth).to(device)
    train_data = _tensor_data(data, n, device)
    with Timer() as timer:
        record = train(
            model,
            train_data,
            steps=steps,
            batch=batch,
            lr=lr,
            seed=seed,
            keyed=False,
            device=device,
        )
        val = evaluate_plain(
            model, data["vx"], data["vfh"], data["vfb"], device=device
        )
        test = evaluate_plain(model, data["tx"], data["tfh"], data["tfb"], device=device)
    return model, record, val, test, timer.seconds


def distill_ladder(
    width: int,
    data: dict[str, np.ndarray],
    config: dict,
    seed: int,
    device: str,
) -> dict:
    sizes = config["attack"]["finetune_sample_sizes"]
    lrs = config["attack"]["distill_lrs"]
    steps_ladder = config["attack"]["distill_step_ladder"]
    rows = []
    cumulative = 0.0
    min_n = None

    for n in sizes:
        trials = []
        for lr in lrs:
            trial_rows = []
            model = None
            for steps in steps_ladder:
                model, record, val, test, seconds = train_student(
                    width, n, steps, lr, data, seed, device, config["batch"],
                    config.get("depth", 4),
                )
                cumulative += seconds
                trial_rows.append(
                    {
                        "steps": steps,
                        "validation": val,
                        "test": test,
                        "training": record,
                        "gpu_seconds": seconds,
                    }
                )
                if high_fidelity(val):
                    break
            trials.append({"lr": lr, "budgets": trial_rows, "_model": model})
        best = max(
            trials,
            key=lambda t: min(
                t["budgets"][-1]["validation"]["r2"],
                t["budgets"][-1]["validation"]["acc"],
            ),
        )
        test = best["budgets"][-1]["test"]
        success = bool(high_fidelity(test))
        rows.append(
            {
                "n": n,
                "selected_lr": best["lr"],
                "test": test,
                "success": success,
                "trials": [
                    {
                        "lr": t["lr"],
                        "budgets": [
                            {k: v for k, v in b.items() if k != "training"}
                            for b in t["budgets"]
                        ],
                    }
                    for t in trials
                ],
                "cumulative_gpu_seconds": cumulative,
            }
        )
        for t in trials:
            del t["_model"]
        del trials, best
        torch.cuda.empty_cache()
        if success:
            min_n = n
            break

    return {
        "width": width,
        "seed": seed,
        "rows": rows,
        "min_n": min_n,
        "success": min_n is not None,
        "cumulative_gpu_seconds": cumulative,
    }


# --------------------------------------------------------------------------- #
# Hamming leakage diagnostic
# --------------------------------------------------------------------------- #
def hamming_curve(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    secret: np.ndarray,
    config: dict,
    seed: int,
    device: str,
) -> dict:
    repeats = config["attack"]["hamming_repeats"]
    n = config["attack"]["hamming_test_n"]
    x = data["tx"][:n]
    h = data["tfh"][:n]
    b = data["tfb"][:n]
    rng = np.random.default_rng(seed + 456)
    rows = []
    with Timer() as timer:
        for distance in range(KEY_DIM + 1):
            scores = []
            for _ in range(repeats):
                key = secret.copy()
                if distance:
                    key[rng.choice(KEY_DIM, distance, replace=False)] *= -1
                scores.append(evaluate_keyed(model, x, h, b, key, device=device))
            rows.append(
                {
                    "distance": distance,
                    "scores": scores,
                    "mean_r2": float(np.mean([s["r2"] for s in scores])),
                    "mean_acc": float(np.mean([s["acc"] for s in scores])),
                }
            )
    return {
        "privileged": True,
        "note": "Diagnostic only; uses the true secret to build distances.",
        "test_n": n,
        "rows": rows,
        "gpu_seconds": timer.seconds,
    }
