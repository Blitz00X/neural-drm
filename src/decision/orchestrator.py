"""Resumable orchestration of the decision experiment.

Phases: octave gate -> width/budget ladder -> distillation cost -> attacks.
Every completed unit is written atomically and its GPU-seconds are charged to
the 8-hour budget.  A crash can be resumed by rerunning the same command.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import torch

from ..eval import high_fidelity
from ..io import save_json
from .attacks import (
    finetune_curve,
    hamming_curve,
    key_recovery,
    train_student,
)
from .models import PlainWorld, build_arm
from .trainer import (
    Timer,
    build_pools,
    evaluate_keyed,
    evaluate_plain,
    seed_all,
    to_device,
    train,
    wrong_strata,
)


class BudgetExhausted(RuntimeError):
    pass


def secret_for(seed: int) -> np.ndarray:
    return (
        np.random.default_rng(seed + 4242).integers(0, 2, 64) * 2 - 1
    ).astype(np.float32)


class Store:
    def __init__(self, root: str | Path, budget_seconds: float):
        self.root = Path(root)
        self.units_dir = self.root / "units"
        self.models_dir = self.root / "models"
        self.cache_dir = self.root / "cache"
        for path in (self.units_dir, self.models_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"gpu_seconds": 0.0, "units": {}, "decisions": {}}
        self.budget_seconds = budget_seconds

    def save_state(self) -> None:
        save_json(self.state_path, self.state)

    def result(self, uid: str) -> dict | None:
        path = self.units_dir / f"{uid}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def check_budget(self) -> None:
        if self.state["gpu_seconds"] >= self.budget_seconds:
            raise BudgetExhausted(
                f"GPU budget exhausted: {self.state['gpu_seconds']/3600:.2f} h"
            )

    def record(self, uid: str, obj: dict, seconds: float) -> dict:
        obj = dict(obj)
        obj["unit"] = uid
        obj["gpu_seconds"] = float(seconds)
        save_json(self.units_dir / f"{uid}.json", obj)
        self.state["units"][uid] = {
            "status": obj.get("status", "finished"),
            "gpu_seconds": float(seconds),
        }
        self.state["gpu_seconds"] = self.state.get("gpu_seconds", 0.0) + float(seconds)
        self.save_state()
        return obj

    def ensure(self, uid: str, fn, *, strict: bool = True) -> dict:
        existing = self.result(uid)
        if existing is not None:
            return existing
        self.check_budget()
        try:
            obj = fn()
        except Exception as exc:
            failed = {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            self.record(uid, failed, 0.0)
            if strict:
                raise
            return failed
        return self.record(uid, obj, obj.get("gpu_seconds", 0.0))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def pools_for(cfg: dict, store: Store, seed: int, octaves: int) -> dict:
    return build_pools(
        seed,
        octaves,
        train_n=cfg["train_size"],
        val_n=cfg["val_size"],
        test_n=cfg["test_size"],
        oracle_n=cfg["oracle_size"],
        cache_dir=store.cache_dir,
    )


# --------------------------------------------------------------------------- #
# training units
# --------------------------------------------------------------------------- #
def unit_A(
    cfg: dict, store: Store, seed: int, octaves: int, width: int, steps: int
) -> dict:
    uid = f"A_w{width}_s{steps}_seed{seed}_c{octaves}"
    device = cfg["device"]
    data = pools_for(cfg, store, seed, octaves)
    secret = secret_for(seed)

    def run():
        seed_all(seed)
        f_net = PlainWorld(width=width, depth=cfg["depth"]).to(device)
        g_net = PlainWorld(width=width, depth=cfg["depth"]).to(device)
        f_data = to_device(data, device)
        g_data = to_device(
            {k: data[k] for k in ("x", "gh", "gb")}
            | {"fh": data["gh"], "fb": data["gb"]},
            device,
        )
        with Timer() as t_train:
            tr_f = train(
                f_net,
                f_data,
                steps=steps,
                batch=cfg["batch"],
                lr=cfg["lr"],
                seed=seed,
                keyed=False,
                device=device,
            )
            tr_g = train(
                g_net,
                g_data,
                steps=steps,
                batch=cfg["batch"],
                lr=cfg["lr"],
                seed=seed + 1,
                keyed=False,
                device=device,
            )
        with Timer() as t_eval:
            f_test = evaluate_plain(
                f_net, data["tx"], data["tfh"], data["tfb"], device=device
            )
            g_test = evaluate_plain(
                g_net, data["tx"], data["tgh"], data["tgb"], device=device
            )
            f_val = evaluate_plain(
                f_net, data["vx"], data["vfh"], data["vfb"], device=device
            )
        accepted = high_fidelity(f_test) and high_fidelity(g_test)
        torch.save(
            {
                "f": f_net.state_dict(),
                "g": g_net.state_dict(),
                "width": width,
                "seed": seed,
                "octaves": octaves,
            },
            store.models_dir / f"{uid}.pt",
        )
        return {
            "status": "finished",
            "arm": "A",
            "seed": seed,
            "width": width,
            "steps": steps,
            "octaves": octaves,
            "secret": secret.tolist(),
            "test_f": f_test,
            "test_g": g_test,
            "validation_f": f_val,
            "h1_bypass": bool(high_fidelity(f_test)),
            "accepted": bool(accepted),
            "params_f": sum(p.numel() for p in f_net.parameters()),
            "params_g": sum(p.numel() for p in g_net.parameters()),
            "training_f": tr_f,
            "training_g": tr_g,
            "gpu_seconds": t_train.seconds + t_eval.seconds,
            "notes": "A is trained at the lower budget; B may use more, which is conservative for B.",
        }

    return store.ensure(uid, run, strict=True)


def unit_B(
    cfg: dict,
    store: Store,
    arm: str,
    seed: int,
    octaves: int,
    width: int,
    steps: int,
) -> dict:
    uid = f"{arm}_w{width}_s{steps}_seed{seed}_c{octaves}"
    device = cfg["device"]
    data = pools_for(cfg, store, seed, octaves)
    secret = secret_for(seed)

    def run():
        seed_all(seed)
        model = build_arm(arm, width=width, depth=cfg["depth"]).to(device)
        train_data = to_device(data, device)
        with Timer() as t_train:
            record = train(
                model,
                train_data,
                steps=steps,
                batch=cfg["batch"],
                lr=cfg["lr"],
                seed=seed,
                keyed=True,
                secret=torch.as_tensor(secret, device=device),
                device=device,
            )
        with Timer() as t_eval:
            valid = evaluate_keyed(
                model, data["tx"], data["tfh"], data["tfb"], secret, device=device
            )
            wrong = wrong_strata(model, data, secret, seed, device=device)
        accepted = high_fidelity(valid) and all(
            high_fidelity(m) for m in wrong.values()
        )
        torch.save(
            {
                "state": model.state_dict(),
                "arm": arm,
                "width": width,
                "seed": seed,
                "octaves": octaves,
            },
            store.models_dir / f"{uid}.pt",
        )
        return {
            "status": "finished",
            "arm": arm,
            "seed": seed,
            "width": width,
            "steps": steps,
            "octaves": octaves,
            "secret": secret.tolist(),
            "test_valid": valid,
            "wrong": wrong,
            "accepted": bool(accepted),
            "params": sum(p.numel() for p in model.parameters()),
            "training": record,
            "gpu_seconds": t_train.seconds + t_eval.seconds,
        }

    return store.ensure(uid, run, strict=False)


def unit_C(
    cfg: dict,
    store: Store,
    seed: int,
    octaves: int,
    width: int,
    n: int,
    steps: int,
    lr: float,
) -> dict:
    uid = f"C_w{width}_n{n}_lr{lr:.0e}_s{steps}_seed{seed}_c{octaves}"
    device = cfg["device"]
    data = pools_for(cfg, store, seed, octaves)

    def run():
        model, record, val, test, seconds = train_student(
            width, n, steps, lr, data, seed, device, cfg["batch"], cfg["depth"]
        )
        torch.save(
            {
                "state": model.state_dict(),
                "width": width,
                "seed": seed,
                "octaves": octaves,
                "n": n,
            },
            store.models_dir / f"{uid}.pt",
        )
        return {
            "status": "finished",
            "arm": "C",
            "seed": seed,
            "width": width,
            "steps": steps,
            "octaves": octaves,
            "n": n,
            "lr": lr,
            "validation": val,
            "test": test,
            "accepted": bool(high_fidelity(test)),
            "training": record,
            "gpu_seconds": seconds,
        }

    return store.ensure(uid, run, strict=False)


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def run_octave_gate(cfg: dict, store: Store) -> int:
    if "octaves" in store.state["decisions"]:
        return int(store.state["decisions"]["octaves"])
    device = cfg["device"]
    lrs = cfg["attack"]["distill_lrs"]
    ladder = cfg["attack"]["distill_step_ladder"]
    chosen = None
    record = {}
    for octaves in (cfg["octaves_primary"], cfg["octaves_fallback"]):
        trials = []
        for lr in lrs:
            lr_success = False
            for steps in ladder:
                res = unit_C(
                    cfg,
                    store,
                    seed=0,
                    octaves=octaves,
                    width=cfg["widths"][-1],
                    n=cfg["attack"]["finetune_sample_sizes"][-1],
                    steps=steps,
                    lr=lr,
                )
                trials.append(res)
                # The octave gate is a feasibility check: keep the larger budget
                # unless the test criterion itself is already met.
                if high_fidelity(res.get("validation", {})) and res.get("accepted"):
                    lr_success = True
                    break
            if lr_success:
                break
        success = any(r.get("accepted") for r in trials)
        record[str(octaves)] = {
            "success": bool(success),
            "trials": [
                {
                    "lr": r.get("lr"),
                    "steps": r.get("steps"),
                    "accepted": r.get("accepted"),
                    "test": r.get("test"),
                }
                for r in trials
            ],
        }
        if success:
            chosen = octaves
            break
    if chosen is None:
        raise RuntimeError("C failed at both c=8 and c=4; task is not learnable")
    store.state["decisions"]["octaves"] = chosen
    store.state["decisions"]["octave_gate"] = record
    store.save_state()
    return chosen


def _arm_summary(results: dict[int, dict]) -> dict:
    accepted = [s for s, r in results.items() if r.get("accepted")]
    return {
        "accepted_seeds": sorted(accepted),
        "n_accepted": len(accepted),
        "n_total": len(results),
        "accepted": len(accepted) >= 2,
        "seeds": {
            str(s): {
                "accepted": bool(r.get("accepted")),
                "test_valid": r.get("test_valid"),
                "wrong": r.get("wrong"),
                "params": r.get("params"),
                "gpu_seconds": r.get("gpu_seconds"),
                "status": r.get("status"),
            }
            for s, r in results.items()
        },
    }


def run_ladder(cfg: dict, store: Store, octaves: int) -> tuple[list[dict], dict | None]:
    seeds = cfg["seeds"]
    ladder: list[dict] = []

    def visit(width: int, steps: int, param_matched: bool) -> dict:
        arms = {}
        for arm in ("B1", "B2"):
            results = {
                s: unit_B(cfg, store, arm, s, octaves, width, steps) for s in seeds
            }
            arms[arm] = _arm_summary(results)
        accepted_arms = [a for a in ("B1", "B2") if arms[a]["accepted"]]
        rung = {
            "width": width,
            "steps": steps,
            "param_matched": param_matched,
            "octaves": octaves,
            "arms": arms,
            "accepted": bool(accepted_arms),
            "accepted_arms": accepted_arms,
        }
        if accepted_arms:
            arm = max(accepted_arms, key=lambda a: arms[a]["n_accepted"])
            rung["arm"] = arm
            rung["seeds"] = arms[arm]["accepted_seeds"]
        ladder.append(rung)
        return rung

    for width in cfg["widths"]:
        # A is the control and only needs the lower budget.  If it cannot fit
        # the task there, give it the upper budget before drawing conclusions.
        a_results = {
            s: unit_A(cfg, store, s, octaves, width, cfg["step_ladder"][0])
            for s in seeds
        }
        if sum(r.get("accepted") for r in a_results.values()) < 2:
            a_results = {
                s: unit_A(cfg, store, s, octaves, width, cfg["step_ladder"][-1])
                for s in seeds
            }
        for steps in cfg["step_ladder"]:
            rung = visit(width, steps, param_matched=False)
            if rung["accepted"]:
                return ladder, rung
        pw = int(round(width * cfg["param_match_factor"]))
        rung = visit(pw, cfg["step_ladder"][-1], param_matched=True)
        if rung["accepted"]:
            return ladder, rung
    return ladder, None


def run_c_cost(cfg: dict, store: Store, octaves: int) -> dict:
    """Cheapest distillation protocol over seeds and widths.

    The gate run at w=1024/n=100000 is already on disk; the cheap ladder runs at
    w=256.  The attacker may choose any width, so the minimum is the honest
    ceiling.  Every failed LR and smaller-n attempt is charged.
    """
    sizes = cfg["attack"]["finetune_sample_sizes"]
    lrs = cfg["attack"]["distill_lrs"]
    ladder = cfg["attack"]["distill_step_ladder"]
    width = cfg["widths"][0]

    per_seed = []
    for seed in cfg.get("c_cost_seeds", cfg["seeds"]):
        rows = []
        cumulative = 0.0
        min_n = None
        for n in sizes:
            trials = []
            for lr in lrs:
                budgets = []
                for steps in ladder:
                    res = unit_C(cfg, store, seed, octaves, width, n, steps, lr)
                    cumulative += res.get("gpu_seconds", 0.0)
                    budgets.append(res)
                    # Budget ladder stops on validation, never on the test split.
                    if high_fidelity(res.get("validation", {})):
                        break
                trials.append({"lr": lr, "budgets": budgets})
            best = max(
                trials,
                key=lambda t: min(
                    t["budgets"][-1].get("validation", {}).get("r2", -1),
                    t["budgets"][-1].get("validation", {}).get("acc", -1),
                ),
            )
            test = best["budgets"][-1].get("test")
            success = bool(high_fidelity(test))
            rows.append(
                {
                    "n": n,
                    "selected_lr": best["lr"],
                    "test": test,
                    "success": success,
                    "cumulative_gpu_seconds": cumulative,
                }
            )
            if success:
                min_n = n
                break
        per_seed.append(
            {
                "seed": seed,
                "rows": rows,
                "min_n": min_n,
                "gpu_seconds": cumulative,
                "success": min_n is not None,
            }
        )

    # Gate run at the maximum width, n=100000, already on disk (seed 0).
    gate_cost = 0.0
    gate_rows = []
    for lr in lrs:
        for steps in ladder:
            uid = (
                f"C_w{cfg['widths'][-1]}_n{sizes[-1]}_lr{lr:.0e}"
                f"_s{steps}_seed0_c{octaves}"
            )
            res = store.result(uid)
            if res is None:
                continue
            gate_cost += res.get("gpu_seconds", 0.0)
            gate_rows.append(res)
    gate_success = any(r.get("accepted") for r in gate_rows)

    candidates = [
        {
            "width": width,
            "min_n": row["min_n"],
            "gpu_seconds": row["gpu_seconds"],
            "success": True,
            "seed": row["seed"],
            "rows": row["rows"],
        }
        for row in per_seed
        if row["success"]
    ]
    if gate_success:
        candidates.append(
            {
                "width": cfg["widths"][-1],
                "min_n": sizes[-1],
                "gpu_seconds": gate_cost,
                "success": True,
                "seed": 0,
                "rows": [],
            }
        )
    if not candidates:
        return {
            "success": False,
            "min_n": None,
            "gpu_hours": min(r["gpu_seconds"] for r in per_seed) / 3600.0,
            "width": width,
            "per_seed": per_seed,
            "rows": per_seed[0]["rows"],
            "gate_rows": gate_rows,
            "note": "C did not reach high fidelity at any tested width/budget.",
        }
    cheapest = min(candidates, key=lambda c: c["gpu_seconds"])
    ladder_candidates = [c for c in candidates if c.get("rows")]
    min_n_cand = (
        min(ladder_candidates, key=lambda c: (c["min_n"], c["gpu_seconds"]))
        if ladder_candidates else cheapest
    )
    return {
        "success": True,
        "min_n": min_n_cand["min_n"],
        "min_n_gpu_hours": min_n_cand["gpu_seconds"] / 3600.0,
        "cheapest_n": cheapest["min_n"],
        "cheapest_gpu_hours": cheapest["gpu_seconds"] / 3600.0,
        "gpu_hours": cheapest["gpu_seconds"] / 3600.0,
        "width": cheapest["width"],
        "gpu_seconds": cheapest["gpu_seconds"],
        "seed": min_n_cand["seed"],
        "per_seed": per_seed,
        "rows": min_n_cand["rows"],
        "gate_rows": [
            {
                "lr": r.get("lr"),
                "steps": r.get("steps"),
                "accepted": r.get("accepted"),
                "gpu_seconds": r.get("gpu_seconds"),
            }
            for r in gate_rows
        ],
        "gate_success": gate_success,
    }


# --------------------------------------------------------------------------- #
# attacks
# --------------------------------------------------------------------------- #
def _load_model(
    store: Store, uid: str, arm: str, width: int, device: str, depth: int = 4
):
    ckpt = torch.load(
        store.models_dir / f"{uid}.pt", map_location="cpu", weights_only=True
    )
    model = build_arm(arm, width=width, depth=depth)
    model.load_state_dict(ckpt["state"])
    return model.to(device)


def attack_seed(
    cfg: dict, store: Store, rung: dict, seed: int
) -> dict:
    device = cfg["device"]
    arm, width, steps = rung["arm"], rung["width"], rung["steps"]
    octaves = rung["octaves"]
    data = pools_for(cfg, store, seed, octaves)
    secret = secret_for(seed)
    uid = f"{arm}_w{width}_s{steps}_seed{seed}_c{octaves}"
    model = _load_model(store, uid, arm, width, device, cfg["depth"])

    out = {}
    names = ("finetune", "key_recovery", "hamming")
    for name in names:
        a_uid = f"ATK_{name}_{arm}_w{width}_s{steps}_seed{seed}_c{octaves}"
        if name == "finetune":
            fn = lambda: finetune_curve(model, arm, data, secret, cfg, seed, device)
        elif name == "key_recovery":
            fn = lambda: key_recovery(model, data, secret, cfg, seed, device)
        else:
            fn = lambda: hamming_curve(model, data, secret, cfg, seed, device)
        out[name] = store.ensure(a_uid, fn, strict=False)
    return out


def run_attacks(
    cfg: dict, store: Store, primary: dict
) -> dict:
    device = cfg["device"]
    octaves = primary["octaves"]
    seeds = cfg["seeds"]

    def block(rung: dict) -> dict:
        return {
            "arm": rung["arm"],
            "width": rung["width"],
            "steps": rung["steps"],
            "seeds": {
                str(s): attack_seed(cfg, store, rung, s)
                for s in rung["seeds"]
            },
        }

    out = {"primary": block(primary), "upper": None}

    # One width up, same budget, if the budget allows.
    widths = cfg["widths"]
    try:
        idx = widths.index(primary["width"])
        upper_width = widths[idx + 1]
    except (ValueError, IndexError):
        upper_width = None
    if upper_width is not None and store.state["gpu_seconds"] < store.budget_seconds * 0.85:
        results = {
            s: unit_B(cfg, store, primary["arm"], s, octaves, upper_width, primary["steps"])
            for s in seeds
        }
        summary = _arm_summary(results)
        if summary["accepted"]:
            rung = {
                "arm": primary["arm"],
                "width": upper_width,
                "steps": primary["steps"],
                "octaves": octaves,
                "seeds": summary["accepted_seeds"],
            }
            out["upper"] = block(rung)
            out["upper"]["n_accepted"] = summary["n_accepted"]
    return out


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def build_summary(
    cfg: dict,
    store: Store,
    octaves: int,
    ladder: list[dict],
    accepted: dict | None,
    attacks: dict | None,
    c_cost: dict,
) -> dict:
    widths = sorted({r["width"] for r in ladder})
    a_by_width = {}
    for width in widths:
        chosen_rows = []
        for seed in cfg["seeds"]:
            found = []
            for steps in cfg["step_ladder"]:
                res = store.result(f"A_w{width}_s{steps}_seed{seed}_c{octaves}")
                if res is not None:
                    found.append(res)
            if found:
                accepted_rows = [r for r in found if r.get("accepted")]
                chosen_rows.append(accepted_rows[0] if accepted_rows else found[-1])
        a_by_width[width] = {
            "accepted_seeds": [r["seed"] for r in chosen_rows if r.get("accepted")],
            "n": len(chosen_rows),
            "seeds": {
                str(r["seed"]): {
                    "accepted": bool(r.get("accepted")),
                    "steps": r.get("steps"),
                    "test_f": r.get("test_f"),
                    "test_g": r.get("test_g"),
                }
                for r in chosen_rows
            },
        }
    a_ok = any(len(v["accepted_seeds"]) >= 2 for v in a_by_width.values())

    return {
        "config": cfg,
        "octaves": octaves,
        "octave_gate": store.state["decisions"].get("octave_gate"),
        "ladder": ladder,
        "accepted": accepted,
        "attacks": attacks,
        "a": {"accepted": a_ok, "by_width": a_by_width},
        "c": c_cost,
        "ladder_complete": bool(
            store.state["decisions"].get("ladder_complete", False)
        ),
        "budget": {
            "gpu_hours": store.state["gpu_seconds"] / 3600.0,
            "limit_hours": cfg["budget_gpu_hours"],
        },
    }
