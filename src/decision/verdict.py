"""Ordered verdict rules, frozen in verdict_rules.md before the run.

Pure functions over the summary dictionary so the decision can be unit-tested
without a GPU.
"""
from __future__ import annotations


def _accepted_rungs(summary: dict) -> list[dict]:
    return [r for r in summary.get("ladder", []) if r.get("accepted")]


def _valid_row(row: dict) -> bool:
    """A seed row supports a verdict only if all three attacks produced data."""
    ft = row.get("finetune") or {}
    kr = row.get("key_recovery") or {}
    if ft.get("status") == "failed" or "rows" not in ft:
        return False
    if kr.get("status") == "failed" or "rows" not in kr:
        return False
    return True


def _attack_rows(summary: dict, rung: dict) -> dict[int, dict]:
    key = (rung["arm"], rung["width"], rung["steps"])
    attacks = summary.get("attacks") or {}
    if not attacks:
        return {}
    for block in (attacks.get("primary"), attacks.get("upper")):
        if not block:
            continue
        if (block["arm"], block["width"], block["steps"]) == key:
            return {int(k): v for k, v in block["seeds"].items()}
    return {}


def _min_n(seed_row: dict):
    return seed_row.get("finetune", {}).get("min_n")


def _ft_cost(seed_row: dict, n) -> float:
    if n is None:
        return 0.0
    for row in seed_row.get("finetune", {}).get("rows", []):
        if row["n"] == n:
            return float(row.get("cumulative_gpu_seconds", 0.0))
    return 0.0


def compute_verdict(summary: dict) -> dict:
    rungs = _accepted_rungs(summary)
    a_ok = bool(summary.get("a", {}).get("accepted"))
    c_ok = bool(summary.get("c", {}).get("success"))
    c_hours = float(summary.get("c", {}).get("gpu_hours") or 0.0)

    if a_ok and c_ok and not rungs and summary.get("ladder_complete"):
        return {
            "label": "DEAD_CAPACITY",
            "rule": 1,
            "reason": "A and C accepted; no B1/B2 combination passed training acceptance.",
            "numbers": _numbers(summary, None),
        }
    if not rungs:
        return {
            "label": "INCONCLUSIVE",
            "rule": None,
            "reason": (
                "No accepted B; DEAD_CAPACITY requires a completed ladder and "
                "A+C acceptance. ladder_complete=%s a_ok=%s c_ok=%s"
                % (summary.get("ladder_complete"), a_ok, c_ok)
            ),
            "numbers": _numbers(summary, None),
        }

    # Per-rung attack evidence.  A missing attack block cannot support ALIVE.
    evidence = []
    for rung in rungs:
        rows = _attack_rows(summary, rung)
        if rows:
            evidence.append((rung, rows))
    if not evidence:
        return {
            "label": "INCONCLUSIVE",
            "rule": None,
            "reason": "B accepted but no attack evidence is available.",
            "numbers": _numbers(summary, None),
        }

    def dead_ft():
        for rung, rows in evidence:
            good = [
                s
                for s, row in rows.items()
                if _min_n(row) is not None and _min_n(row) <= 1000
            ]
            if len(good) >= 2:
                return rung, good
        return None

    hit = dead_ft()
    if hit:
        rung, seeds = hit
        return {
            "label": "DEAD_FINETUNE",
            "rule": 2,
            "reason": "Fine-tune bypass with n<=1000 on >=2/3 seeds.",
            "numbers": _numbers(summary, rung, seeds),
        }

    hit_key = None
    for rung, rows in evidence:
        good = [s for s, row in rows.items() if row.get("key_recovery", {}).get("success")]
        if good:
            hit_key = (rung, good)
            break
    if hit_key:
        rung, seeds = hit_key
        return {
            "label": "DEAD_KEY",
            "rule": 3,
            "reason": "Key recovery succeeded with <=1000 oracle samples.",
            "numbers": _numbers(summary, rung, seeds),
        }

    fully_evidenced = all(
        sum(1 for row in rows.values() if _valid_row(row)) >= 2
        for _, rows in evidence
    )
    if c_ok and c_hours < 1.0 and fully_evidenced:
        return {
            "label": "CAPPED",
            "rule": 4,
            "reason": (
                "B survived fine-tune and key recovery, but C reaches high fidelity "
                "in under one GPU-hour; the protection is capped by task learnability."
            ),
            "numbers": _numbers(summary, rungs[0]),
        }

    def weak_ft():
        for rung, rows in evidence:
            good = [
                s
                for s, row in rows.items()
                if _min_n(row) is not None and 1000 < _min_n(row) <= 10000
            ]
            if len(good) >= 2:
                return rung, good
        return None

    hit = weak_ft()
    if hit:
        rung, seeds = hit
        return {
            "label": "ALIVE_WEAK",
            "rule": 5,
            "reason": "Fine-tune bypass needs 1000<n<=10000 on >=2/3 seeds.",
            "numbers": _numbers(summary, rung, seeds),
        }

    if c_ok and c_hours >= 1.0:
        ok_rungs = []
        for rung, rows in evidence:
            valid = {s: row for s, row in rows.items() if _valid_row(row)}
            if len(valid) < 2:
                continue
            hard = [
                s
                for s, row in valid.items()
                if _min_n(row) is None or _min_n(row) >= 100000
            ]
            if len(hard) >= 2:
                ok_rungs.append(rung)
        if ok_rungs:
            return {
                "label": "ALIVE",
                "rule": 6,
                "reason": (
                    "Fine-tune requires n>=100000 or fails on >=2/3 seeds; key "
                    "recovery fails; C needs >=1 GPU-hour."
                ),
                "numbers": _numbers(summary, ok_rungs[0]),
            }

    return {
        "label": "INCONCLUSIVE",
        "rule": None,
        "reason": "Accepted B exists but the observed attack pattern matches no ordered rule.",
        "numbers": _numbers(summary, rungs[0]),
    }


def _numbers(summary: dict, rung: dict | None, seeds: list[int] | None = None) -> dict:
    c = summary.get("c", {})
    out = {
        "a_cost_gpu_hours": 0.0,
        "c_min_n": c.get("min_n"),
        "c_min_n_cost_hours": c.get("min_n_gpu_hours"),
        "c_cheapest_n": c.get("cheapest_n", c.get("min_n")),
        "c_gpu_hours": c.get("gpu_hours"),
        "b_min_finetune_n": None,
        "b_finetune_gpu_hours": None,
        "b_over_c_cost_ratio": None,
        "key_recovery_success": None,
    }
    if rung is None:
        return out
    rows = _attack_rows(summary, rung)
    if seeds is None:
        seeds = sorted(rows)
    ns = [_min_n(rows[s]) for s in seeds if s in rows]
    known = [n for n in ns if n is not None]
    out["b_min_finetune_n"] = min(known) if known else None
    if known:
        n = min(known)
        costs = [_ft_cost(rows[s], n) for s in seeds if s in rows and _min_n(rows[s]) == n]
        if costs and any(c > 0 for c in costs):
            hours = (sum(costs) / len(costs)) / 3600.0
            out["b_finetune_gpu_hours"] = hours
            if c.get("gpu_hours"):
                out["b_over_c_cost_ratio"] = hours / float(c["gpu_hours"])
    out["key_recovery_success"] = any(
        rows[s].get("key_recovery", {}).get("success") for s in seeds if s in rows
    )
    return out
