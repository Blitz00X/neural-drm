"""Tiny CPU/GPU smoke test for every decision code path (not a scientific run)."""
import numpy as np
import torch

from src.decision.attacks import (
    finetune_curve,
    hamming_curve,
    key_recovery,
    train_student,
)
from src.decision.models import ConcatKeyed, FilmKeyed, PlainWorld
from src.decision.trainer import (
    build_pools,
    evaluate_keyed,
    seed_all,
    to_device,
    train,
    wrong_strata,
)

CFG = {
    "batch": 16,
    "lr": 1e-3,
    "depth": 2,
    "attack": {
        "finetune_sample_sizes": [10, 100],
        "finetune_steps": 10,
        "finetune_lrs": [1e-3],
        "distill_lrs": [1e-3],
        "distill_step_ladder": [10, 20],
        "key_oracle_n": 50,
        "key_restarts": 2,
        "key_steps": 10,
        "hamming_repeats": 1,
        "hamming_test_n": 100,
    },
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    data = build_pools(0, 2, train_n=256, val_n=64, test_n=500, oracle_n=256)
    secret = (np.random.default_rng(42).integers(0, 2, 64) * 2 - 1).astype(np.float32)
    tensors = to_device(data, device)

    for arm, model in (("B1", ConcatKeyed(16, depth=2)), ("B2", FilmKeyed(16, depth=2))):
        seed_all(0)
        model.to(device)
        train(
            model,
            tensors,
            steps=5,
            batch=16,
            lr=1e-3,
            seed=0,
            keyed=True,
            secret=torch.as_tensor(secret, device=device),
            device=device,
        )
        valid = evaluate_keyed(
            model, data["tx"], data["tfh"], data["tfb"], secret, device=device
        )
        wrong = wrong_strata(model, data, secret, 0, device=device)
        print(arm, "valid", round(valid["r2"], 3), "wrong keys", len(wrong))
        ft = finetune_curve(model, arm, data, secret, CFG, 0, device)
        print(arm, "finetune rows", len(ft["rows"]), "min_n", ft["min_n"])
        kr = key_recovery(model, data, secret, CFG, 0, device)
        print(arm, "key recovery success", kr["success"])
        ham = hamming_curve(model, data, secret, CFG, 0, device)
        print(arm, "hamming rows", len(ham["rows"]))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    model, _, _, test, seconds = train_student(
        16, 100, 10, 1e-3, data, 0, device, 16, 2
    )
    print("C student", round(test["r2"], 3), round(seconds, 3))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
