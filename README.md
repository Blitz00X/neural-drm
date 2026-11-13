# Neural DRM — Phase 0

**Status: public research note, version 1.0.1 (13 November 2026). Not peer reviewed.**

A small, pre-registered study of whether embedding a license check in the same
learned representation as a task makes the check expensive to remove. The short
answer, in this setup, is no: the attacker's cost tracked the learnability and
compute cost of the underlying function, not the entanglement. This repository
contains the code, the frozen verdict rules, every raw run (including failures),
and the three verdicts.

Read **[`NOTE.md`](NOTE.md)** first. It is the paper. `CLAIMS.md` traces every
number in it back to a file in `results/` and marks the two claims that were
dropped for lack of an artifact.

This is a toy procedural world, not a commercial DRM system, game engine, or
renderer. Read `docs/negative-result.md` before citing any number.

## What is here

- **Decision experiment** (`VERDICT.md`, **CAPPED**): a key-gated world at width
  256, three seeds. Arm A deletes the gate for free; B2 passes training
  acceptance 3/3; gradient key recovery fails; fine-tuning opens five of six
  keys in two seeds; a keyless student reaches high fidelity in 0.143 CPU-hours.
  A privileged Hamming scan finds a secondary accepted region in one seed at
  distance 50–53.
- **Expansive cellular automaton** (`VERDICT_EXPANSIVE.md`, **POLYNOMIAL**): an
  8×8 binary grid with a keyed k×k rule. At k=3 the rule is learned exactly at
  n ≥ 10,000 and the horizon becomes irrelevant; at n = 3,000 one seed dies at
  step 3 and two reach the full horizon. The k=4 precondition failed in
  calibration (0.8849 < 0.99), so the exponential-in-T mechanism was not
  isolated.
- **Smooth tent map** (`VERDICT_TENT.md`, **INCONCLUSIVE**): a 16-bit fixed-point
  keyed tent map. One-step RMSE is 4.0–4.6×10⁻³ and T95 is 6 in all nine runs;
  the sample lever is flat at this resolution. The experiment cannot distinguish
  a flat lever from a weak one.

`docs/negative-result.md` is the one-page synthesis. `HANDOFF.md` records the
state, environment, and resume protocol.

## Quick start

Python 3.10+ and PyTorch. The recorded decision run used Python 3.11.15 on CPU;
no GPU was available. Exact versions are in `requirements.lock.txt` and
`results/decision_cpu_v1/environment.json`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

## Reproduce

Every output path must be new: the runner refuses to overwrite an earlier run.
Run from the repository root.

```bash
# Decision experiment (CPU configuration; about 1.9 CPU-hours on six threads)
python run_decision.py --config configs/decision_cpu.json \
  --output results/reproduce_decision --phase all

# Expansive cellular automaton (about 0.36 CPU-hours)
python experiments/run_expansive.py --config configs/expansive.json
python experiments/analyze_expansive.py

# Smooth tent map (about 0.05 CPU-hours)
python experiments/run_tent.py --config configs/tent.json
python experiments/analyze_tent.py
```

The GPU configuration (`configs/decision.json`) is kept for a machine with a
working driver; it was not run here. The CPU run is the recorded result.

## Results at a glance

| Experiment | Verdict | Decisive measurement |
|---|---|---|
| Decision, w = 256, c = 8, 3 seeds | CAPPED | Keyless student R² 0.993 / acc 0.967 in 0.143 CPU-h; fine-tune 5/6 keys at n = 10k; key recovery best 29/64 |
| Expansive CA, k = 3 control, 3 seeds | POLYNOMIAL | One-step 1.000, T95 = 129 at n ≥ 10k; at n = 3k one seed T95 = 3 |
| Tent map, 3 seeds | INCONCLUSIVE | RMSE 4.0–4.6×10⁻³, T95 = 6 in 9/9 runs |

Total recorded compute: 2.32 CPU-hours; 0 GPU-hours.

## Figures

- `figures/decision_hamming.png` — privileged Hamming scan. Seed 0 shows a
  secondary accepted region at distance 50–53; seeds 1–2 are flat.
- `figures/decision_finetune_vs_n.png` — worst-key fidelity versus fine-tune
  sample budget.
- `figures/decision_costs.png` — attacker cost for fine-tune, key recovery, and
  distillation.
- `figures/expansive_t95_vs_n.png` — CA survival horizon versus sample budget.
- `figures/expansive_acc_vs_n.png` — CA one-step accuracy versus sample budget.
- `figures/tent_t95_vs_n.png` — tent T95 versus sample budget.
- `figures/tent_rmse_vs_n.png` — tent one-step RMSE versus sample budget.

Missing measurements are labelled as missing in the figures; no curve is
fabricated.

## What the results do and do not claim

They claim, in this setup: a learned key gate is not a point; distillation set
the bypass cost; a keyed discrete rule produced a learnability threshold; and a
smooth keyed map was inconclusive at this resolution.

They do not generalize to deployed DRM, to black-box models, or to entanglement
in general, and they do not transfer to any commercial system. The scope is one
toy world, one key per seed, three seeds, a 128-step horizon, and one width in
the decision experiment.

## Verification

`verify_claims.py` recomputes the aggregates, counts, ranges and ratios quoted
in `NOTE.md` from the per-run JSON under `results/`. It does not retrain models.

```bash
python verify_claims.py .
```

Its recorded output is `verify_claims.out.txt`.

## License

- **Code**: MIT — see [`LICENSE`](LICENSE).
- **Text, figures and results**: Creative Commons Attribution 4.0 International
  (CC BY 4.0) — see [`LICENSE-CC-BY-4.0.txt`](LICENSE-CC-BY-4.0.txt).

## DOI and citation

No DOI is registered yet. Until one is assigned, cite the repository and the
commit hash. A [`CITATION.cff`](CITATION.cff) is included.

## Provenance

- Hypothesis files and verdict rules were committed before the runs and were
  not edited afterwards: `hypotheses.md`, `hypothesis.md`,
  `hypothesis_tent.md`, `verdict_rules.md`, `kill_rule.md`, `kill_rule_tent.md`.
- Failed, NaN, and partial runs are kept and counted; they are not deleted.
- The decision run is CPU-only because the installed GPU driver failed at
  kernel level; `results/decision_cpu_v1/environment.json` records
  `cuda_available: false`.
- The five-seed phase-0 cell and the three-seed decision run are different
  experiments; do not pool them.
