# Neural DRM — Phase 0

A reproducible toy experiment, **not a claim of effective DRM**. Read REPORT.md
and DECISIONS.md before interpreting any number. The original task is docs/TASK.md.
Hypotheses were committed before training (849ba73); do not edit hypotheses.md.

## Environment
Python 3.10+ and PyTorch. Recorded execution uses Python 3.12 on CPU.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

For a CPU-only environment, installing torch from its CPU wheel index avoids
unnecessary CUDA libraries. Existing supported PyTorch installations can be used;
compare environment.json and requirements.lock.txt for exact execution versions.

## Reproduction
Every output path must be NEW: the runner refuses to overwrite a previous run.
Run from this directory. Seeds, training budget, loss weights, architecture and
attack budgets are in the committed configuration and source.

```bash
# Original short pilot: known H1 failure is intentionally preserved.
python -m src.run --config configs/pilot.json --output results/reproduce_pilot_001
# Fixed-budget repair and full width64,c4 experiment for seed0.
python -m src.run --config configs/pilot_repair.json --output results/reproduce_seed0
# Remaining four seeds of the measured CPU cell.
python -m src.run --config configs/cpu_seed1.json --output results/reproduce_seed1
python -m src.run --config configs/cpu_seed2.json --output results/reproduce_seed2
python -m src.run --config configs/cpu_seed3.json --output results/reproduce_seed3
python -m src.run --config configs/cpu_seed4.json --output results/reproduce_seed4
# Original requested eight-cell grid: substantial compute, NOT executed here.
python -m src.grid --config configs/full_grid.json --output results/new_full_grid
# Rebuild delivered tables, figures and report from the five canonical result roots.
python -m src.analyze
```

Use a copy of the configuration with device="cuda" on a GPU; record it as a new
configuration, never silently change an existing run. CUDA determinism may require
CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python.

## Layout and interpretation
- `src/data.py`: deterministic hash-grid value noise, four biomes, non-overlapping pools.
- `src/models.py`: four GELU hidden layers, external gate A, concat B1, FiLM B2, C.
- `src/train.py`: seeded training, samples processed, 6*parameters*samples FLOPs.
- `src/attacks/`: ablation, fine-tune, key recovery, SAE/probe and distillation.
- `results/*/seed_*/config.json`: full configuration plus seed for each execution.
- `*.pt`: defender checkpoint (contains secret). These are WHITE-BOX defender
  artifacts; secret-free optimization attacks are not passed the secret.
- `STOP.json`: H1 stopping condition. `COMPLETE.json` means runner completed,
  not that a model was accepted or the scientific hypotheses were resolved.
- `results/test_*.log`: failing-first development checks and passing verifications.
- `figures/`: missing measurements explicitly labeled, never fabricated as curves.

H1 failure stops that seed before B/C attack inference. Rejected B models are
excluded from attacks. Activation ablation, Hamming diagnostics and SAE/probes use
privileged access to the licensed state and are explicitly marked as such.
Fixed-key fine-tuning and continuous key recovery never receive the secret.
Key recovery success is distinguished from six-key key-independent bypass.

C learns exact analytic f labels, a stronger oracle than outputs of an imperfect
licensed neural model. Validation chooses the learning rate; all six unauthorized
keys must meet both .95 thresholds for fine-tune/ablation bypass success. Do not
interpret low sensitivity, a good height score alone, an untested attack, or a
budget-censored cost as a successful defense.

## Decision experiment (phase 1, `VERDICT.md`)

This is the preregistered single-decision run: one label, three attack families,
an 8 GPU-hour cap.  The rules in `verdict_rules.md` were committed before the
first run and must not be edited afterwards.

```bash
python run_decision.py --phase all
# or resume a single phase: gate | ladder | c | attacks | report
python run_decision.py --phase ladder
```

- `configs/decision.json`: widths 256/512/1024, budgets 20k/60k, 3 seeds,
  c=8 primary with a c=4 fallback, 100k test points, 8 GPU-hour cap.
- `results/decision_v1/units/`: one immutable JSON per unit (training run,
  attack, C trial); `models/` holds checkpoints; `state.json` is the resume
  ledger and the GPU-hour budget.
- `results/decision_v1/summary.json`: the machine-readable evidence behind the
  label; `verdict.json` is the rule evaluation.
- `figures/decision_*.png`: fine-tune/distillation fidelity vs n, attacker cost,
  Hamming leakage.
- `VERDICT.md`: one Turkish page, first line `VERDICT: <label>`.

The run is append-only: rerunning skips completed units and never overwrites a
failed one.  A crash is resumed by rerunning the same command.  `VERDICT:
INCONCLUSIVE` is written when the budget or a missing measurement prevents an
ordered rule from matching.

### CPU fallback run (2026-10-21)

The installed NVIDIA driver had crashed at kernel level on 19 Oct; CUDA could
create a context but not allocate even a few bytes (`cudaSetDevice` returned
OOM while `nvidia-smi` reported 3.7 GiB free).  A reboot would be required and
no passwordless sudo was available.  The frozen protocol was therefore run on
CPU at the first ladder rung (w=256), which is where the ladder stops because
B was accepted:

```bash
python run_decision.py --config configs/decision_cpu.json \
  --output results/decision_cpu_v1 --phase all
```

Result: `VERDICT: CAPPED`.  Deviations are recorded in `DECISIONS.md` and in the
limitations paragraph of `VERDICT.md`: CPU instead of GPU (0 GPU-hours), the
octave gate at w=256 instead of w=1024, and no upper 512 run.  The GPU path
(`configs/decision.json`) is unchanged and ready for a machine with a working
driver; `results/pilot_preview_256/` is the one-seed preview, not a verdict.

## Expansive chained function (goal-loop, `VERDICT_EXPANSIVE.md`)

Tests whether a keyed, integer-only, expansive cellular automaton makes
black-box distillation unusable over long autonomous rollouts.  The state is an
8x8 binary grid; each cell's next bit is a keyed lookup of its k x k
neighbourhood.  k=3 is a 512-bit control; k=4 is 65,536 bits; k=5 is 33.5M bits.

```bash
python experiments/run_expansive.py --config configs/expansive.json
python experiments/analyze_expansive.py
head -1 VERDICT_EXPANSIVE.md
```

- `hypothesis.md` and `kill_rule.md` were frozen before the confirmatory sweep.
- `results/expansive_v1/runs/`: one JSON per (k, n, seed), with one-step accuracy,
  T95, distribution TV, exact-state accuracy and timings.
- `results/expansive_v1/calibration.json`: pre-freeze calibration table (not
  confirmatory evidence).
- `figures/expansive_*.png`: T95 and one-step accuracy vs sample budget.

The `.goal/` directory holds the goal-loop manifest and an append-only log.

## Negative result (2026-10-22)

The three experiments are consolidated in `docs/negative-result.md`:
`VERDICT.md` (CAPPED), `VERDICT_EXPANSIVE.md` (POLYNOMIAL) and
`VERDICT_TENT.md` (INCONCLUSIVE).  The shared conclusion is that entanglement
hides representation, not the function; the security ceiling is task
learnability/compute, not superposition.  Read that note before citing any
number from this repository.
