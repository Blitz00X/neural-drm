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
