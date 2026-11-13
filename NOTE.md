# Entanglement Hides the Representation, Not the Function

Version 1.0 · 13 November 2026 · Not peer reviewed. Every number is traced in `CLAIMS.md`.

## 0. Abstract

Can a license check be embedded in the task's learned representation, so that removing it becomes expensive? We ran three pre-registered experiments on CPU: a key-gated procedural world, a keyed expansive cellular automaton, and a keyed tent map. Deleting an external gate cost no optimization. A keyless student reached R² = 0.993 and 0.967 biome accuracy in 0.143 CPU-hours, below the one-GPU-hour cap. In one of three seeds, wrong keys at Hamming 50–53 reproduced the world at R² up to 0.982, while gradient key recovery failed. A 512-bit rule was learned exactly or not at all. The smooth map gave a flat sample lever and is inconclusive. Where the measurement resolved, bypass cost tracked the learnability and compute cost of the function, not entanglement. Nothing here generalizes to a deployed DRM system, to black-box models in general, or to commercial products.

## 1. Scope box

**What this is.** A small, pre-registered study of three keyed function families, run on CPU, with frozen verdict rules and every failed run kept.

**What this is not.** Not an evaluation of a commercial DRM product or platform; not a piracy study; not peer reviewed. It makes no general claim about entanglement and evaluates no deployed system. The world is a toy height and biome field; one key, three seeds, a 128-step horizon, one network width.

## 2. Question and threat model

A classical license check is a separate branch: `if valid(key): f(x) else g(x)`. An attacker who edits the client deletes it and keeps `f`. Entanglement asks whether putting the check inside the same parameters makes deletion useless.

The attacker sees the code and weights, can query a licensed installation, and does not know the key. Memory capture during licensed execution is out of scope and assumed to defeat the scheme; weights can be copied. Three families follow: (A) delete the gate; (B) read parameters or recover the key; (C) distil I/O into a keyless student. Success means matching the world at R² ≥ 0.95 and biome accuracy ≥ 0.95 without the key — producing the function, not stealing the key.

## 3. Setup

The world is deterministic hash-grid value noise: eight octaves, base frequency 2, amplitudes halving per octave, normalized to [0,1], plus an independent moisture field for four biomes. `g`, with another seed, is the decoy.

The key is 64 bits per seed, entering as {−1,+1}^64. Training mixes 50% correct, 25% uniform wrong, 12.5% Hamming-1, and 12.5% Hamming-2 keys. A is an external gate around task and decoy networks. B1 concatenates the key to the coordinates. B2 uses a learned key encoder (64→128→64) and per-layer FiLM. C is keyless. All arms share four GELU layers, batch 512, Adam at 10⁻³, and 10× height MSE plus biome CE. The decision run used width 256, 65,536/4,096/100,000 train/validation/test points, three seeds, and 20,000 steps (60,000 ceiling), consuming 1.91 CPU-hours (2.32 total).

Eight octaves were nearly collinear with four (r = 0.9978 on 100,000 points), so octaves are not an independent difficulty knob. The width ladder stopped at 256 because B2 was accepted; width scaling is unmeasured.

```
python run_decision.py --config configs/decision_cpu.json --output results/<new> --phase all
```

**Figures.** `decision_hamming.png`, `decision_finetune_vs_n.png`, `decision_costs.png`, `expansive_t95_vs_n.png`, `tent_t95_vs_n.png`.

## 4. Experiment 1 — a gate learned as a region, capped by distillation

Arm A passed 3/3 (R² 0.996, biome accuracy 0.980) at zero cost. B1 passed 2/3; B2 passed 3/3 (validation R² ≈ 0.995, biome accuracy ≈ 0.980) and tracked the decoy at R² ≈ 0.996 under a wrong key.

Key recovery did not. A 64-vector was optimized against 1,000 oracle examples for 1,000 steps from 20 starts, rounded to ±1, without the secret. The closest was 29/64 bits away.

Fine-tuning partially opened it. With the key frozen at all-minus-one, five of six keys reached high fidelity at n = 10,000 in seed 0 and at n = 100,000 in seed 1; seed 2 never exceeded one of six. Six-of-six was never met. Minimum n: 10,000 / 100,000 / none — a partial bypass in two seeds.

The privileged Hamming scan scored three wrong keys per distance on 20,000 points. One bit flip destroyed the output (R² = −3.71); distances 1–49 stayed negative. In seed 0, five of forty-two probes in the 45–58 band, all at 50–53, exceeded R² = 0.89; the best reached 0.982 (biome accuracy 0.976). Seeds 1 and 2 had none above 0.5. The gate is not a point, but this is a seed-specific island, not a general shrinking of key space; the scan is privileged, so it is diagnostic.

Distillation capped the result. A keyless student trained on analytic labels, a stronger oracle than the licensed outputs, reached R² 0.9926 and 0.9669 accuracy at n = 100,000 (60,000 steps). The cheapest protocol cost 515.6 s including a failed 20,000-step attempt; the first successful n cost 3,352.3 s cumulatively. Both are below one GPU-hour, measured as CPU wall clock. In the cheapest seed, fine-tuning cost 294.5 s, 0.571 times the cheapest distillation protocol (at a different sample budget).

Verdict: **CAPPED** — cheap keyless route, not an easy gate.

## 5. Experiment 2 — a discrete rule is learned or not

The second experiment used an 8×8 binary cellular automaton: each cell's next bit is a keyed lookup over its k×k neighbourhood. k = 3 is a 512-bit rule, k = 4 65,536 bits, k = 5 33.5 million bits. A CNN student (three seeds, 2,000 steps) saw one-step pairs and ran autonomously for 128 steps; T95 is the first step with cell accuracy below 0.95, capped at 129.

At k = 3 the transition is sharp. With n = 3,000, mean one-step accuracy was 0.9987 but T95 averaged 87 (range 3–129): seed 0 died at step 3 while seeds 1 and 2 reached the full horizon. At n = 10,000 and 30,000 all seeds reached 1.0000 and T95 = 129. Either the rule is recovered exactly, making the horizon irrelevant, or one-step error kills the trajectory.

The best k = 4 calibration run reached 0.8849 at n = 100,000, 4,000 steps, 256 channels; this is below the required 0.99 one-step accuracy. k = 5 sat at chance (0.4989, 0.5001). Because k = 4 failed, the exponential-in-T mechanism was not isolated, and POLYNOMIAL rests on the k = 3 control.

Verdict: **POLYNOMIAL**. The barrier is learnability and description length, not exponential amplification. The k = 4 and 5 rows are calibration; T95 = 129 is censored.

## 6. Experiment 3 — a smooth map, inconclusive

The third experiment used a keyed tent map on a 16-bit fixed-point state, with a keyed slope a in [1.9, 2.0): expansive, piecewise linear. A two-layer, 256-wide MLP (three seeds, 4,000 steps) learned one-step transitions from 10,000, 30,000, or 100,000 samples and rolled out for 128 steps; T95 is the first step with relative state error above 5%.

The sample lever was flat. One-step RMSE was 4.56×10⁻³, 4.00×10⁻³, and 4.64×10⁻³; T95 was 6 in all nine runs, with no seed deviation, and the largest n was worse than the middle one. The pre-registered fit gave a slope of 1.6×10⁻¹⁶ and R² = 0.

T95 is one crossing of a 5% threshold; all runs sit at 0.4–0.46% one-step error, so it cannot resolve differences of this size. A 512-unit, three-layer, 12,000-step calibration model reached RMSE 9.4×10⁻⁴ and T95 = 8, but it changes width, depth, and steps together.

Verdict: **INCONCLUSIVE**, not "flat, therefore no effect". The experiment lacked the resolution to distinguish a flat sample lever from a weak one.

## 7. What travels

Table 1 summarizes. In this setup:

1. A learned key gate is not a point. One seed accepted wrong keys at Hamming 50–53 at R² up to 0.982, while gradient search failed. Hamming distance is not a security metric.
2. Distillation, not key extraction, set the bypass cost. A keyless student matched high fidelity in 0.143 CPU-hours; the cheapest fine-tune seed cost 0.571 times that.
3. A keyed discrete rule produced a learnability threshold, not exponential-in-horizon cost.
4. A smooth keyed map showed no measurable sample lever and a T95 that moved only when model size and training length changed together, but cannot establish the scaling law.

Where the measurement resolved, the property we varied did not bind; the cost of learning the function did.

| Experiment | Decisive measurement (this setup) | Verdict |
|---|---|---|
| Decision, w = 256, c = 8, 3 seeds | Keyless student R² 0.993 / acc 0.967 in 0.143 CPU-h; fine-tune 5/6 keys at n = 10k; key recovery 29/64 | CAPPED |
| Expansive CA, k = 3 control, 3 seeds | One-step 1.000, T95 = 129 at n ≥ 10k; at n = 3k one seed T95 = 3 | POLYNOMIAL |
| Tent map, 3 seeds | RMSE 4.0–4.6×10⁻³, T95 = 6 in 9/9 runs | INCONCLUSIVE |

## 8. Related work

Obfuscation theory says general programs cannot be hidden (Barak et al., 2001), and practical indistinguishability obfuscation remains expensive (Garg et al., 2013). Our observation is weaker: a representation can hide where the key is without hiding the function. Watermarking (Adi et al., 2018) acts on representation, but for provenance, not access. Model extraction (Tramèr et al., 2016) and distillation (Hinton et al., 2015) are the relevant attacks.

The closest neighbour is Arun, Abraham, and Lekshmi P (2026), who audit subspace disentanglement in a dual-encoder paraphrase model. They find near-perfect geometric separation (pairwise CKA ≈ 0.0007) without functional specialization: each subspace classifies paraphrase at F1 above 0.986, and removing the penalty costs 0.013 F1 at matched capacity. A geometric metric can certify a property the function does not have. Ours is the control-side analogue: an entanglement metric can look like security while the function stays learnable. Locatello et al. (2019, arXiv:1811.12359) make a related point; Athalye, Carlini, and Wagner (ICML 2018) warn that a weak attack can make a defense look strong.

## 9. What would change the conclusion

A paper needs a testbed with a visible sample lever: one-step error falling by orders of magnitude, not flooring at 0.4%, T95 varying rather than sticking at 6, and a compute sweep that separates compute from samples. A width sweep, a harder task, and a measured cost above one GPU-hour would test claims 1–2.

## 10. Reproduction

Repository: `src/`, `configs/`, `results/` (raw JSON and checkpoints), `figures/`. Verdict and kill rules were frozen before the runs; failures remain.

```
python -m unittest discover -s tests
python run_decision.py --config configs/decision_cpu.json --output results/<new> --phase all
python experiments/run_expansive.py --config configs/expansive.json
python experiments/run_tent.py --config configs/tent.json
python experiments/analyze_expansive.py
python experiments/analyze_tent.py
```

Decision run: 1.91 CPU-hours on six threads; CA 0.36; tent 0.05; total 2.32. No GPU was used. Code is MIT; text and figures are CC BY 4.0.
