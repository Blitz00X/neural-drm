# CLAIMS — Neural-DRM note draft

Companion to `NEURAL-DRM-NOTE-draft-2026-11-13.md`. Verification date: 2026-11-13.

All repo paths are relative to the source repository root (`neural-drm-phase0`). Status labels:

- **reproduced** — recomputed in this session from the raw per-run JSON, not from prose. No model was retrained; aggregates, counts, ranges, and ratios were recomputed.
- **in-repo** — stated in a committed repo document, configuration, or source file, and not independently recomputed here.
- **verified external** — checked against the publisher's page on 2026-11-13; not a repo artifact.
- **unsupported** — no artifact supports the number. Such numbers are **not used** in the note.

The recomputation script is `verify_claims.py`; its full output is `verify_claims.out.txt`. It reads:

`results/decision_cpu_v1/summary.json`, `results/decision_cpu_v1/verdict.json`, `results/decision_cpu_v1/state.json`, the C unit JSON, `results/expansive_v1/summary.json`, `results/expansive_v1/calibration.json`, the nine CA run JSONs, `results/tent_v1/summary.json`, `results/tent_v1/calibration.json`, and the nine tent run JSONs.

## Claim table

| # | Claim in NOTE | Value used | Source artifact | Status |
|---:|---|---|---|---|
| 1 | Three experiments, CPU-only | 3 runs; `cuda_available: false` | `configs/decision_cpu.json`, `configs/expansive.json`, `configs/tent.json`; `results/decision_cpu_v1/environment.json` | in-repo |
| 2 | World: hash-grid value noise, 8 octaves, 4 biomes, decoy `g` | 8; 4 | `REPORT.md` §2; `src/data.py` (`noise`, `world`, `keys`) | in-repo |
| 3 | 64-bit key, entered as {−1,+1}^64 | 64 | `REPORT.md` §2; `src/data.py` (`keys`) | in-repo |
| 4 | Training key mix 50 / 25 / 12.5 / 12.5% | 50, 25, 12.5, 12.5 | `REPORT.md` §2 | in-repo |
| 5 | Arms A, B1, B2, C | 4 arms | `REPORT.md` §2; `src/models.py`; `src/decision/models.py` | in-repo |
| 6 | 4 GELU layers, batch 512, Adam 1e-3, 10× height MSE + biome CE | 4; 512; 1e-3; 10 | `REPORT.md` §2; `configs/decision_cpu.json` | in-repo |
| 7 | Decision run: width 256; 65,536 / 4,096 / 100,000 points; 3 seeds; 20,000 steps with 60,000 ceiling | 256; 65,536; 4,096; 100,000; 3; 20,000; 60,000 | `configs/decision_cpu.json` | in-repo |
| 8 | Decision run 1.91 CPU-hours; all three experiments 2.32 (CA 0.36, tent 0.05) | 1.910947 h; 0.362367 h; 0.048576 h; 2.32189 h | `results/decision_cpu_v1/summary.json` (`budget`); sum of `total_seconds` in both sweep run folders | reproduced |
| 9 | c=8 and c=4 height fields nearly collinear | r = 0.997823 on 100,000 points | `DECISIONS.md` (pre-check, 0.9978); recomputed with `src.data.pools` + `src.data.world` | reproduced |
| 10 | Width ladder stopped at 256 because B2 was accepted | accepted width 256 | `DECISIONS.md`; `results/decision_cpu_v1/summary.json` (`accepted.width`) | in-repo |
| 11 | Arm A: 3/3 accepted; test R² 0.996; biome accuracy 0.980; zero optimization for gate deletion | R² 0.996055–0.996580; acc 0.97927–0.98073 | `results/decision_cpu_v1/summary.json` (`a.by_width.256.seeds`) | reproduced |
| 12 | B1 accepted in 2/3 seeds | seeds 0, 2 | `results/decision_cpu_v1/summary.json` (`accepted.arms.B1.accepted_seeds`) | reproduced |
| 13 | B2 accepted 3/3; validation R² ≈ 0.995; accuracy ≈ 0.980 | R² 0.995228–0.995572; acc 0.97972–0.98064 | `results/decision_cpu_v1/summary.json` (`accepted.arms.B2.seeds.*.test_valid`) | reproduced |
| 14 | B2 tracks decoy `g` at R² ≈ 0.996 under a wrong key | 0.99604–0.99638 across random/h1/h2 strata | `results/decision_cpu_v1/summary.json` (`accepted.arms.B2.seeds.*.wrong`) | reproduced |
| 15 | Key recovery: 1,000 oracle samples, 1,000 steps, 20 starts, ±1 rounding, no secret passed | 1,000; 1,000; 20 | `configs/decision_cpu.json` (`attack.key_*`); `results/decision_cpu_v1/summary.json` (`key_recovery`) | in-repo |
| 16 | Key recovery failed; best rounded key 29 of 64 bits away | 29 (seed 0); seeds 1 and 2: 30 and 35 | `results/decision_cpu_v1/summary.json` (`key_recovery.best`, `rows`) | reproduced |
| 17 | Fine-tune: 5/6 keys at n = 10k (seed 0); 5/6 at n = 100k (seed 1); seed 2 max 1/6; never 6/6 | 5, 5, 1; max 5 | `results/decision_cpu_v1/summary.json` (`finetune.rows[*].n_high_fidelity_keys`) | reproduced |
| 18 | Minimum fine-tune n: 10,000 / 100,000 / none | 10,000; 100,000; null | `results/decision_cpu_v1/summary.json` (`finetune.min_n`) | reproduced |
| 19 | Hamming scan: 3 wrong keys per distance; 20,000 test points | 3; 20,000 | `configs/decision_cpu.json` (`attack.hamming_repeats`, `hamming_test_n`) | in-repo |
| 20 | One bit flip destroys output (R² = −3.71); distances 1–49 remain negative | d=1 mean R² = −3.7103; range −3.6668 to −3.8129 | `results/decision_cpu_v1/summary.json` (`attacks.primary.seeds.0.hamming.rows`) | reproduced |
| 21 | In seed 0, 5 of 42 probes in the 45–58 band, all at 50–53, exceed R² = 0.89; best 0.982 / 0.976 | 5 / 42; d = 50, 51, 53, 53, 53; best R² 0.98222, acc 0.976 | same as #20 | reproduced |
| 22 | Seeds 1–2 have no wrong key above R² = 0.5 in the 45–58 band | seed 1 max 0.3913; seed 2 max 0.4754 (both inside the band, both below 0.5) | same as #20, seeds 1 and 2 | reproduced |
| 23 | C: R² 0.9926 (0.993 rounded); accuracy 0.9669 (0.967 rounded) at n = 100,000, 60,000 steps | 0.992619; 0.96691 | `results/decision_cpu_v1/units/C_w256_n100000_lr1e-04_s60000_seed0_c8.json` | reproduced |
| 24 | C cheapest protocol 515.6 s; first successful n 3,352.3 s cumulative | 515.610 s (134.339 + 381.271); 3,352.285 s | `results/decision_cpu_v1/summary.json` (`c`) | reproduced |
| 25 | Both C costs below one GPU-hour, measured as CPU wall clock | 0.143225 h; 0.931190 h | `results/decision_cpu_v1/verdict.json`; `DECISIONS.md` (CPU fallback) | in-repo (CPU-as-proxy assumption) |
| 26 | Cheapest fine-tune seed 294.5 s; 0.571× the cheapest C protocol | 294.543 s; 0.571252 | `results/decision_cpu_v1/summary.json` (`finetune.seed0.cumulative_gpu_seconds`); `verdict.json` (`b_over_c_cost_ratio`) | reproduced |
| 27 | CA: 8×8 binary grid; k=3 512 bits, k=4 65,536, k=5 33,554,432 (33.5M rounded) | 512; 65,536; 33,554,432 | `README.md`; `results/expansive_v1/calibration.json`; `hypothesis.md` | in-repo |
| 28 | CA student: CNN, 3 seeds, 2,000 steps, 128-step rollouts, threshold 0.95, cap 129 | 3; 2,000; 128; 0.95; 129 | `configs/expansive.json`; `src/expansive/*` | in-repo |
| 29 | k=3, n=3,000: one-step 0.9987; T95 mean 87 (3–129); seed 0 = 3, seeds 1–2 = 129 | 0.998747; 87.0; 3, 129, 129 | `results/expansive_v1/summary.json`; `runs/k3_n3000_seed*.json` | reproduced |
| 30 | k=3, n=10,000 and 30,000: one-step 1.0000; T95 = 129 | 1.0; 129 | `results/expansive_v1/summary.json`; run JSONs | reproduced |
| 31 | Best k=4 calibration: 0.8849 at n=100,000, 4,000 steps, 256 channels | 0.8849; 100,000; 4,000; 256 | `results/expansive_v1/calibration.json` row 5 | reproduced |
| 32 | k=5 at chance: 0.4989 and 0.5001 | 0.4989; 0.5001 | `results/expansive_v1/calibration.json` rows 6–7 | reproduced |
| 33 | k=4 precondition failed; POLYNOMIAL rests on the k=3 control | 0.8849 < 0.99 | `kill_rule.md`; `VERDICT_EXPANSIVE.md` | in-repo |
| 34 | T95 = 129 is censored, not a measured survival time | cap = T_max + 1 | `src/expansive/*`; `VERDICT_EXPANSIVE.md` | in-repo |
| 35 | Tent map: 16-bit fixed-point state; keyed slope in [1.9, 2.0); piecewise linear, expansive | 16; 1.9; 2.0 | `src/expansive/tent.py` (`BITS`, `key_params`) | in-repo |
| 36 | Tent student: 2 layers, 256 wide, 3 seeds, 4,000 steps, n = 10k/30k/100k, 128 steps, 5% threshold | 2; 256; 3; 4,000; 10,000/30,000/100,000; 128; 0.05 | `configs/tent.json`; `src/expansive/tent_student.py` | in-repo |
| 37 | RMSE 4.56 / 4.00 / 4.64 ×10⁻³ (4.0–4.6 rounded); T95 = 6 in 9/9 runs; largest n worse than middle | 4.5598e-3; 3.9951e-3; 4.6351e-3; 6 | `results/tent_v1/summary.json`; nine run JSONs | reproduced |
| 38 | T95 vs ln n slope 1.6×10⁻¹⁶; R² = 0 | 1.6282e-16; 0.0 | `results/tent_v1/summary.json` (`verdict`) | reproduced |
| 39 | All runs sit at 0.4–0.46% one-step error | 0.3995%–0.4635% | arithmetic on #37 RMSE | reproduced (arithmetic) |
| 40 | Calibration model 512×3, 12,000 steps: RMSE 9.4×10⁻⁴; T95 = 8 | 9.44e-4; 8 | `results/tent_v1/calibration.json` row 4 | reproduced |
| 41 | Table 1 values | as rows #11–#17, #23–#26, #29–#30, #37 | same as above | reproduced |
| 42 | Arun et al. 2026: title, authors, venue, pages, DOI | Arun, Abraham, Lekshmi P; Insights 2026; pp. 1–6; DOI 10.18653/2026.insights-1.1 | ACL Anthology `2026.insights-1.1` (accessed 2026-11-13; PDF: `arun-2026-insights-1.1.pdf`) | verified external |
| 43 | Arun et al.: pairwise CKA ≈ 0.0007; each subspace F1 > 0.986; removing penalty costs 0.013 F1 | 0.0007; 0.986; 0.013 | same as #42 (abstract; PDF pp. 1–6) | verified external |
| 44 | Locatello et al. 2019, arXiv:1811.12359 | 2019; 1811.12359 | arXiv record | verified external |
| 45 | Athalye, Carlini, Wagner, ICML 2018 | 2018 | PMLR 80:274–283 | verified external |
| 46 | Barak et al. 2001; Garg et al. 2013; Adi et al. 2018; Tramèr et al. 2016; Hinton et al. 2015 | years | standard bibliographic records | verified external |

## Numbers deliberately not used

These appear in the repository or in earlier summaries, but no artifact supports them at the precision or generality implied. They are **not** in the note.

| Claim | Why excluded | Status |
|---|---|---|
| "Effective key space is ~10⁶ trials" | `docs/negative-result.md` states it, but no artifact derives it from the Hamming scan. The scan sampled 3 keys at each of 65 distances in one seed; a hit rate cannot be converted to a 10⁶ search estimate without assumptions about the island's size. | unsupported |
| "128 tent steps require ~1e-36 precision" | `VERDICT_TENT.md` and `HANDOFF.md` state it. It follows from the map slope (0.05 / 1.91^128), not from the measured bit-divergence rate; the two quantities are not reconciled in the repo. | unsupported (derived, not measured) |
| "B/C ratio at matched n" | The pre-registered ratio compares B's minimum fine-tune n with C's cheapest protocol at a different n. The note states the different sample budget explicitly and does not call it matched. | in-repo (definition), not matched |
| Any width-scaling claim | The ladder stopped at 256; 512 and 1024 were not run. | unsupported |
| Any claim that the k=4 or k=5 rule was learned | Calibration only; k=4 0.8849 < 0.99, k=5 ≈ 0.50. | unsupported |
| Any claim that the tent experiment supports a compute-scaling law | The calibration changes width, depth, and steps together. | unsupported |

## References (full)

1. Arun, G., Abraham, A., & Lekshmi P, G. (2026). *Geometric Separation Without Functional Specialization: Auditing Subspace Disentanglement in Dual-Encoder LLM Paraphrase Detection.* In Proceedings of the Seventh Workshop on Insights from Negative Results in NLP (pp. 1–6), Budapest, Hungary. ACL. DOI: 10.18653/2026.insights-1.1. https://aclanthology.org/2026.insights-1.1/
2. Athalye, A., Carlini, N., & Wagner, D. (2018). *Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples.* ICML 2018, PMLR 80:274–283.
3. Barak, B., Goldreich, O., Impagliazzo, R., Rudich, S., Sahai, A., Vadhan, S., & Yang, K. (2001). *On the (Im)possibility of Obfuscating Programs.* CRYPTO 2001, LNCS 2139, pp. 1–18.
4. Garg, S., Gentry, C., Halevi, S., Raykova, M., Sahai, A., & Waters, B. (2013). *Candidate Indistinguishability Obfuscation and Functional Encryption for All Circuits.* FOCS 2013.
5. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
6. Locatello, F., Bauer, S., Lucic, M., Raetsch, G., Gelly, S., Schölkopf, B., & Bachem, O. (2019). *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML 2019; arXiv:1811.12359.
7. Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T. (2016). *Stealing Machine Learning Models via Prediction APIs.* USENIX Security 2016.
8. Adi, Y., Baum, C., Cisse, M., Pinkas, B., & Keshet, J. (2018). *Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring.* USENIX Security 2018.

## Re-check

From the repository root, with the verification script placed one level above it or anywhere on `PYTHONPATH`:

```
python verify_claims.py /path/to/neural-drm-phase0
```

Expected lines (exact values may differ in the last decimal place if the JSON is edited):

```
A.test_r2_range: [0.996273, 0.996055, 0.99658]
B2.accepted: [0, 1, 2]
finetune.seed0.rows: [(100, 0, False, 93.105), (1000, 0, False, 192.209), (10000, 5, True, 294.543)]
keyrecovery.seed0: {'success': False, 'restarts': 20, 'best_hamming': 29, 'max_hamming': 36}
hamming.seed0.probes_above_0.89: 5
hamming.seed0.best_wrong_key: {'r2': 0.9822, 'distance': 53, 'acc': 0.9755}
C.cheapest_seconds: 515.61
B_over_C: 0.571252
expansive.k3_n3000: {'k': 3, 'n': 3000, 'acc': 0.998747, 't95': 87.0, 't95_min': 3, 't95_max': 129, ...}
tent.verdict: {'label': 'INCONCLUSIVE', 'slope': 1.6282232830282249e-16, 'r2': 0.0}
total.all_hours: 2.32189
precheck.corr_c8_c4: 0.997823
```
