# Goal log

- 2026-10-21T19:10:42+03:00 created: keyed expansive distillation experiment; no training yet.
- 2026-10-21T20:37:18 exact-map-and-tests done: CA step/rule/λ tests pass.
- 2026-10-21T20:37:18 student-pipeline done: CNN tiny train + rollout + multistep tests pass.
- 2026-10-21T21:04:23 pilot-one-step done: calibration table recorded (k3 full, k4 0.885, k5 chance).
- 2026-10-21T21:04:23 sample-size-sweep doing: k=3 n=3k/10k/30k, 3 seeds, T=128.
- 2026-10-21T21:04:23 multi-step-student failed: not run at scale because the k=4 one-step precondition failed.
- 2026-10-21T21:27:25 sample-size-sweep done: 9/9 k=3 runs; n=3000 split, n=10k/30k full horizon.
- 2026-10-21T21:27:25 analyze-and-verdict done: VERDICT_EXPANSIVE = POLYNOMIAL (threshold, not exponential).
- 2026-10-21T21:27:25 audit done: tests/reproduce/runs/freeze checks pass; goal paused because multi-step at scale was not run.
