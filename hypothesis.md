# Hypothesis — expansive chained function as a distillation barrier

Frozen before the confirmatory sweep: 2026-10-21.

## H1 (mechanism)

For a keyed 8x8 binary cellular automaton with a k=4 rule (65,536-bit table),
a CNN student trained on one-step (grid_t, grid_{t+1}) pairs reaches >=0.99
cell accuracy at n<=100,000. Then its autonomous trajectory survival T95 grows
with ln n (>=0.5 steps per e-fold) and a free-running multi-step student does
not reach T=128.

## H0 (kill)

The one-step precondition fails at the attacker's budget (the rule is not
learnable), or T95 saturates at the full horizon once the rule is learned, or a
multi-step student bypasses it. In each case the exponential-in-T claim is not
supported; the observed barrier is either learnability (description length) or
a one-time learning threshold, not error amplification.

## Pre-freeze calibration (not confirmatory evidence)

| k | rule bits | n | steps | channels | one-step acc | T95 |
|---|---|---:|---:|---:|---:|---:|
| 3 | 512 | 10k | 2000 | 32 | 0.996 | 4 |
| 3 | 512 | 10k | 5000 | 32 | 1.000 | 65/64 (full) |
| 4 | 65,536 | 30k | 1200 | 128 | 0.538 | 1 |
| 4 | 65,536 | 30k | 1500 | 256 | 0.555 | 1 |
| 4 | 65,536 | 100k | 4000 | 256 | 0.885 | 1 |
| 5 | 33,554,432 | 300k | 4000 | 32 | 0.500 | 1 |

Reading: at k=3 the rule is learned exactly and T is irrelevant (threshold).
At k=4 the best CPU-budget student is at 88.5% one step; the trajectory dies
immediately. The k=4 failure is a learnability/description-length barrier, not
the hypothesized exponential error amplification. k=5 is at chance.

## Confirmatory sweep

k=3, n in {3,000, 10,000, 30,000}, seeds 0/1/2, 2,000 steps, 64 channels,
T=128. If the student learns the rule, T95 should reach the full horizon at all
n above the threshold, independent of T. That is a POLYNOMIAL/threshold result.
