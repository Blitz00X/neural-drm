# Hypothesis — smooth tent map: sample vs compute limit

Frozen before the confirmatory sweep: 2026-10-22.

H1: For a keyed 16-bit tent map (a in [1.9,2.0)), a 1D scalar-regression MLP
student's one-step RMSE decreases with n and its autonomous T95 (state error
>5%) grows with ln n (>=0.5 steps per e-fold), implying exponential sample
complexity for a fixed T.

H0: The one-step RMSE plateaus at the model/optimisation floor, T95 is flat in
n, and only more compute/capacity lowers the floor; then the sample-size lever
is ineffective and the exponential-sample claim is not supported.

Calibration (pre-freeze): n=10k/30k/100k, 256 hidden, 4k/6k steps -> RMSE
4.3e-3/2.9e-3/3.4e-3; T95_state 5/6/6. A 512x3 model with 12k steps -> RMSE
9.4e-4, T95=8. Error amplification follows the Lyapunov prediction; the lever
is compute, not sample count.

Confirmatory sweep: n in {10k,30k,100k}, 3 seeds, 4k steps, 256 hidden, T=128.
