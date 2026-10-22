# Kill rule — smooth tent map

Frozen: 2026-10-22. Do not edit after the confirmatory sweep starts.

1. **Underfit guard.** If the best n has one-step RMSE > 1e-2, label
   **INCONCLUSIVE** (the student is not in the approximation regime).
2. **Exact threshold.** If any seed reaches T95>=T_max=129, the map is learned
   exactly and T is irrelevant: **POLYNOMIAL**.
3. **Sample lever.** Fit T95 vs ln n. Slope >=0.5 and R^2 >=0.6 ->
   **EXPONENTIAL**; slope <0.5 or R^2 <0.6 -> **INCONCLUSIVE** (flat/compute-
   limited), not support.
4. **Compute check.** The calibration's larger model (512x3, 12k steps) reached
   RMSE 9.4e-4/T95=8 vs 2.9e-3/T95=6 at 256x2/6k. If the sweep confirms a flat
   T95 at fixed compute, the bottleneck is capacity/compute, not samples.
5. Missing/contradictory evidence -> **INCONCLUSIVE**. No upgrade without data.
