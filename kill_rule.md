# Kill rule — frozen before the confirmatory sweep

Frozen: 2026-10-21. Do not edit after the k=3 sweep starts.

1. **Precondition.** The main k=4 student at n_max=100,000, 4,000 steps,
   256 channels must reach >=0.99 one-step cell accuracy. Calibration reached
   0.885; therefore the exponential-in-T mechanism is **not isolated**. The
   observed barrier is learnability, not error amplification.
2. **Control threshold.** In the confirmatory k=3 sweep, if one-step accuracy
   >=0.99 and T95 >= T_max (129) at any n, the rule is learned exactly and T is
   irrelevant: label **POLYNOMIAL** (a one-time threshold, not exponential in T).
3. **If the k=4 precondition were met** (it was not), fit T95 vs ln n on
   non-saturated points: slope >=0.5 and R^2 >=0.6 would be **EXPONENTIAL**;
   otherwise **POLYNOMIAL**.
4. **Multi-step bypass.** If a free-running 8-step student reached T95>=T_max
   at n<=100,000, label **POLYNOMIAL**. Not run at the required scale because
   the one-step precondition failed.
5. **Missing/contradictory evidence** -> **INCONCLUSIVE**. This rule is not used
   to upgrade a failed precondition into support.

Expected outcome: **POLYNOMIAL** if the k=3 control saturates; the exponential-T
claim is not supported in this testbed.
