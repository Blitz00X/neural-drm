# Audit checklist (run after the verdict)

- [ ] `hypothesis.md` and `kill_rule.md` committed before the confirmatory sweep:
      `git log --oneline -- hypothesis.md kill_rule.md`
- [ ] Tests: `python -m unittest tests.test_expansive -v` -> 6/6 OK
- [ ] Raw runs: `ls results/expansive_v1/runs/*.json | wc -l` >= 9
- [ ] Calibration table: `results/expansive_v1/calibration.json` present and
      labelled `pre_freeze_calibration`
- [ ] Reproduce verdict: `python experiments/analyze_expansive.py` then
      `head -1 VERDICT_EXPANSIVE.md` matches `summary.json.verdict.label`
- [ ] No threshold/rule edits after the sweep: `git diff` since the freeze commit
      touches only results/figures/README/manifest
- [ ] Numbers in `VERDICT_EXPANSIVE.md` match `summary.json` groups (acc/T95)
- [ ] Limitations state: CPU only, 3 seeds, one key, T_max=128, exact-trajectory
      threat model, k=4 precondition 0.885 < 0.99
