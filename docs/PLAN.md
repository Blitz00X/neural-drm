# Implementation plan
User TASK.md is the design authority. Execute inline without permission pauses.
1. Freeze hypotheses and decisions, initial commit before measurements.
2. Write tests for deterministic bounded world, disjoint pools, exact gate,
   near-key rejection labels, fidelity threshold and training FLOPs; implement and run.
3. Implement four-layer GELU f/g/A/C, concat B1, fixed-projection FiLM B2,
   deterministic training/checkpoint/acceptance and append-only run records.
4. Implement separate attack modules: mean ablation, fixed-key fine-tuning,
   continuous key optimization, SAE/probe, and distillation; gate on acceptance.
5. Run pilot A then C/B conditional on H1. Preserve failure and stop if invalid.
6. Generate honest figures (unmeasured panels explicitly labeled), Turkish report,
   acceptance/cost/hypothesis tables, environment lock, tests and reproduction check.
7. Commit meaningful milestones and archive full git history, source and results.
