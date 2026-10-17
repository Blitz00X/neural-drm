# Independent review and closeout

A separate read-only reviewer inspected the source and scientific protocol.
No Critical finite-path issue was reported. Three Important findings were fixed:

1. Nonfinite metrics could break strict JSON failure logging. Shared atomic
   save_json now records null and original field locations; regression test
   test_nonfinite_results_remain_durable covers both runner and attacks.
2. Acceptance was not persisted before long attacks. Training metrics/checkpoint
   are now persisted before entry, attack_status is separate, and each attack
   ledger is written before work. test_interrupted_attack_ledger_is_durable
   verifies KeyboardInterrupt evidence survives.
3. Scientific test_size was not enforced. validate_config now rejects <100000;
   test_scientific_small_test_is_rejected covers the boundary.

All 15 tests pass. A fresh original 2000-step pilot reproduces all three test
metrics exactly. Hypotheses match initial commit 849ba73 byte-for-byte.
Six PNGs were opened/verified; sample and acceptance plots inspected visually.
The width64,c4 five-seed cell completed: A/C accepted 5/5, B1/B2 accepted 0/5.
The original remaining seven cells are NOT executed. No scientific B attack
outcome, H2/H3 decision, width scaling, or interpretation-cost correlation exists.
The attack implementations passed small end-to-end smoke tests only.
