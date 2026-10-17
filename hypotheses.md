# Preregistered hypotheses — immutable
Registered before any training or measurement. See initial git commit.

High fidelity ALWAYS means R2_height >= 0.95 AND acc_biome >= 0.95,
on an independent test pool of at least 100,000 coordinates.
H1: Removing A's external gate gives high fidelity at zero optimization/oracle cost.
If H1 fails, stop before defense/attack inference; diagnose task fitting first.
H2: B's mean-ablation Pareto curve is materially worse than A's.
Operational rule: across all five seeds, no B intervention affecting <=20% of
units reaches high fidelity on all six unauthorized fixed keys, while A does.
Reduced sensitivity alone is not success (a constant decoy is insensitive too).
H3: Both fine-tuning AND key recovery require >=10x A and C cost and costs
increase with width. A's zero cost makes multiplicative comparison undefined;
report B-A absolute cost and B/C ratios, NEVER divide by epsilon.
Censored/unreached thresholds do not establish a 10x ratio.
Width scaling requires >=3 measured widths and five seeds each.
Acceptance: correct key matches f AND uniform-wrong, Hamming-1, Hamming-2
strata separately match g at high fidelity. Rejected models excluded from attacks.
H1/H2/H3 use PASS, FAIL, or NOT_EVALUABLE. Missing evidence is not FAIL.
Five seeds: 0,1,2,3,4. Report mean and min/max; pilot n=1 has no interval.
No post-hoc threshold changes or deletion of failed runs.

| Sonuç | Anlamı |
|---|---|
| H2 ve H3 geçiyor | Fikir çalışıyor; Faz 1'e geçilir |
| H2 geçiyor, H3 düşüyor (az örnekle açılıyor) | Statik analize karşı iyi, fine-tune'a karşı zayıf; min-max eğitim eklenmeli |
| İkisi de düşüyor | Entanglement ayrık kapıdan farksız; fikir bu formda çürüdü, nedeni raporlanır |
| Her şey C'ye çöküyor | Güvenlik tavanı fonksiyonun öğrenme karmaşıklığı; bu ölçülmüş bir sınır olarak raporlanır |
