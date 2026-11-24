# Blood-only detection and validation: results (run 20 Nov 2026, 3 h 23 min on the Mac, exit 0)

Script: blood_validation.py (pipeline classes, env hpc settings: B = 1,000, PBI 0.95, thresholds 25/5/5; BH across the blood
candidates). Outputs: blood_run/. Comparison: compare.py -> blood_vs_pooled_validation.csv.

- Blood culture episodes 9,205; episode-pair rows 6,773,445; co-testing screen (within blood) removed 34 pairs; 632 candidates.
- Blood labels: robust 138, supported 50 (validated 188), mixed 384, insufficient 60.
- Of the 188 blood-validated pairs: 151 (80.3%) also pooled-validated in the same direction; 9 pooled-validated in the opposite
  direction; 28 pooled candidates that were not pooled-validated; 0 pairs that were not pooled candidates.
- Of the 435 pooled-validated pairs: 321 were candidates in blood (the rest too sparse in blood, e.g. ceftriaxone -> fosfomycin:
  fosfomycin is rarely tested in blood); 151 validated in blood in the same direction, 9 in the opposite direction, 161 not
  validated in blood (118 of them with the blood ER on the same side of 1).
- Pooled >= 1.5-fold (120): 63 candidates in blood, 31 validated there in the same direction.
- The one pair hidden by the pooled screen, moxifloxacin -> ceftolozane-tazobactam: a blood candidate (blood ER 0.65) but not
  validated in blood (mixed).
- Opposite directions concern mostly tetracycline and the beta-lactamase-inhibitor/cephalosporin downstream agents, e.g.
  cefepime -> tetracycline: pooled 0.63 (robust), blood 1.73 (robust); aztreonam -> tetracycline 0.67 vs 1.50.
- Examples kept in both: gentamicin -> amikacin 2.16 pooled, 1.90 blood (robust in both); aztreonam -> piperacillin-tazobactam
  0.66 pooled, 0.76 blood (robust in both).
