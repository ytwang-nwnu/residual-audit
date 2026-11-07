# PREREG_R8b — distance-scorer additions (2026-11-05)

Written before run_u1.py / run_b1.py are executed. Companion to PREREG_R8 (learned detectors).
Reads frozen round3/k1 outputs (auc_O, auc_D2, labelled recovery) as baselines; writes only to
runs/prl_v2/round8/.

## U1 — label-free personal calibration (review item #12)

Same seven applicable cells, same k ∈ {1,2,3,5}, 50 draws, seed 0, same quantile threshold
rule as K1. Difference from K1: the k reps are drawn from ALL of the held-out person's reps of
that action (correct and incorrect, no label used), and the threshold is set on the other
people with the same label-free rule. Drawn reps are excluded from that draw's test set.
Also "U_all": mean of all own reps, LOO for the item itself.

Recovery R_k = (AUC_U,k − AUC_D2) / (AUC_O − AUC_D2) with the frozen O/D2 values.

- **H28.** On RD and RGS (errors : correct ≈ 3 : 1), label-free k=1 recovery ≥ 0.5.
  Rationale: a single rep of any label still removes most of s_i when s_i ≫ b.
- **H29.** On the five EC3D applicable cells, label-free recovery at k=2 is lower than the
  labelled K1 recovery at k=2 in at least 4 of 5 cells, and the gap is largest on the two squat
  cells (largest error share and largest b). Descriptive if not.
- **H30.** U_all recovery ≥ 0.8 on at least 4 of the 7 cells.

## B1 — uncertainty of ΔAUC (review item #13)

GAITEX: person-cluster bootstrap, B = 2000, seed 0, resampling held-out persons with replacement
and recomputing pooled AUC_O, AUC_D2 and Δ from each person's frozen score vectors (reference and
threshold per person are not re-estimated; this is a bootstrap of the evaluation, stated as such).
Percentile 95% interval. EC3D (4 persons): no bootstrap; leave-one-person-out jackknife range of
Δ over the 4 persons, per cell.

- **H31.** The 95% interval of ΔAUC excludes 0 for both RD and RGS.
- **H32.** The RD interval lies entirely above the RGS interval (order by ρ holds under resampling).
  Partial if they overlap but RD's point estimate stays above RGS's upper bound.
- **H33.** EC3D jackknife: the sign of Δ is positive for every left-out person on the five
  applicable cells; the ρ order of the six cells is preserved under every left-out person
  (descriptive count reported if not).
