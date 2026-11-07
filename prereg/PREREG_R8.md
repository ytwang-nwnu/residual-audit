# PREREG_R8 — calibration ablations for the learned-detector audit (2026-11-05)

Written before any S3/S4 run is launched. Raised by the simulated TCSS review of the
manuscript (gaojianfanyi/模拟审稿-TCSS-20261104.md, items #6 and #12). Original PREREG_R7
and amend1 (H16–H23), their methods, splits, seeds and run_selection.json are frozen and
untouched; this round adds two new splits and reads the frozen s1/s2 runs only as baselines.

## Why

- Item #6: in S2 ("calibrated"), the TRAINING persons are also centred on their own
  correct mean and the model is retrained. The manuscript attributes the S1→S2 gain to
  removing the held-out learner's individual term alone; that attribution is not closed
  without a condition that calibrates the held-out learner only.
- Item #12: personal calibration in the paper requires teacher-confirmed CORRECT reps.
  A reviewer will ask whether subtracting the learner's mean over ALL reps (no labels)
  achieves the same, which would remove the labelling cost.

## Splits (code: round7/src/ec3d.py prepare_fold; selfcheck_r8.py passed, s1/s2 byte-identical)

- **S3 test-only calibration.** Training set identical to S1 (raw). Held-out learner
  calibrated exactly as in S2 (own CORRECT-rep mean, LOO for correct items; k ∈ {1,2,4}
  drawn correct reps × 20 draws, same draw sequence as S2 under the same seed). The same
  trained model also re-scores the raw held-out set ("s3_raw") as a within-run S1 control.
- **S4 label-free calibration.** Training persons centred on their ALL-rep mean (+ training
  ALL-rep grand mean), retrained. Held-out learner centred on the mean of ALL own reps
  (LOO for the item itself) or on k drawn reps of ANY label × 20 draws.

Methods and seeds: m1 Zhao'22 (seeds 0 1 2), m3p Zheng'23 (seeds 0 1 2), m5 ConvTran
(seed 0), m4 LITEMV (seed 0). m2 Deb'22 is **deferred**: measured 575 min per unit ⇒ 77 GPU-h
for the two splits, above the amend1 budget cap; if run later it will be added as a separate
amendment, not silently merged. GPU 0 of 314-v100 (shared host; other user's jobs present).
Version offsets V0=500 (m1), 520 (m3p), 540 (m5), 560 (m4). Existing run dirs never overwritten.

## Metrics (same definitions as R7: pooled AUC over 4 held-out learners, per-action)

- ΔAUC_S3 = AUC(s3_all) − AUC(s3_raw); ΔAUC_S2 = AUC(s2_all) − AUC(s1) from frozen runs.
- ΔAUC_S4 = AUC(s4_all) − AUC(s1).
- Recovery at k: R_k = (AUC(sX_k) − AUC(raw)) / (AUC(sX_all) − AUC(raw)) averaged over draws,
  on cells where ΔAUC_S2 ≥ 0.05 (the R7 failing cells).

## Predictions (verdict rules fixed now)

- **H24 (attribution).** On every R7 failing cell, ΔAUC_S3 ≥ 0.5 · ΔAUC_S2. Supported if it
  holds on all failing cells of the four methods; partial if it holds on ≥ half; else not.
  Rationale: if the gain comes mainly from removing the held-out learner's individual term,
  test-only calibration should recover at least half of it without retraining.
- **H25 (training-centring contribution).** ΔAUC_S2 − ΔAUC_S3 ≥ 0.02 on at least one failing
  cell per method (i.e. training-set centring contributes a measurable, non-zero share).
  Descriptive if neither direction is clear; this is reported either way.
- **H26 (label-free).** On every R7 failing cell, ΔAUC_S4 ≥ 0.5 · ΔAUC_S2. Supported / partial
  / not as in H24. Rationale: error reps are a minority of the mean's inputs in most cells,
  so the all-rep mean still removes most of s_i.
- **H27 (label-free costs separability where errors dominate).** For cells where incorrect
  reps outnumber correct ones by ≥ 2:1 (EC3D squat, lunge, plank all qualify, 100/41, 81/46,
  70/33), ΔAUC_S4 < ΔAUC_S2 on at least 2 of 3 actions for each method. Descriptive otherwise.

## Reporting

- New summary script (make_summary_r8.py) writes exp/prl_v2/round7/summary_r8/*.csv and
  prereg_check_r8.json; it does not modify summary/ or run_selection.json.
- Manuscript: one added column pair in Table S10 (test-only, label-free) and one sentence
  each in Section VI RQ6 and Section S10. No change to the frozen H16–H23 verdicts.

md5 of this file is stamped in R8_stamp.json before the first run.
