# Residual Audit for Follow-Along Motion Feedback

Code, preregistration files, and result tables for the manuscript
*Residual Audit for Follow-Along Motion Feedback: Individual Differences, Error-Detection Failures, and Personal Calibration*
(submitted to IEEE Transactions on Computational Social Systems, 2026).

The paper shows that the aligned gap between a learner and a demonstration mixes a stable
individual difference, trial-to-trial variation, and the true error, and that reference-based
feedback treats their sum as the error. The residual audit separates the three with
segment-level labels only, measures the detection loss of a demonstration reference, and the
number of the learner's own repetitions needed to recover it.

## What is here

| Folder | Content |
|---|---|
| `prereg/` | Preregistration files written before each round of experiments, with md5 stamps (`*_stamp.json`). PREREG (E1–E5), R2, R3 (K1 calibration), R7 and amend1 (learned detectors H16–H23), R8 and R8b (ablations and uncertainty H24–H33). |
| `code/distance_scorer/` | Residual decomposition, reference substitution and personal calibration for the distance scorer: `run_e1.py` (O vs D2, Table II), `round2/` (bone angles, GAITEX), `round3/run_k1.py` (minimal calibration, Fig. 5), `round8/run_u1.py` (label-free control, Table S11), `round8/run_b1.py` (bootstrap / jackknife of ΔAUC). |
| `code/learned_audit/` | Leave-one-person-out audit of five learned detectors (Section VI RQ6, Fig. 6, Table S10). Adapters for Zhao'22, Deb'22, Zheng'23, LITEMV and ConvTran, split definitions (`src/ec3d.py`: s1 LOPO, s2 calibrated, s3 test-only calibration, s4 label-free), launch and aggregation scripts. |
| `results/` | Table-level outputs that the paper's numbers are read from: pooled AUC, per-person rates, recovery ratios, preregistration verdicts, run manifests. Per-run score files are not included. |
| `figures/` | Plotting script for Fig. 6. |

## Paper table / figure → file

| Paper | File |
|---|---|
| Table II, Fig. 3 | `results/distance_scorer/pooled.csv`, `rho_realign.csv` |
| Fig. 5, Table S6 (H13, H14) | `results/distance_scorer/k1/recovery.csv`, `k1/prereg_check.json` |
| Section VI RQ2 intervals, Table S2 jackknife | `results/distance_scorer/b1/delta_ci.csv`, `b1/jackknife.csv` |
| Table S11 (label-free control) | `results/distance_scorer/u1/recovery.csv`, `u1/u_all.csv` |
| Fig. 6, Table S10 | `results/learned_audit/<method>/metrics.json`, `per_person.csv`, `recovery.csv`; `summary/prereg_check.json` (H16–H23) |
| Table S10 ablation columns | `results/learned_audit/summary_r8/s3_s4_table.csv`, `prereg_check_r8.json` (H24–H27) |

## Data

The datasets are not redistributed. Obtain them from their authors and follow their licences:
EC3D (Zhao et al., ACCV 2022), GAITEX (Spilz et al., Scientific Data), CoMPAS3D (Burkanova et al., 2025),
IntelliRehabDS (Miron et al., Data 2021). The Dunhuang classroom recordings contain identifiable
people and are not released.

Third-party detector implementations are not copied here. Commits used: see
`results/learned_audit/MANIFEST.json`. DeepRehabPile is used at commit `f7e4d81`.

## Running

Paths in the scripts point to the authors' cluster (`/data1/users/yangrui/...`) and are read from
`code/learned_audit/configs/*.yaml` or from constants at the top of each script. Set them to your
own data and output locations before running. Environments: Python 3.12 / PyTorch 2.14 for the
distance scorer and Zhao'22; TensorFlow 2.15.1 for Deb'22; TensorFlow 2.16.1 + Keras 3.6 for
LITEMV and ConvTran (see `code/learned_audit/README.md`).

Typical order: `run_e1.py` → `round2/run_r2.py`, `run_r4.py` → `round3/run_k1.py` →
`learned_audit/scripts/run_matrix.sh` → `scripts/make_summary.py` → `round8/run_u1.py`, `run_b1.py` →
`scripts/make_summary_r8.py`.

## Licence

Code: MIT. Preregistration files and result tables: CC BY 4.0. See `LICENSE`.
