# prl_v2 round7 — learning-based detector audit (PREREG_R7)

Spec: results `exp/prl_v2/round7/PREREG_R7.md`. Audit M1 Zhao'22 GCN classifier
(env `zhao-ec3d`), M2 Deb'22 ST-GCN+LSTM (env `stgcn-rehab`), M3' fallback
Kelly510/RehabExerAssess (env `idea67b`) on EC3D leave-one-person-out.

Layout: `src/` shared data + per-method adapters, `scripts/` entry points,
`configs/` per-method yaml (all paths read from config).

Run one unit:
```
<env python> scripts/run_split.py --config configs/<method>.yaml \
  --split s1 --fold 0 --seed 0 --run-dir <run dir>
```
Full matrix: `bash scripts/run_matrix.sh "0 1 2" m1 m2 m3p`
Aggregate: `python scripts/make_summary.py` (env `idea67b`)
Self-check: `python scripts/selfcheck.py` (env `idea67b`)

## PREREG_R7_amend1 — M4 LITEMV, M5 ConvTran (DeepRehabPile)

Spec: `exp/prl_v2/round7/PREREG_R7_amend1.md`. Both estimators come unmodified
from `third_party/deeprehabpile` (github.com/MSD-IRIMAS/DeepRehabPile, pip -e
into env `deeprehabpile`: tensorflow 2.16.1 / keras 3.6.0 per its README; run it
through `scripts/deeprehabpile_python.sh`, which puts the pip CUDA libs on
LD_LIBRARY_PATH because TF 2.16.1 does not preload them itself).
Adapters: `src/drp_bench.py` (shared benchmark recipe: hydra defaults 1500
epochs, batch 64, 5 initializations, lowest-training-loss checkpoint,
probability-averaging ensemble, one binary model per action, their
`_normalize_skeletons` fit on the training persons), `src/m4_litemv.py`,
`src/m5_convtran.py`; configs `configs/m4.yaml`, `configs/m5.yaml`.
Run versions: `V0=200` for m4, `V0=300` for m5, e.g.
`TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 V0=200 bash scripts/run_matrix.sh "0 1 2" m4`.
Aggregation: `make_summary.py` adds `m4/`, `m5/` and appends H20–H23 to
`summary/prereg_check.json` (H16–H19 and their method set untouched);
`make_manifest.py` records the DeepRehabPile commit, the env and the
amendment's md5/stamp.
