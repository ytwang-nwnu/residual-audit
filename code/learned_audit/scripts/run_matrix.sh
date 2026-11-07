#!/bin/bash
# Launch the round7 run matrix sequentially on one GPU.
# usage: [V0=<n>] [SPLITS_OVERRIDE="s1 s2"] run_matrix.sh <seeds> <method...>
#   e.g. run_matrix.sh "0 1 2" m1 m3p
#        V0=200 run_matrix.sh "0 1 2" m4     # amend1 methods m4 (V0=200) / m5 (V0=300)
#        V0=27 run_matrix.sh "0 1 2" m1      # version numbers start at v28
#        SPLITS_OVERRIDE="s3 s4" V0=400 run_matrix.sh "0" m5   # PREREG_R8 splits
# V0: version offset so a re-run never reuses an earlier vNN (default 0).
# Existing run dirs are never overwritten: the script aborts if one exists.
set -u
CODE=/data1/users/yangrui/projects/idea67b/experiment/code/prl_v2/round7
RES=/data1/users/yangrui/results/idea67b/exp/prl_v2/round7
DATE_TAG=$(date +%F)
PY_m1=/data1/users/yangrui/envs/zhao-ec3d/bin/python
PY_m2=/data1/users/yangrui/envs/stgcn-rehab/bin/python
PY_m3p=/data1/users/yangrui/envs/idea67b/bin/python
PY_m4=$CODE/scripts/deeprehabpile_python.sh   # PREREG_R7_amend1: env deeprehabpile + pip CUDA libs on LD_LIBRARY_PATH
PY_m5=$CODE/scripts/deeprehabpile_python.sh

SEEDS="$1"; shift
V=${V0:-0}
if [ -n "$(cd "$CODE" && git status --porcelain)" ]; then
  echo "refusing to run: dirty working tree in $CODE"; exit 1
fi
for METHOD in "$@"; do
  case $METHOD in
    m1) SPLITS="s0 s1 s2"; PY=$PY_m1 ;;
    m2) SPLITS="s1 s2";     PY=$PY_m2 ;;
    m3p) SPLITS="s1 s2";    PY=$PY_m3p ;;
    m4) SPLITS="s1 s2";     PY=$PY_m4 ;;
    m5) SPLITS="s1 s2";     PY=$PY_m5 ;;
    *) echo "unknown method $METHOD"; exit 1 ;;
  esac
  SPLITS=${SPLITS_OVERRIDE:-$SPLITS}
  for SPLIT in $SPLITS; do
    if [ "$SPLIT" = "s0" ]; then FOLDS="3"; else FOLDS="0 1 2 3"; fi; FOLDS=${FOLDS_OVERRIDE:-$FOLDS}
    for FOLD in $FOLDS; do
      for SEED in $SEEDS; do
        V=$((V+1))
        RUN=$(printf "%s_v%02d_%s_%s_fold%d_seed%d" "$DATE_TAG" "$V" "$METHOD" "$SPLIT" "$FOLD" "$SEED")
        DIR=$RES/runs/$RUN
        if [ -e "$DIR" ]; then
          echo "refusing to overwrite existing run dir $DIR"; exit 1
        fi
        mkdir -p "$DIR"
        echo "=== $RUN ==="
        (cd "$CODE" && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
          "$PY" scripts/run_split.py --config "configs/$METHOD.yaml" \
          --split "$SPLIT" --fold "$FOLD" --seed "$SEED" --run-dir "$DIR") \
          2>&1 | tee "$DIR/log.txt"
      done
    done
  done
done
