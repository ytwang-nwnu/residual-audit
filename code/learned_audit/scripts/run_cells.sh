#!/bin/bash
# Run an explicit list of round7 cells sequentially (same conventions as
# run_matrix.sh: clean tree required, run dir = YYYY-MM-DD_vNN_<method>_<split>_fold<h>_seed<s>,
# never overwrite an existing run dir, log.txt via tee).
# usage: run_cells.sh <method>:<split>:<fold>:<seed>:<version> [...]
#   e.g. CUDA_VISIBLE_DEVICES=0 DRP_GPU_MEM_LIMIT_MIB=7000 run_cells.sh m4:s1:0:0:201 m4:s1:1:0:202
# Used for PREREG_R7_amend1 (m4/m5) to spread the cells of one matrix over
# both GPUs / several processes with a per-process memory cap; the version
# numbers follow run_matrix.sh's scheme (V0=200 m4, V0=300 m5; +20 for s2).
set -u
CODE=/data1/users/yangrui/projects/idea67b/experiment/code/prl_v2/round7
RES=/data1/users/yangrui/results/idea67b/exp/prl_v2/round7
DATE_TAG=$(date +%F)
PY_m1=/data1/users/yangrui/envs/zhao-ec3d/bin/python
PY_m2=/data1/users/yangrui/envs/stgcn-rehab/bin/python
PY_m3p=/data1/users/yangrui/envs/idea67b/bin/python
PY_m4=$CODE/scripts/deeprehabpile_python.sh
PY_m5=$CODE/scripts/deeprehabpile_python.sh

if [ -n "$(cd "$CODE" && git status --porcelain)" ]; then
  echo "refusing to run: dirty working tree in $CODE"; exit 1
fi
for CELL in "$@"; do
  IFS=: read -r METHOD SPLIT FOLD SEED V <<< "$CELL"
  case $METHOD in
    m1) PY=$PY_m1 ;; m2) PY=$PY_m2 ;; m3p) PY=$PY_m3p ;; m4) PY=$PY_m4 ;; m5) PY=$PY_m5 ;;
    *) echo "unknown method $METHOD"; exit 1 ;;
  esac
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
