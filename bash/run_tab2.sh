#!/bin/bash

# Repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Setup environment
source $REPO_ROOT/venv/bin/activate

# Prepare log files
LOGDIR=$REPO_ROOT/log_dir/
JSON=$REPO_ROOT/json/tab2.json
[ ! -d ${LOGDIR} ] && mkdir -p ${LOGDIR}
rm -f ${LOGDIR}/err_SS.log ${LOGDIR}/out_SS.log

JSON=$REPO_ROOT/json/tab2.json

HSIZE=(1 10 20 50 100 200 500 1000)

# ===============================
# Run PBDW for each H
# ===============================

for DIM in "${HSIZE[@]}"; do
    sed -i \
        "s/\"H\"[[:space:]]*:[[:space:]]*[0-9]\+/\"H\": ${DIM}/" \
        "$JSON"

    python $REPO_ROOT/mains/run_pbdw.py --json="$JSON"
done

# ===============================
# Extract parameters from JSON
# ===============================

DATA_BASEPATH=$(jq -r '.general.path_output_folder' $JSON)
SS_ALG=$(jq -r '.general.SSs[0]' $JSON)
NUM_SS_COMPONENTS=$(jq -r '.SS.wOMP.sensor_dim' $JSON)
OUT=$DATA_BASEPATH/table2.txt

# These are NOT in JSON — must be set manually
FOLD_IDX=0
NUM_SNAPSHOTS=100

# ===============================
# Build table
# ===============================

python $REPO_ROOT/mains/make_tab.py \
    --fold_idx $FOLD_IDX \
    --num_snapshots $NUM_SNAPSHOTS \
    --data_basepath "$DATA_BASEPATH" \
    --ss_alg "$SS_ALG" \
    --H_list "${HSIZE[@]}" \
    --num_ss_components $NUM_SS_COMPONENTS \
    --out "$OUT"
