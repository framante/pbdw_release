#!/bin/bash

# Repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Setup environment
source $REPO_ROOT/venv/bin/activate

# Run python script
LOGDIR=$REPO_ROOT/log_dir
JSON=$REPO_ROOT/json/fig12b.json
[ ! -d ${LOGDIR} ] && mkdir -p ${LOGDIR}
rm -f ${LOGDIR}/err_xi.log ${LOGDIR}/out_xi.log

XI_VALS=(0. 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3 1e4 1e5)
SNR_VALS=(0.1 0.2 0.3 0.5 0.7)

# ===============================
# Run PBDW for each XI_VALS and SNR_VALS
# ===============================

for SNR in "${SNR_VALS[@]}"; do
    sed -i
    "s/\"SNR\"[[:space:]]*:[[:space:]]*[0-9.eE+-]\+/\"SNR\": ${SNR}/" \
	"$JSON"

    for XI in "${XI_VALS[@]}"; do
	sed -i \
	    "s/\"xi\"[[:space:]]*:[[:space:]]*[0-9.eE+-]\+/\"\xi\": ${XI}/" \
	    "$JSON"

	python $REPO_ROOT/mains/run_pbdw.py --json="$JSON" \
	       1>>${LOGDIR}/out_xi.log 2>>${LOGDIR}/err_xi.log
    done
done

# ===============================
# Extract parameters from JSON
# ===============================

NUM_FOLDS=$(jq -r '.general.num_splits' $JSON)
NUM_RANDOM_RUNS=$(jq -r '.noise.num_random_runs' $JSON)
NUM_RECON=$(jq -r '.general.fraction_test' $JSON)
DATA_BASEPATH=$(jq -r '.general.path_output_folder' $JSON)
PERMUTATION=$(jq -r '.algebra.permutation' $JSON)
RUN_TYPE=$(jq -r '.algebra.run_type' $JSON)
RB_ALG=$(jq -r '.general.RBs[0]' $JSON)
SS_ALG=$(jq -r '.general.SSs[0]' $JSON)
H=$(jq -r '.SS.wOMP.H' $JSON)
NUM_SS_COMPONENTS=$(jq -r '.SS.wOMP.sensor_dim' $JSON)
SNR_TYPE=$(jq -r '.noise.type' $JSON)
SNR_VAL=$(jq -r --arg key "$SNR_TYPE" '.noise[$key]' $JSON)
XI=$(jq -r '.noise.xi' $JSON)

# These are NOT in JSON — must be set manually
NUM_SNAPSHOTS=100
FOLD_IDX=0

# ===============================
# Draw figure
# ===============================

python $REPO_ROOT/mains/plot_xi.py \
       --fold_idx $FOLD_IDX \
       --num_random_runs $NUM_RANDOM_RUNS \
       --num_snapshots $NUM_SNAPSHOTS \
       --num_reconstructions $NUM_RECON \
       --data_basepath $DATA_BASEPATH \
       --permutation $PERMUTATION \
       --run_type $RUN_TYPE \
       --ss_alg $SS_ALG \
       --rb_alg $RB_ALG \
       --H $H \
       --num_ss_components $NUM_SS_COMPONENTS \
       --SNR_type $SNR_TYPE \
       --SNR_values $SNR_VALS \
       --xi $XI_VALS

