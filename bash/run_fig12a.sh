#!/bin/bash

# Repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Setup environment
source $REPO_ROOT/venv/bin/activate

# Run python script
LOGDIR=$REPO_ROOT/log_dir
JSON=$REPO_ROOT/json/fig12a.json
[ ! -d ${LOGDIR} ] && mkdir -p ${LOGDIR}
rm -f ${LOGDIR}/err_kfolds.log ${LOGDIR}/out_kfolds.log


python $REPO_ROOT/mains/run_pbdw.py --json="$JSON" \
	   1>>${LOGDIR}/out_kfolds.log 2>>${LOGDIR}/err_kfolds.log

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

# ===============================
# Draw figure
# ===============================

python $REPO_ROOT/mains/plot_kfolds.py \
       --num_folds $NUM_FOLDS \
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
       --SNR_value $SNR_VALUE \
       --xi $XI
