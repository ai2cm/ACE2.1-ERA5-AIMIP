#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# ACE2.2-ERA5 AIMIP inference: 15 jobs = 3 experiments x 5 realizations.
#
# ACE2.2 is stochastic, so all five realizations share ONE initial condition and
# differ only by `seed`. (ACE2.1 needed lagged ICs because it was deterministic.)
# Do not use n_ensemble_per_ic: it would put all members in one job's sample
# dimension, breaking the one-realization-per-job assumption in the realization
# label rewrite below, simulations-ace2.2-6h.yaml and the postprocessing.
#
# Jobs are --preemptible: inference is not resumable, but each run is short, so a
# preemption costs only a from-scratch redo while preemptible slots avoid
# serializing the 15 jobs behind quota.

# Knobs the smoke wrapper overrides; defaults are the production 15-job sweep.
JOB_NAME_BASE="${JOB_NAME_BASE:-ace22-era5-6h-aimip-inference-oct-1978-2024}"
JOB_GROUP="${JOB_GROUP:-ace22-era5-6h-aimip}"
EXPERIMENTS="${EXPERIMENTS:-control p2k p4k}"
MEMBERS="${MEMBERS:-1 2 3 4 5}"
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-}"
PREEMPTION_FLAG="${PREEMPTION_FLAG:---preemptible}"

# Stage-3 pressure-level FT result dataset. The stage-2 checkpoint cannot serve here:
# these configs request pressure-level outputs only the fine-tuned decoder produces.
# Stopped by hand at epoch 35 rather than at the early-stopping criterion: inference
# error had no trend across 17 evaluations (0.0309-0.0335), so later "improvements"
# would select noise. best_inference_ckpt.tar is the epoch-32 checkpoint, 0.030889.
PLEV_FT_RESULTS_DATASET="01M0WVHBW4G5H2M2NZ25REP8G4"  # stage-3 plev FT (wandb lmvpfmrp)
if [ -z "$PLEV_FT_RESULTS_DATASET" ]; then
    echo "ERROR: set PLEV_FT_RESULTS_DATASET to the stage-3 plev FT result dataset ID" >&2
    exit 1
fi

# ai2cm/ace commit the checkpoint was trained at (exp/2026-08-12-aimip-1deg-6hourly).
ACE_GIT_REF="fa856b459dc6c25b4d13b8e927d258aa7cefe543"
OUTPUT_ROOT="${OUTPUT_ROOT:-/climate-default/2026-08-25-ace22-era5-6h-aimip-inference-results}"
IC_PATH="/climate-default/2026-08-24-aimip-evaluation/aimip-evaluation-ics/1978-09-30_IC0.nc"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")

BASE=$REPO_ROOT/configs/ace2.2-era5/ace-aimip-inference-config.yaml
BASE_P2K=$REPO_ROOT/configs/ace2.2-era5/ace-aimip-inference-p2k-config.yaml
BASE_P4K=$REPO_ROOT/configs/ace2.2-era5/ace-aimip-inference-p4k-config.yaml

launch_job () {

    local JOB_NAME=$1
    local TEMPLATE_CONFIG=$2
    local MEMBER=$3
    local OVERRIDE=$4

    # Rewrite the realization label in the output filenames, then encode for
    # transfer: gantry clones the remote repo, so local paths don't exist there.
    local CONFIG_B64
    CONFIG_B64=$(sed "s/_r[0-9]i/_r${MEMBER}i/g" "$TEMPLATE_CONFIG" | base64 | tr -d '\n')

    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref $ACE_GIT_REF \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2.2-ERA5 (6-hourly) AIMIP inference' \
        --beaker-image "$(cat $REPO_ROOT/configs/ace2.2-era5/deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        $PREEMPTION_FLAG \
        --cluster ai2/titan-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --env WANDB_USERNAME=$WANDB_IDENTITY \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $PLEV_FT_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/member-config.yaml && python -I -m fme.ace.inference /tmp/member-config.yaml --override ${OVERRIDE}"

}

# 46 years (1979-2024) with spinup from 1978-10-01, control + two SST perturbations.
for EXPERIMENT in $EXPERIMENTS; do
    case $EXPERIMENT in
        control) TEMPLATE_CONFIG=$BASE;     SUFFIX="" ;;
        p2k)     TEMPLATE_CONFIG=$BASE_P2K; SUFFIX="-p2k" ;;
        p4k)     TEMPLATE_CONFIG=$BASE_P4K; SUFFIX="-p4k" ;;
    esac
    for MEMBER in $MEMBERS; do
        JOB_NAME="${JOB_NAME_BASE}${SUFFIX}-r${MEMBER}"
        OUTPUT_PATH="${OUTPUT_ROOT}/${JOB_NAME}"
        OVERRIDE="initial_condition.path=${IC_PATH} experiment_dir=${OUTPUT_PATH} seed=${MEMBER} ${EXTRA_OVERRIDE}"
        echo "Launching $JOB_NAME with override: $OVERRIDE"
        launch_job "$JOB_NAME" "$TEMPLATE_CONFIG" "$MEMBER" "$OVERRIDE"
    done
done
