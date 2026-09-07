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
# Jobs carry --min-runtime 8h. Last month's 15 ran 75-320 min each (median ~2 h) and,
# under the since-deprecated --preemptible, 8 of them were killed at least once -- one
# nine times -- discarding 57% of the GPU-hours spent. Inference is not resumable, so
# the guarantee has to cover the slowest observed run, and a longer request costs
# nothing under the current scheduler.

# Knobs the smoke wrapper overrides; defaults are the production 15-job sweep.
JOB_NAME_BASE="${JOB_NAME_BASE:-ace22-era5-6h-rs3-aimip-inference-oct-1978-2024}"
JOB_GROUP="${JOB_GROUP:-ace22-era5-6h-rs3-aimip}"
EXPERIMENTS="${EXPERIMENTS:-control p2k p4k}"
MEMBERS="${MEMBERS:-1 2 3 4 5}"
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-}"
MIN_RUNTIME="${MIN_RUNTIME:-8h}"

# Stage-3 pressure-level FT result dataset. The stage-2 checkpoint cannot serve here:
# these configs request pressure-level outputs only the fine-tuned decoder produces.
# Training seed 3 of the four-seed ACE2.2 ensemble, chosen on the 36-year in-sample
# rollouts (first on 11 of 12 basis x metric combinations; see the findings report in
# explore2/brianh/2026-08-26-ace22-variants-findings). best_inference_ckpt.tar is the
# epoch-42 checkpoint of 50, best_inference_error 0.036006. Supersedes seed 0's
# 01M0WVHBW4G5H2M2NZ25REP8G4 (epoch 32, wandb lmvpfmrp), evaluated as v20260825.
PLEV_FT_RESULTS_DATASET="01M1WBBJDFYA9VS46V6HH3K1MW"  # stage-3 plev FT, seed 3 (wandb dd0f9jnb)
if [ -z "$PLEV_FT_RESULTS_DATASET" ]; then
    echo "ERROR: set PLEV_FT_RESULTS_DATASET to the stage-3 plev FT result dataset ID" >&2
    exit 1
fi

# ai2cm/ace commit the checkpoint was trained at (exp/2026-08-28-ace22-variants). Its
# fme/ tree is byte-identical to fa856b459, the ref the seed-0 runs used.
ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"
OUTPUT_ROOT="${OUTPUT_ROOT:-/climate-default/2026-09-07-ace22-era5-6h-rs3-aimip-inference-results}"
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
        --min-runtime "$MIN_RUNTIME" \
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
