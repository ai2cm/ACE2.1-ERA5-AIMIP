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
#
# Every job runs with latent global-mean clipping off. The clip acts only at inference,
# so turning it off changes no weights: scripts/clip_off.py copies the mounted checkpoint
# with the clip's envelope set to -inf/+inf, and inference reads that copy.

# Knobs the smoke wrapper overrides; defaults are the production 15-job sweep.
JOB_NAME_BASE="${JOB_NAME_BASE:-ace22-no-gmr-diagnostic-near-surface-rs2-aimip-inference-oct-1978-2024}"
JOB_GROUP="${JOB_GROUP:-ace22-no-gmr-diagnostic-near-surface-rs2-aimip}"
EXPERIMENTS="${EXPERIMENTS:-control p2k p4k}"
MEMBERS="${MEMBERS:-1 2 3 4 5}"
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-}"
MIN_RUNTIME="${MIN_RUNTIME:-8h}"

# Stage-3 pressure-level FT result dataset. The stage-2 checkpoint cannot serve here:
# these configs request pressure-level outputs only the fine-tuned decoder produces.
# Training seed 2 of the four-seed ensemble without global mean removal and with
# diagnostic near-surface fields (TMP2m, Q2m, UGRD10m, VGRD10m), chosen on 1979-2014
# rollouts of the stage-2 checkpoints (see explore2/brianh/2026-11-05-ace22-1deg-near-surface-choice).
# best_inference_ckpt.tar is the epoch-12 checkpoint, best_inference_error 0.039866; the
# run was stopped at epoch 35 of 50 after 11 inline evaluations without improvement. The
# experiment was preempted once; this is the result dataset of its final job. Supersedes
# seed 3 of the earlier recipe with global mean removal and prognostic near-surface fields
# (01M1WBBJDFYA9VS46V6HH3K1MW, wandb dd0f9jnb).
PLEV_FT_RESULTS_DATASET="01M6PN2NBP00X8XTV4DE6QERBZ"  # stage-3 plev FT, seed 2 (wandb kdx8qepi)
if [ -z "$PLEV_FT_RESULTS_DATASET" ]; then
    echo "ERROR: set PLEV_FT_RESULTS_DATASET to the stage-3 plev FT result dataset ID" >&2
    exit 1
fi

# ai2cm/ace commit the checkpoint was trained at, the same as for the earlier recipe.
ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"
OUTPUT_ROOT="${OUTPUT_ROOT:-/climate-default/2026-11-06-ace22-no-gmr-diagnostic-near-surface-rs2-aimip-inference-results}"
IC_PATH="/climate-default/2026-08-24-aimip-evaluation/aimip-evaluation-ics/1978-09-30_IC0.nc"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")

BASE=$REPO_ROOT/configs/ace2.2-era5/ace-aimip-inference-config.yaml
BASE_P2K=$REPO_ROOT/configs/ace2.2-era5/ace-aimip-inference-p2k-config.yaml
BASE_P4K=$REPO_ROOT/configs/ace2.2-era5/ace-aimip-inference-p4k-config.yaml
CLIP_OFF_B64=$(base64 < "$SCRIPT_DIR/clip_off.py" | tr -d '\n')

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
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/member-config.yaml && \
                    echo '${CLIP_OFF_B64}' | base64 -d > /tmp/clip_off.py && \
                    python -I /tmp/clip_off.py /ckpt.tar /tmp/ckpt.tar && \
                    python -I -m fme.ace.inference /tmp/member-config.yaml \
                    --override checkpoint_path=/tmp/ckpt.tar ${OVERRIDE}"

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
