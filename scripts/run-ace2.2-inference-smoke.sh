#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# ACE2.2-ERA5: 1-deg daily-06Z v2 ERA5-only recipe (non-residual, no CO2 input),
# core checkpoint from wandb ai2cm/ace/nobgd4ek (beaker 01KXKBKW2DCAYX6Q3FRJ60K3Q6),
# plus a fine-tuned pressure-level secondary decoder (see
# scripts/run-ace2.2-fine-tune-decoder-pressure-levels.sh).

JOB_NAME_BASE="ace22-era5-aimip-inference-SMOKE"
JOB_GROUP="ace22-era5-aimip"
# Set to the pressure-level-decoder fine-tune's beaker result dataset once that
# stage is evaluated (README workflow stage 4). The un-fine-tuned core checkpoint
# (01KXKBKW2DCAYX6Q3FRJ60K3Q6) cannot serve here: the inference configs request
# pressure-level outputs that only the fine-tuned decoder produces.
#
# Mounts best_inference_ckpt.tar (inference-error-selected), matching the ACE2.1
# workflow. The FT's inline inference now merges the co-extensive core+companion
# datasets, so it scores all pressure-level targets and best_inference_error is
# finite -> best_inference_ckpt.tar is written and drives selection.
EXISTING_RESULTS_DATASET="01KZGF9DX8Q2TD3ATJ93NSDQMQ"  # ACE2.2 PL-decoder FT (wandb gvoj4hf7, early-stopped e73, best_inference_error 0.0586)
if [ -z "$EXISTING_RESULTS_DATASET" ]; then
    echo "ERROR: set EXISTING_RESULTS_DATASET to the fine-tuned ACE2.2 checkpoint dataset ID" >&2
    exit 1
fi
# ai2cm/ace commit compatible with the nobgd4ek checkpoint (needs global_mean_removal;
# the ACE2.1 pin 70c966ed5 predates it). Head of
# experiment/2026-07-17-aimip-1deg-no-co2-nobgd4ek-run, used by the July 2026 runs.
ACE_GIT_REF="632ca493eb710eba91ba0ca717b05823f9b38f9e"
OUTPUT_ROOT="/climate-default/2026-08-08-ace22-era5-aimip-inference-SMOKE"
IC_DIR="/climate-default/2026-07-20-aimip-evaluation-daily/aimip-evaluation-ics-daily"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")

AIMIP_INFERENCE_BASE_CONFIG_PATH=$REPO_ROOT/configs/ace2.2-era5/ace-aimip-inference-config.yaml
AIMIP_INFERENCE_BASE_P2K_CONFIG_PATH=$REPO_ROOT/configs/ace2.2-era5/ace-aimip-inference-p2k-config.yaml
AIMIP_INFERENCE_BASE_P4K_CONFIG_PATH=$REPO_ROOT/configs/ace2.2-era5/ace-aimip-inference-p4k-config.yaml

python -m fme.ace.validate_config --config_type inference $AIMIP_INFERENCE_BASE_CONFIG_PATH
python -m fme.ace.validate_config --config_type inference $AIMIP_INFERENCE_BASE_P2K_CONFIG_PATH
python -m fme.ace.validate_config --config_type inference $AIMIP_INFERENCE_BASE_P4K_CONFIG_PATH

launch_job () {

    local JOB_NAME=$1
    local TEMPLATE_CONFIG=$2
    local IC=$3
    local OVERRIDE=$4

    # Apply IC substitution locally and encode for transfer to container.
    # gantry clones the remote repo so local paths don't exist on the cluster.
    local CONFIG_B64
    CONFIG_B64=$(sed "s/_r[0-9]i/_r${IC}i/g" "$TEMPLATE_CONFIG" | base64 | tr -d '\n')

    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref $ACE_GIT_REF \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2.2-ERA5 AIMIP inference' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/titan-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/ic-config.yaml && python -I -m fme.ace.inference /tmp/ic-config.yaml --override ${OVERRIDE}"

}

# launch 46-year (1979-2024) with spinup from 1978-10-01
# use 5 different initial conditions files
for IC in 1; do
    JOB_NAME="${JOB_NAME_BASE}-IC${IC}"
    IC_PATH="${IC_DIR}/1978-09-30_IC$(( IC - 1 )).nc" # files are 0-indexed
    OUTPUT_PATH="${OUTPUT_ROOT}/${JOB_NAME}"
    OVERRIDE="initial_condition.path=${IC_PATH} experiment_dir=${OUTPUT_PATH} n_forward_steps=400"
    echo "Launching job $JOB_NAME with override: $OVERRIDE"
    launch_job "$JOB_NAME" "$AIMIP_INFERENCE_BASE_CONFIG_PATH" "$IC" "$OVERRIDE"
done

