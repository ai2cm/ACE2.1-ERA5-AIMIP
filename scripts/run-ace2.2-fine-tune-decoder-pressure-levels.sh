#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Fine-tune a trainable pressure-level secondary decoder on top of the FROZEN
# ACE2.2-ERA5 core (1-deg daily-06Z v2 ERA5-only recipe, non-residual, no CO2;
# wandb ai2cm/ace/nobgd4ek). Single seed: the frozen core is identical across
# seeds — only the decoder init and data shuffling vary.
#
# Prerequisites (see configs/ace2.2-era5/ace-fine-tune-pressure-level-separate-decoder-config.yaml
# header for details):
#   1. MERGED_STATS_DATASET: beaker dataset combining the core's daily training
#      stats with the pressure-level variable stats, mounted at /statsdata.
#   2. The config's pressure-level target must be aligned to the core's
#      daily-06Z time coordinate.

JOB_NAME_BASE="ace22-era5-aimip-fine-tune-decoder-pressure-levels"
JOB_GROUP="ace22-era5-aimip"
CONFIG_FILENAME="ace-fine-tune-pressure-level-separate-decoder-config.yaml"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
CONFIG_PATH=$REPO_ROOT/configs/ace2.2-era5/$CONFIG_FILENAME
# ACE2.2-ERA5 core checkpoint: wandb ai2cm/ace/nobgd4ek (best_inference_ckpt.tar),
# trained on ai2cm/ace branch experiment/2026-07-15-1deg-daily-v2-era5-only-no-residual-no-co2 @ f9f91c9.
EXISTING_RESULTS_DATASET="01KXKBKW2DCAYX6Q3FRJ60K3Q6"
# Merged normalization stats (core daily stats + pressure-level stats); see header.
MERGED_STATS_DATASET="01KZC115ZD5Q39V3JPZGMG1DF9"  # companion build result (2026-03-19-source plev stats + core stats, merged core-first)
if [ -z "$MERGED_STATS_DATASET" ]; then
    echo "ERROR: set MERGED_STATS_DATASET to the merged core+plev stats dataset ID" >&2
    exit 1
fi
# ai2cm/ace commit compatible with the nobgd4ek checkpoint (needs global_mean_removal;
# the ACE2.1 pin 70c966ed5 predates it).
ACE_GIT_REF="632ca493eb710eba91ba0ca717b05823f9b38f9e"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
N_GPUS=4

python -m fme.ace.validate_config --config_type train $CONFIG_PATH
CONFIG_B64=$(base64 < "$CONFIG_PATH" | tr -d '\n')

launch_job () {

    JOB_NAME=$1
    shift
    OVERRIDE="$@"

    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref $ACE_GIT_REF \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Fine-tune ACE2.2-ERA5 pressure-level decoder on AIMIP period' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --preemptible \
        --cluster ai2/titan-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $MERGED_STATS_DATASET:/statsdata \
        --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/base_weights/ckpt.tar \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && torchrun --nproc_per_node $N_GPUS -m fme.ace.train /tmp/config.yaml --override ${OVERRIDE}"

}

# fine tune with separate decoder for pressure levels, with LR warmup
for SEED in 0; do
    JOB_NAME="${JOB_NAME_BASE}-separate-decoder-lr-warmup-RS${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job $JOB_NAME $OVERRIDE
done
