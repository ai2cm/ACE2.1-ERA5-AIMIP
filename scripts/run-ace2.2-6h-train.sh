#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Train the ACE2.2-ERA5 (6-hourly) model, in three stages:
#
#   1. 1-step pre-training
#   2. multi-step fine-tuning, initialized from stage 1's best_ckpt.tar
#   3. pressure-level decoder fine-tuning, initialized from stage 2's
#      best_inference_ckpt.tar, with the core frozen
#
# This supersedes the daily ACE2.2-ERA5 (wandb nobgd4ek), whose 06Z-snapshot output
# could not meet AIMIP's evenly-sampled diurnal-average spec for surface fields.
# Unlike that model, the 6-hourly core is trained entirely from configs in this
# repo, so the whole pipeline is reproducible from here.
#
# Each stage reads its donor checkpoint from a `# arg:` header line in its config
# (see the parsing loop below), so a config records the exact dataset it was run
# against. To reproduce from scratch, substitute the dataset ids your own runs
# produce.
#
# Uncomment one stage at a time: each depends on the previous stage's result
# dataset, which must be committed (job fully finalized) before beaker will mount it.

JOB_GROUP="ace22-era5-6h-aimip"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
CONFIG_DIR=$REPO_ROOT/configs/ace2.2-era5/training
BASE_NAME="train-1deg-6hourly-v2-era5-only-no-residual-no-co2"

# ai2cm/ace commit the configs were run at. The 6-hourly configs live on the
# exp/2026-08-12-aimip-1deg-6hourly branch there; this is its tip.
ACE_GIT_REF="fa856b459"

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
# wandb runs on a service-account key, so this is the only thing attributing them
# to a human, and it differs from the beaker username.
WANDB_USERNAME="bhenn1983"
# Single cluster on purpose: per-GPU memory is cluster-specific, so N_GPUS is only
# correct on titan. Retarget by changing both together, not by adding a --cluster.
N_GPUS=4

launch_training () {
    CONFIG_FILENAME=$1
    JOB_NAME=$2
    CONFIG_PATH=$CONFIG_DIR/$CONFIG_FILENAME

    # Per-stage beaker args (the donor checkpoint mount) are declared in the
    # config itself as `# arg:` lines, keeping the config and the dataset it
    # needs in one place.
    local extra_args=()
    while IFS= read -r line; do
        [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
    done < "$CONFIG_PATH"

    CONFIG_B64=$(base64 < "$CONFIG_PATH" | tr -d '\n')

    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref $ACE_GIT_REF \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Train ACE2.2-ERA5 (1 degree, 6-hourly)' \
        --beaker-image "$(cat $REPO_ROOT/configs/ace2.2-era5/deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --preemptible \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        "${extra_args[@]}" \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/train-config.yaml && \
                    torchrun --nproc_per_node ${N_GPUS} -m fme.ace.train /tmp/train-config.yaml"
}

# --- Stage 1: 1-step pre-training ---
# Ran as beaker 01KZYJ4HT4ZMZH296KBNWMPCQF / wandb g94277n6 (40 epochs, 44.5h on 4 GPUs).
# Result dataset 01KZYJ4HTBWED5VG3VFTRYKDRC is stage 2's donor.
# launch_training "$BASE_NAME.yaml" "ace22-era5-6h-1-step-pre-training-rs0"

# --- Stage 2: multi-step fine-tuning ---
# Ran as beaker 01M06W0WN4WY1HJBWFXJNJEXC7 / wandb 78crdqjr (40 epochs, 81h across 35
# preempted attempts). best_inference_error was flat from epoch 8 onward, so its
# best_inference_ckpt.tar is the epoch-8 checkpoint. Result dataset
# 01M0RFP2DKAGABV89KRPMXX5C3 is stage 3's donor.
# launch_training "$BASE_NAME-multi-step-ft.yaml" "ace22-era5-6h-multi-step-fine-tuning-rs0"

# --- Stage 3: pressure-level decoder fine-tuning ---
# Adds ta/hus/ua/va at the 7 AIMIP evaluation levels (1000/850/700/500/250/100/50 hPa)
# via a secondary decoder head, with the core frozen (~20k trainable parameters).
# Ran as beaker 01M0TQP4447HF0J96D7FWQ02V6 / wandb lmvpfmrp across 3 preempted attempts,
# stopped by hand at epoch 35: inference error had no trend across 17 evaluations
# (0.0309-0.0335). Result dataset 01M0WVHBW4G5H2M2NZ25REP8G4; its best_inference_ckpt.tar
# is the epoch-32 checkpoint (0.030889) that AIMIP inference mounts.
# launch_training "$BASE_NAME-plev-ft.yaml" "ace22-era5-6h-plev-fine-tuning-rs0"
