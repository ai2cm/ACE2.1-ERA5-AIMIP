#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Seed selection for ACE2.2-ERA5, following ACE2.1-ERA5's design: for each of the four
# training seeds' stage-2 checkpoints (best_inference_ckpt.tar), one 36-year rollout from a
# single initial condition and seven 5-year rollouts spread across 1980-2010, all ending by
# 2014-12-31. Stage 3 then ran for the selected seed only; it fits the pressure-level
# decoder on a frozen core, so selecting before it loses nothing.
#
# Seed 3 was selected. It ranked first on every measure on the 36-year rollout
# (across-channel mean and median of normalized time-mean error, annual-mean 2 m
# temperature RMSE, and share of the observed 2 m temperature trend reproduced), and on all
# but the median on the 5-year rollouts, where seed 2 ranked first.
#
# ================================== run record ==================================
# All jobs ran at ai2cm/ace 394e41b5e on one titan GPU, with the configs here.
#
#   run                  beaker experiment           wandb     result dataset
#   single, seed 0       01M1TPA320KXJ09DP1AYJKCA5R  32gm1x7i  01M1TPA324973346DS0WM55HGY
#   single, seed 1       01M1TPA9KYATSSVGTR9DD6X3RQ  xuepf9gq  01M1TPA9M4WX9XTT3G9ZTZQ2Q1
#   single, seed 2       01M1TPAFVATZ9G8WRT3G7GGFYM  kv0d7t5y  01M1TPAFVF1XA6ZK0QXPN9Q06Z
#   single, seed 3       01M1TPAP0YKF38VT9W1E3NT8WR  qh7kv0s5  01M1TPAP1576692S8D193HHP48
#   7x 5-year, seed 0    01M1TPA642SE28FYDRVFQYJCCG  aqktw8an  01M1TPA646QJBEH3RKHC6SN7FT
#   7x 5-year, seed 1    01M1TPACRKMN7NZZ09WK8E8BWS  1fyae9ax  01M1TPACRP81MHJ74FCRTDJDHY
#   7x 5-year, seed 2    01M1TPAJXR4YX0BD10NQE0E02R  58kacqdd  01M1TPAJXVCBN5RPPW4KCMK6V4
#   7x 5-year, seed 3    01M1TPAS3H9M26YXG545ASJ11E  3zmsbnnw  01M1TPAS3N367TVDEEHNV4BCQC

JOB_GROUP="ace22-era5-6h-aimip"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
CONFIG_DIR=$REPO_ROOT/configs/ace2.2-era5

# Same ref as scripts/run-ace2.2-6h-train.sh; see the note there.
ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
# wandb runs on a service-account key, so this is the only thing attributing them
# to a human, and it differs from the beaker username.
WANDB_USERNAME="bhenn1983"

# Each seed's stage-2 result dataset, as in STAGE2_DONOR in scripts/run-ace2.2-6h-train.sh.
declare -A STAGE2_CHECKPOINT=(
    [0]=01M0RFP2DKAGABV89KRPMXX5C3
    [1]=01M1NGC06ZYNV8CNX62WE7JQ4P
    [2]=01M1NGNEN34KX4BVKWPTNFA2YG
    [3]=01M1NHPDTXGJGJBBCP24H42DRK
)

launch_job () {
    local CONFIG_FILENAME=$1
    local JOB_NAME=$2
    local CHECKPOINT_DATASET=$3
    local CONFIG_PATH=$CONFIG_DIR/$CONFIG_FILENAME

    CONFIG_B64=$(base64 < "$CONFIG_PATH" | tr -d '\n')

    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref $ACE_GIT_REF \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2.2-ERA5 evaluator for AIMIP seed selection' \
        --beaker-image "$(cat $CONFIG_DIR/deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --min-runtime 8h \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $CHECKPOINT_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 200GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && \
                    python -I -m fme.ace.evaluator /tmp/config.yaml"
}

for SEED in 0 1 2 3; do
    launch_job "ace-evaluator-seed-selection-single-config.yaml" \
        "long36-seedsel-single-rs$SEED" "${STAGE2_CHECKPOINT[$SEED]}"
    launch_job "ace-evaluator-seed-selection-config.yaml" \
        "long36-seedsel-7ic-rs$SEED" "${STAGE2_CHECKPOINT[$SEED]}"
done
