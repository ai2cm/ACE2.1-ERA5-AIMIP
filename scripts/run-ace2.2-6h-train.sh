#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Train the ACE2.2-ERA5 (6-hourly) model, in three stages:
#
#   1. 1-step pre-training
#   2. multi-step fine-tuning, initialized from the same seed's stage-1 best_ckpt.tar
#   3. pressure-level decoder fine-tuning, initialized from the same seed's stage-2
#      best_inference_ckpt.tar, with the core frozen
#
# This supersedes the daily ACE2.2-ERA5 (wandb nobgd4ek), whose 06Z-snapshot output
# could not meet AIMIP's evenly-sampled diurnal-average spec for surface fields.
# Unlike that model, the 6-hourly core is trained entirely from configs in this
# repo, so the whole pipeline is reproducible from here.
#
# As for ACE2.1-ERA5, stages 1 and 2 ran for four training seeds (0-3), the seed was
# chosen on 1979-2014 rollouts of the stage-2 checkpoints
# (scripts/run-ace2.2-6h-seed-selection.sh), and stage 3 ran for that seed only. Stage 3
# fits the pressure-level decoder on a frozen core, so it cannot change the comparison
# between seeds.
#
# Usage:
#   STAGE=1 SEEDS="0 1 2 3" bash scripts/run-ace2.2-6h-train.sh
#   STAGE=2 SEEDS="0 1 2 3" bash scripts/run-ace2.2-6h-train.sh
#   STAGE=3 SEEDS="3"       bash scripts/run-ace2.2-6h-train.sh
#   CLUSTER=jupiter STAGE=1 SEEDS="1" bash scripts/run-ace2.2-6h-train.sh
#
# Stages 2 and 3 mount the same seed's previous-stage result dataset from the
# STAGE1_DONOR / STAGE2_DONOR tables below, which hold the runs that produced the
# submitted model. To reproduce from scratch, replace them with the ids your own runs
# produce. Take each id from the job that exited 0, once its dataset is committed: a
# preempted-and-resumed experiment also commits each dead job's dataset, whose checkpoints
# stop short of the final epoch.
#
# ================================== run record ==================================
# Every run used the configs in configs/ace2.2-era5/training/ (all four stage-1 runs and seed 0's
# stage 2 with an earlier inline inference list; see those configs' headers), the beaker image in
# configs/ace2.2-era5/deps_only_image.txt, and ai2cm/ace code identical to ACE_GIT_REF.
#
# stage 1, 40 epochs
#   seed  beaker experiment           wandb     cluster  GPUs  result dataset
#   0     01KZYJ4HT4ZMZH296KBNWMPCQF  g94277n6  titan    4     01KZYJ4HTBWED5VG3VFTRYKDRC
#   1     01M1J6WF8B2FDS0GCFMXVNEA87  eqyklvjp  jupiter  8     01M1J6WF8NQM6AV14WW90MF2T9
#   2     01M1J6WW2Y7EQB8KWZGACRC2JC  1g0jp15g  jupiter  8     01M1J6WW372RKP423AYD0A9JYQ
#   3     01M1J6X90YXZ7EFFG01BD8CWEB  qywpn6kq  jupiter  8     01M1J6X918YK0W7ZP9WA422MT9
#
# stage 2, 40 epochs; best_inference_ckpt.tar is the epoch given
#   0     01M06W0WN4WY1HJBWFXJNJEXC7  78crdqjr  titan    4     01M0RFP2DKAGABV89KRPMXX5C3  epoch 8
#   1     01M1NGC06NPTEZJS1H2KSE8KV2  umx5wd9k  jupiter  8     01M1NGC06ZYNV8CNX62WE7JQ4P  epoch 4
#   2     01M1NGNEMV2A1ETM4ZGAZ1X2T4  fijdbmy6  jupiter  8     01M1NGNEN34KX4BVKWPTNFA2YG  epoch 20
#   3     01M1NHPDTMZ0N8K9W7XEWPRYP5  t7cxx6bo  jupiter  8     01M1NHPDTXGJGJBBCP24H42DRK  epoch 24
#
# stage 3, up to 50 epochs
#   3     01M1WBBJD18MDTWJD67TJKMM9N  dd0f9jnb  titan    4     01M1WBBJDFYA9VS46V6HH3K1MW  epoch 42
#   0     01M0TQP4447HF0J96D7FWQ02V6  lmvpfmrp  titan    4     01M0WVHBW4G5H2M2NZ25REP8G4  epoch 32
#
# Seed 3's stage 3 is the submitted model (AIMIP version v20260908). Seed 0's stage 3 ran
# before the seed ensemble existed and was evaluated as v20260825; it is superseded. Seed
# 0's stage 2 took 35 preempted attempts and its stage 3 was stopped by hand at epoch 35;
# every other run completed in a single job.

JOB_GROUP="ace22-era5-6h-aimip"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
CONFIG_DIR=$REPO_ROOT/configs/ace2.2-era5/training
BASE_NAME="train-1deg-6hourly-v2-era5-only-no-residual-no-co2"

# ai2cm/ace commit to run. The runs above were launched at b4a688d85, 379e1ec15, ee05a75f4
# (seed 0) and af0a56769, e10a9befe, 394e41b5e (seeds 1-3); fme/ is the same tree at all
# six, so they are interchangeable. This is the last, which AIMIP inference also uses.
ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
# wandb runs on a service-account key, so this is the only thing attributing them
# to a human, and it differs from the beaker username.
WANDB_USERNAME="bhenn1983"

# Cluster and GPU count move together: per-GPU memory sets how many ranks the batch
# needs. batch_size is global and fme splits it across data-parallel ranks, so the
# sequence of global batches is the same at 4 and 8 ranks, and a seed run on either
# cluster is the same experiment up to floating-point reduction order.
CLUSTER=${CLUSTER:-titan}
case "$CLUSTER" in
    titan)   BEAKER_CLUSTER="ai2/titan";   N_GPUS=4 ;;
    jupiter) BEAKER_CLUSTER="ai2/jupiter"; N_GPUS=8 ;;
    *)
        echo "ERROR: CLUSTER must be 'titan' or 'jupiter', got '$CLUSTER'." >&2
        exit 1
        ;;
esac

STAGE=${STAGE:-}
SEEDS=${SEEDS:-}

declare -A STAGE1_DONOR=(
    [0]=01KZYJ4HTBWED5VG3VFTRYKDRC
    [1]=01M1J6WF8NQM6AV14WW90MF2T9
    [2]=01M1J6WW372RKP423AYD0A9JYQ
    [3]=01M1J6X918YK0W7ZP9WA422MT9
)
declare -A STAGE2_DONOR=(
    [0]=01M0RFP2DKAGABV89KRPMXX5C3
    [1]=01M1NGC06ZYNV8CNX62WE7JQ4P
    [2]=01M1NGNEN34KX4BVKWPTNFA2YG
    [3]=01M1NHPDTXGJGJBBCP24H42DRK
)

launch_training () {
    local CONFIG_FILENAME=$1
    local JOB_NAME=$2
    local SEED=$3
    local DONOR=${4:-}
    local CONFIG_PATH=$CONFIG_DIR/$CONFIG_FILENAME

    # The donor is this seed's previous-stage result dataset, mounted at /weights where
    # the config's parameter_init expects it. Callers check it with require_donor first:
    # an initialization from the wrong seed's weights would run without complaint.
    local extra_args=()
    if [[ -n "$DONOR" ]]; then
        extra_args+=(--dataset "$DONOR:/weights")
    fi

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
        --min-runtime 8h \
        --cluster $BEAKER_CLUSTER \
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
                    torchrun --nproc_per_node ${N_GPUS} -m fme.ace.train /tmp/train-config.yaml \
                    --override seed=${SEED}"
}

require_donor () {
    local TABLE_NAME=$1
    local SEED=$2
    local -n TABLE=$TABLE_NAME
    if [[ -z "${TABLE[$SEED]:-}" ]]; then
        echo "ERROR: no $TABLE_NAME entry for seed $SEED." >&2
        exit 1
    fi
}

if [[ -z "$SEEDS" ]]; then
    echo "ERROR: set SEEDS, e.g. SEEDS=\"0 1 2 3\"." >&2
    exit 1
fi

case "$STAGE" in
    1)
        for SEED in $SEEDS; do
            launch_training "$BASE_NAME.yaml" "$BASE_NAME-rs$SEED" "$SEED"
        done
        ;;
    2)
        for SEED in $SEEDS; do
            require_donor STAGE1_DONOR "$SEED"
            launch_training "$BASE_NAME-multi-step-ft.yaml" "$BASE_NAME-multi-step-ft-rs$SEED" \
                "$SEED" "${STAGE1_DONOR[$SEED]}"
        done
        ;;
    3)
        # Stage 3 is for the selected seed only.
        if [[ $(wc -w <<< "$SEEDS") -ne 1 ]]; then
            echo "ERROR: STAGE=3 takes exactly one seed, the one seed selection chose; got SEEDS='$SEEDS'." >&2
            exit 1
        fi
        require_donor STAGE2_DONOR "$SEEDS"
        launch_training "$BASE_NAME-plev-ft.yaml" "$BASE_NAME-plev-ft-rs$SEEDS" \
            "$SEEDS" "${STAGE2_DONOR[$SEEDS]}"
        ;;
    *)
        echo "Usage: STAGE=<1|2|3> SEEDS=\"<seeds>\" [CLUSTER=titan|jupiter] $0" >&2
        exit 1
        ;;
esac
