#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Seed selection for ACE2.2-ERA5. For each of the four training seeds' stage-2 checkpoints
# (best_inference_ckpt.tar), three runs, each with the latent clip as trained and off:
#
#   long36    1979-2014 rollout from eight initial conditions
#             (configs/ace2.2-era5/ace-evaluator-seed-selection-long36-config.yaml)
#   control   46-year AIMIP-forced run from 1978-09-30T18Z, inference seed 1
#             (configs/ace2.2-era5/ace-seed-selection-inference-control-config.yaml)
#   p4k       the same with SSTs +4 K
#             (configs/ace2.2-era5/ace-seed-selection-inference-p4k-config.yaml)
#
# Scored on 1979-2014 only: normalized time-mean error over 34 outputs, the share of the
# ERA5 1979-2014 trend in annual global-mean 2 m temperature reproduced, and the +4 K
# response (p4k minus control, mean of 1979-2014 annual global means). Nothing from 2015 on
# was used. Stage 3 then ran for the selected seed only; it fits the pressure-level decoder on
# a frozen core, so selecting before it loses nothing.
#
# Seed 2 was selected, on both time-mean climate and trend. Seed 1 had the lowest time-mean
# error (0.0193 vs seed 2's 0.0247, clip as trained) but reproduced the least trend of the
# three seeds kept (47% vs 57%); seed 3 (39%) was screened out on trend. Turning the clip
# off moves trend and time-mean error little and leaves this choice unchanged, and raises
# seed 2's +4 K 2 m temperature response from 2.17 to 2.45 K.
#
# ================================== run record ==================================
# All 24 runs ran at ai2cm/ace 394e41b5e with the image in configs/ace2.2-era5/deps_only_image.txt,
# on one GPU, launched by ai2cm/ace configs/experiments/2026-10-28-ace22-diagnostic-near-surface/
# run.sh at 1e9ed6437 (job names as below, wandb group ace22-no-gmr-diagnostic-near-surface).
# This launcher reproduces them. Every run exited 0 in a single job.
#
#   seed  run      clip  beaker experiment           wandb
#   0     long36   on    01M6F9VSHQZ6DRVR2J1QAK4MDT  wlcn9hi7
#   0     long36   off   01M6F9VX9T32YX9SAMXQC7XVAX  nb2je773
#   0     control  on    01M6F9W0ZCNDAH60JQVVXYJ6YF  ux5nxp33
#   0     p4k      on    01M6F9W4WXCGKCB7A224TN9BZH  1qrsqn5s
#   0     control  off   01M6F9W8Q275XWKQD4525S0S3M  mvlzf51z
#   0     p4k      off   01M6F9WCMBAER5B9WSQ8YT256W  4ctc7isw
#   1     long36   on    01M6F9WH1ZM9K039MG5TM13E08  yd3yye4f
#   1     long36   off   01M6F9WMYX1S906MD2SZ6JR66R  hz4sfn2s
#   1     control  on    01M6F9WRRSXER3W80NDV3J154N  7qhycj64
#   1     p4k      on    01M6F9WWJEK1GKGH0NV3SBWSY8  4r8eypwh
#   1     control  off   01M6F9X0XNH0K074ZKJAYTPBZY  k8kl9qa4
#   1     p4k      off   01M6F9X4VE724S0VB1N9HFVG4R  bxx4a0ax
#   2     long36   on    01M6F9X9C0Y95MTTES298DYXJT  yow2bk32
#   2     long36   off   01M6F9XDQBMZZB26VFS57W906J  xt0c4jp0
#   2     control  on    01M6F9XHM0GGE7J2XHEJQMCAVP  bw7pyfld
#   2     p4k      on    01M6F9XNPSXMYP6WDT27CFECK4  ohwt8vpq
#   2     control  off   01M6F9XSJCJP5VFF7BQRWAJ39C  myvm2fz0
#   2     p4k      off   01M6F9XXDNF6X79W0D9FGBWKA9  bhnb0yaf
#   3     long36   on    01M6F9Y33QKM44T8ZT0VT3ZMT7  5x95v4nn
#   3     long36   off   01M6F9Y6YKV7K1JMZBVEDZJAB7  tmgft0s1
#   3     control  on    01M6F9YB5D297YRZAMBZZ0DVA1  j42fdndb
#   3     p4k      on    01M6F9YF6AQV3PPKHDPFEE9JCH  rkxn5244
#   3     control  off   01M6F9YJRB8878801C901GBNNC  1sblzcwc
#   3     p4k      off   01M6F9YPSJB3JEAMF4PBW8R340  dchupuz9
#
# The earlier ACE2.2-ERA5 recipe (global mean removal, prognostic near-surface fields) was
# selected with ACE2.1's two evaluations instead; its configs and record are in this
# repository's history at bbb23aa.
#
# Usage:
#   JOB_PREFIX=<new prefix> bash scripts/run-ace2.2-6h-seed-selection.sh
#   JOB_PREFIX=<new prefix> SEEDS="2" bash scripts/run-ace2.2-6h-seed-selection.sh
# The default prefix gives the recorded runs' names; beaker experiment names must be
# unique, so a rerun needs its own.

JOB_PREFIX="${JOB_PREFIX:-ace22-no-gmr-diagnostic-near-surface}"
JOB_GROUP="${JOB_GROUP:-ace22-no-gmr-diagnostic-near-surface}"
SEEDS="${SEEDS:-0 1 2 3}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
CONFIG_DIR=$REPO_ROOT/configs/ace2.2-era5

# Same ref as scripts/run-ace2.2-6h-train.sh; see the note there.
ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"
IC_PATH="/climate-default/2026-08-24-aimip-evaluation/aimip-evaluation-ics/1978-09-30_IC0.nc"
OUTPUT_ROOT="${OUTPUT_ROOT:-/climate-default/2026-10-28-ace22-diagnostic-near-surface-perturbation}"

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
# wandb runs on a service-account key, so this is the only thing attributing them
# to a human, and it differs from the beaker username.
WANDB_USERNAME="bhenn1983"

# Each seed's stage-2 result dataset, as in STAGE2_DONOR in scripts/run-ace2.2-6h-train.sh.
declare -A STAGE2_CHECKPOINT=(
    [0]=01M67JG3AGYQDE682FMZJ3B3NF
    [1]=01M67JGAH1RY6AP76N8E49CF50
    [2]=01M67JGHR6BW6ZWAQFXX94F331
    [3]=01M67JGRTM7SBJS5159JE8CA7S
)

CLIP_OFF_B64=$(base64 < "$SCRIPT_DIR/clip_off.py" | tr -d '\n')
# The clip-off copy is made inside the job, since /ckpt.tar is read-only.
CLIP_OFF="python -I /tmp/clip_off.py /ckpt.tar /tmp/ckpt.tar"

launch_job () {
    local CONFIG_FILENAME=$1
    local JOB_NAME=$2
    local CHECKPOINT_DATASET=$3
    local COMMAND=$4
    local CONFIG_B64
    CONFIG_B64=$(base64 < "$CONFIG_DIR/$CONFIG_FILENAME" | tr -d '\n')

    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref $ACE_GIT_REF \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2.2-ERA5 seed-selection evaluation' \
        --beaker-image "$(cat $CONFIG_DIR/deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --min-runtime 8h \
        --cluster ai2/titan \
        --cluster ai2/jupiter \
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
                    echo '${CLIP_OFF_B64}' | base64 -d > /tmp/clip_off.py && $COMMAND"
}

EVAL="torchrun --nproc_per_node 1 -m fme.ace.evaluator /tmp/config.yaml"
for SEED in $SEEDS; do
    CKPT_DATASET=${STAGE2_CHECKPOINT[$SEED]:?no stage-2 dataset for seed $SEED}
    launch_job ace-evaluator-seed-selection-long36-config.yaml \
        "$JOB_PREFIX-long36-rs$SEED" "$CKPT_DATASET" "$EVAL"
    launch_job ace-evaluator-seed-selection-long36-config.yaml \
        "$JOB_PREFIX-long36-clipoff-rs$SEED" "$CKPT_DATASET" \
        "$CLIP_OFF && $EVAL --override checkpoint_path=/tmp/ckpt.tar"
    for CLIP in on off; do
        for EXP in control p4k; do
            if [[ $CLIP == on ]]; then
                JOB_NAME="$JOB_PREFIX-perturbation-$EXP-rs$SEED" PREP="true" CKPT_PATH=/ckpt.tar
            else
                JOB_NAME="$JOB_PREFIX-perturbation-$EXP-clipoff-rs$SEED" PREP=$CLIP_OFF CKPT_PATH=/tmp/ckpt.tar
            fi
            OVERRIDE="checkpoint_path=$CKPT_PATH initial_condition.path=$IC_PATH"
            OVERRIDE="$OVERRIDE experiment_dir=$OUTPUT_ROOT/$JOB_NAME seed=1"
            launch_job "ace-seed-selection-inference-$EXP-config.yaml" "$JOB_NAME" "$CKPT_DATASET" \
                "$PREP && python -I -m fme.ace.inference /tmp/config.yaml --override $OVERRIDE"
        done
    done
done
