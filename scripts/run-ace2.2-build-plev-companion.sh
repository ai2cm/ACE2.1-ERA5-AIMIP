#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Build the ACE2.2-ERA5 pressure-level fine-tune inputs in one beaker job:
#   1. Daily-06Z subsample of the 6-hourly ERA5 pressure-level zarr
#      (scripts/data_process/time_coarsen.py; snapshot selection is valid because
#      all 65 pressure-level variables are instantaneous state fields).
#   2. Normalization stats (1990-2019) for the subsampled zarr
#      (scripts/data_process/get_stats.py).
#   3. Merged stats: ACE2.2 core daily training stats + the new pressure-level
#      stats (scripts/data_process/merge_stats.py), written to /results so the
#      job's beaker result dataset IS the merged stats dataset to mount at
#      /statsdata in the fine-tune.
#
# Each stage skips work whose output already exists, so the job is idempotent.

JOB_NAME="ace22-era5-aimip-build-plev-companion-0319"
# ai2cm/ace commit providing scripts/data_process (same pin as the other ACE2.2 stages).
ACE_GIT_REF="632ca493eb710eba91ba0ca717b05823f9b38f9e"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
CONFIG_DIR=$REPO_ROOT/configs/ace2.2-era5/data-process

COARSEN_B64=$(base64 < "$CONFIG_DIR/plev-0319-daily-06z-coarsen.yaml" | tr -d '\n')
STATS_B64=$(base64 < "$CONFIG_DIR/plev-0319-daily-06z-stats.yaml" | tr -d '\n')
MERGE_B64=$(base64 < "$CONFIG_DIR/merge-core-plev-0319-stats.yaml" | tr -d '\n')
PATCH_B64=$(base64 < "$CONFIG_DIR/patch_time_coarsen_empty_windows.py" | tr -d '\n')

gantry run \
    --remote https://github.com/ai2cm/ace \
    --ref $ACE_GIT_REF \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Build ACE2.2-ERA5 fine-tune inputs (daily-06Z plev zarr + merged stats)' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --not-preemptible \
    --cluster ai2/titan-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --env WANDB_MODE=disabled \
    --cpus 16 \
    --memory 64GiB \
    --shared-memory 16GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install --no-deps . && pip install xpartition dacite click distributed" \
    -- bash -c "echo '${COARSEN_B64}' | base64 -d > /tmp/coarsen.yaml && echo '${STATS_B64}' | base64 -d > /tmp/stats.yaml && echo '${MERGE_B64}' | base64 -d > /tmp/merge.yaml && echo '${PATCH_B64}' | base64 -d > /tmp/patch_time_coarsen.py && python /tmp/patch_time_coarsen.py && python scripts/data_process/time_coarsen.py /tmp/coarsen.yaml 0 && python scripts/data_process/get_stats.py /tmp/stats.yaml 0 && python scripts/data_process/merge_stats.py /tmp/merge.yaml"
