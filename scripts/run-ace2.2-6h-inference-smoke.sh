#!/bin/bash

set -e

# One short non-preemptible job to validate the ACE2.2-ERA5 (6-hourly) inference
# configs end to end before committing the 15-job sweep. Check the raw output
# carries every files.yaml entry and all 7 pressure levels.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

JOB_NAME_BASE="ace22-era5-6h-aimip-inference-SMOKE" \
JOB_GROUP="ace22-era5-6h-aimip" \
EXPERIMENTS="control" \
MEMBERS="1" \
EXTRA_OVERRIDE="n_forward_steps=400" \
PREEMPTION_FLAG="--not-preemptible" \
OUTPUT_ROOT="/climate-default/2026-08-25-ace22-era5-6h-aimip-inference-SMOKE" \
    exec "$SCRIPT_DIR/run-ace2.2-6h-inference.sh"
