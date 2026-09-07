#!/bin/bash

set -e

# One short min-runtime-protected job to validate the ACE2.2-ERA5 (6-hourly) inference
# configs end to end before committing the 15-job sweep. Check the raw output
# carries every files.yaml entry and all 7 pressure levels.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

JOB_NAME_BASE="ace22-era5-6h-rs3-aimip-inference-SMOKE" \
JOB_GROUP="ace22-era5-6h-rs3-aimip" \
EXPERIMENTS="control" \
MEMBERS="1" \
EXTRA_OVERRIDE="n_forward_steps=400" \
MIN_RUNTIME="1h" \
OUTPUT_ROOT="/climate-default/2026-09-07-ace22-era5-6h-rs3-aimip-inference-SMOKE" \
    exec "$SCRIPT_DIR/run-ace2.2-6h-inference.sh"
