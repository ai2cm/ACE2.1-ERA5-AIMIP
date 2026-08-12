#!/bin/bash

set -e

# Mirror the ACE2.2-ERA5 raw AIMIP inference results from weka to GCS so that
# postprocess.py (which runs locally) can read them, matching the ACE2.1
# precedent (gs://vcm-ml-intermediate/2025-11-25-ace-aimip-inference-results).
# Runs as a small beaker CPU job with the weka mount and the shared GCS
# service-account credentials; uses the Cloud SDK image for gcloud storage.

RESULTS_NAME="2026-08-05-ace22-era5-aimip-inference-results"
SRC="/climate-default/${RESULTS_NAME}"
DEST="gs://vcm-ml-intermediate/${RESULTS_NAME}"
JOB_NAME="ace22-era5-aimip-mirror-raw-results"

SPEC=$(mktemp /tmp/mirror-spec-XXXX.yaml)
cat > "$SPEC" <<EOF
version: v2
description: Mirror ACE2.2-ERA5 raw AIMIP inference results weka -> GCS
budget: ai2/atec-climate
tasks:
  - name: ${JOB_NAME}
    image:
      docker: google/cloud-sdk:slim
    command:
      - bash
      - -c
      - >-
        gcloud auth activate-service-account
        --key-file=/tmp/google_application_credentials.json &&
        gcloud storage cp -r ${SRC} gs://vcm-ml-intermediate/
    envVars:
      - name: GOOGLE_APPLICATION_CREDENTIALS
        value: /tmp/google_application_credentials.json
    datasets:
      - mountPath: /tmp/google_application_credentials.json
        source:
          secret: google-credentials
      - mountPath: /climate-default
        source:
          weka: climate-default
    result:
      path: /results
    resources:
      cpuCount: 8
      memory: 32 GiB
    context:
      priority: normal
      preemptible: true
    constraints:
      cluster:
        - ai2/titan
        - ai2/jupiter
        - ai2/ceres
        - ai2/saturn
EOF

beaker experiment create "$SPEC" --name "$JOB_NAME" --workspace ai2/ace
echo "Mirroring ${SRC} -> ${DEST}"
