# ACE2.1-ERA5 (AIMIP) training and evaluation

This directory contains scripts and configurations for training and running an ACE2 model on ERA5
data for the AIMIP evaluation protocol. This configuration is referred to as **ACE2.1-ERA5**. The
model predicts pressure-level diagnostic variables over 1978–2024 with multiple initial conditions
and SST perturbation scenarios.

## Setup

Create and activate the conda environment:

```bash
make create-env
conda activate ace-aimip
```

This installs `fme` (for config validation), `beaker-gantry` (for job submission), and `cftime` (for postprocessing).

## Workflow

The full experiment proceeds in six stages. Checkpoint IDs embedded in the scripts must be updated
manually after evaluating each stage.

### 1. Train

```bash
make train
```

Trains 4 random-seed ensemble members (RS0–RS3) on ERA5 1979–2008, with validation on 2009–2014.
Config: `ace-train-config.yaml`.

### 2. Evaluate training seeds

```bash
make evaluate
```

Evaluate all 4 trained checkpoints to select the best seed for fine-tuning.

- `run-ace-evaluator-seed-selection.sh` — 7x 5-year evaluations (starting in 1980, 1985, 1990,
  1995, 2000, 2005, 2010). Config: `ace-evaluator-seed-selection-config.yaml`.
- `run-ace-evaluator-seed-selection-single.sh` — single continuous 36-year run (1978-10-01 to
  2014-12-31). Config: `ace-evaluator-seed-selection-single-config.yaml`.

The best seed is chosen based on comparing the time-mean climate and trend skill, both in
the 7x 5-year and 36-year evaluations; this is somewhat subjective. The chosen seed is used
in `run-ace-fine-tune-decoder-pressure-levels.sh`.

### 3. Fine-tune

```bash
make fine-tune
```

Freezes the best trained checkpoint and trains a secondary MLP decoder for 65 pressure-level
diagnostic variables (TMP, Q, UGRD, VGRD, h at 13 pressure levels plus near-surface fields) across
4 new random seeds. Config: `ace-fine-tune-pressure-level-separate-decoder-config.yaml`.

### 4. Evaluate fine-tuned seeds

```bash
make evaluate
```

Re-run both evaluator scripts from step 2. The scripts already include checkpoint IDs for both
the trained and fine-tuned ensemble members, enabling direct comparison. After evaluating seeds
similarly as before (though there is little variability due frozen prognostic state), the best
checkpoint ID is used in `run-ace-inference.sh`.

### 5. Run inference

```bash
make inference
```

Runs 15 parallel 46-year simulations (1978-10-01 to 2024-12-31) using the best fine-tuned
checkpoint:

- 5 initial conditions (IC1–IC5) from the AIMIP IC dataset
- 3 SST scenarios: baseline, +2 K, +4 K

Configs: `ace-aimip-inference-config.yaml`, `ace-aimip-inference-p2k-config.yaml`,
`ace-aimip-inference-p4k-config.yaml`.

### 6. Postprocess inference outputs

```bash
make postprocess ARGS="--raw-results-dir gs://... --processed-results-dir gs://..."
```

Converts the raw 6-hourly inference outputs from step 5 into CMIP6-compliant daily and monthly
mean NetCDF files. Transformations include: time coordinate standardization, stacking of
per-level variables into a single 3D array along a `plev` or `model_layer` dimension, coordinate
bounds computation, CF metadata assignment, and CMIP6 global attribute assignment.

The postprocessing script and its configuration files live in `scripts/aimip_postprocessing/`.
Run `python scripts/aimip_postprocessing/postprocess.py --help` for the full option list.

Key options:

| Option | Description |
|---|---|
| `--raw-results-dir` | Directory containing raw inference outputs (required) |
| `--processed-results-dir` | GCS destination for processed results |
| `--output-version` | Version string in output paths (default: `v20251130`) |
| `--simulation NAME` | Process a single simulation instead of all 15 |
| `--skip-gcs-upload` | Write locally only, skip GCS upload |

Output files follow the CMIP6 Data Reference Syntax:

```
{local_dir}/{experiment_id}/{variant_label}/{table_id}/{varname}/{grid_label}/{version}/{filename}.nc
```

Run `make test-postprocess` to execute the unit test suite for the postprocessing helpers.
