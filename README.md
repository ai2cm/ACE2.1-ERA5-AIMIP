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

## ACE2.2-ERA5

A second evaluated checkpoint, **ACE2.2-ERA5** (CMIP source_id `ACE2-2-ERA5`), runs the same
AIMIP inference/postprocessing workflow for a substantially revised recipe: 1° **daily-06Z**
timestep, v2 ERA5-only training recipe with revised normalization and module hyperparameters,
non-residual prediction, and no CO₂ input. Its configs live in `configs/ace2.2-era5/` and its
launchers in `scripts/run-ace2.2-*.sh`.

### Evaluated checkpoints

| Model | Core training | Beaker checkpoint dataset | ACE code ref |
|---|---|---|---|
| ACE2.1-ERA5 | `ace-aimip-train-rs3` + PL-decoder fine-tune RS0 | `01KAKXY0EK24K7BZK2N8SPJ5SJ` | `70c966ed5` |
| ACE2.2-ERA5 | wandb `ai2cm/ace/nobgd4ek` (`01KXKBKW2DCAYX6Q3FRJ60K3Q6`) + PL-decoder fine-tune (see below) | set in `scripts/run-ace2.2-inference.sh` after stage 3 | `632ca493e` |

The ACE2.2 core was trained on ai2cm/ace branch
`experiment/2026-07-15-1deg-daily-v2-era5-only-no-residual-no-co2` @ `f9f91c9` (beaker experiment
`01KXKBKVTVPPQHBY4DRAQ8NHH1`); stages 1–2 of the six-stage workflow above are replaced by that
training run. The remaining ACE2.2 stages are:

### 1. Build fine-tune inputs

```bash
bash scripts/run-ace2.2-build-plev-companion.sh
```

One beaker job that (a) subsamples the 6-hourly ERA5 pressure-level zarr to the core's daily-06Z
cadence (valid because all 65 pressure-level variables are instantaneous snapshots), (b) computes
its 1990–2019 normalization stats, and (c) merges those with the core's own daily training stats.
The job's beaker **result dataset** is the merged stats dataset; note its ID.
Configs: `configs/ace2.2-era5/data-process/`.

### 2. Fine-tune the pressure-level decoder

```bash
bash scripts/run-ace2.2-fine-tune-decoder-pressure-levels.sh   # set MERGED_STATS_DATASET first
```

Single seed (the frozen core is identical across seeds). Config:
`configs/ace2.2-era5/ace-fine-tune-pressure-level-separate-decoder-config.yaml` — see its header
for the daily-cadence adaptations relative to the ACE2.1 fine-tune.

### 3. Evaluate the fine-tuned checkpoint

Check wandb validation loss and inline-inference time-mean maps; note the fine-tune's beaker
result dataset ID and set it as `EXISTING_RESULTS_DATASET` in `scripts/run-ace2.2-inference.sh`.

### 4. Run inference

```bash
bash scripts/run-ace2.2-inference.sh
```

Same 15-job matrix as ACE2.1 (5 ICs × baseline/+2K/+4K), with daily ICs and forcing from
`/climate-default/2026-07-20-aimip-evaluation-daily/` and 16894 daily steps
(1978-09-30 → 2024-12-31).

### 5. Postprocess

```bash
make postprocess ARGS="--raw-results-dir ... --processed-results-dir ... \
    --simulations-file simulations-ace2.2.yaml \
    --model-source-name ACE2-2-ERA5 \
    --source-description 'ACE2-2-ERA5: ACE (Ai2 climate emulator) version 2.2 trained on ERA5' \
    --daily-time-shift-hours 6 --output-version v20260808"
```

`--daily-time-shift-hours 6` accounts for the daily model's single 06Z sample per day (the
6-hourly model's 0/6/12/18Z daily means are stamped 9Z, hence the default of 9).
