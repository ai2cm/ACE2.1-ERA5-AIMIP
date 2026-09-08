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
| `--skip-gcs-upload` | Write locally only; `--local-dir` is the output and is not deleted |

Output files follow the CMIP6 Data Reference Syntax:

```
{local_dir}/{experiment_id}/{variant_label}/{table_id}/{varname}/{grid_label}/{version}/{filename}.nc
```

Run `make test-postprocess` to execute the unit test suite for the postprocessing helpers.

## ACE2.2-ERA5

A second evaluated checkpoint, **ACE2.2-ERA5** (CMIP `source_id` `ACE2-2-ERA5`), runs the same
inference/postprocessing workflow for a revised recipe: 1° 6-hourly timestep, v2 ERA5-only
training with revised normalization and module hyperparameters, non-residual prediction, and no
CO₂ input. Configs are in `configs/ace2.2-era5/`, launchers in `scripts/run-ace2.2-6h-*.sh`.

Training replaces stages 1–4 above with three stages, whose configs are kept here under
`configs/ace2.2-era5/training/` and launched from `scripts/run-ace2.2-6h-train.sh`: pretrain
(1-step), multi-step fine-tune, and pressure-level decoder fine-tune. Each stage mounts the
previous stage's beaker result dataset, declared in a `# arg:` header in its config.

The ACE2.2 launchers pin their own deps-only beaker image in
`configs/ace2.2-era5/deps_only_image.txt`; the ACE2.1 launchers keep the repo-root
`latest_deps_only_image.txt`. Both models stay reproducible against the image they were
actually run under.

ACE2.2 is stochastic, so the five AIMIP realizations share one initial condition and differ only
by `seed`; ACE2.1 needed lagged ICs because it was deterministic.

### Run inference

```bash
bash scripts/run-ace2.2-6h-inference-smoke.sh
bash scripts/run-ace2.2-6h-inference.sh
```

15 jobs (5 realizations × baseline/+2K/+4K), 67576 6-hourly steps (1978-09-30T18Z → 2024-12-31T18Z).
Run the smoke job first and confirm the raw output carries every `files.yaml` entry and all 7
pressure levels. Before postprocessing the full sweep, check the five realizations actually differ
— identical global-mean `tas` means the seeding did not take, and nothing downstream flags a
zero-spread ensemble.

### Postprocess

**Copy the raw results to local disk first.** `postprocess.py` reads each file through fsspec,
which streams from GCS single-threaded at ~5 MB/s -- around 150 s per file against ~3 s from
local disk, and there are 56 raw files per member. `--raw-results-dir` takes a local path.

```bash
bash scripts/run-ace2.2-6h-mirror-raw-results.sh   # weka -> GCS (postprocess.py runs locally)
gcloud storage cp -r "gs://vcm-ml-intermediate/<results-name>/<job-name>"-r{1..5} /scratch/raw/
```

Then run one process per simulation, five in parallel, writing straight into the evaluation
repo's `local_data` so no upload round trip is needed:

```bash
make postprocess ARGS="--raw-results-dir /scratch/raw --simulation <job-name>-r1 \
    --simulations-file simulations-ace2.2-6h.yaml \
    --model-source-name ACE2-2-ERA5 \
    --source-description 'ACE2-2-ERA5: ACE (Ai2 climate emulator) version 2.2 trained on ERA5' \
    --output-version vYYYYMMDD --skip-gcs-upload \
    --local-dir /path/to/AIMIP/local_data/Ai2/ACE2-2-ERA5/"
```

Parallel runs are safe **only** with `--skip-gcs-upload`, which is also what leaves the output
in place: otherwise each process deletes the shared `--local-dir` once its own upload finishes.
It further avoids the upload's nesting behaviour, since `gsutil cp -r` writes the first
simulation's tree at the destination root and every later one under an extra `aimip-ace/`
level; archive the finished tree in one `gcloud storage rsync -r` pass instead.

Confirm 48 files per member before evaluating -- `postprocess.py` exiting 0 does not mean the
output is complete.

`--daily-time-shift-hours` keeps its default of 9 (the 0/6/12/18Z daily mean is stamped 9Z).
