# ACE2.1-ERA5 and ACE2.2-ERA5 AIMIP Inference Outputs

Daily and monthly mean atmospheric output from multidecadal AMIP-style ensemble simulations using the ACE2.1-ERA5 and ACE2.2-ERA5 machine learning atmospheric models, 15 simulations each, contributed to [AIMIP Phase 1](https://github.com/ai2cm/AIMIP).

## Overview

This dataset contains output from ~46-year atmospheric simulations produced by two versions of the Ai2 Climate Emulator, ACE2.1-ERA5 and ACE2.2-ERA5, as contributions to the AI Model Intercomparison Project (AIMIP) Phase 1. AIMIP systematically evaluates AI weather/climate models trained on ERA5 reanalysis by running standardized AMIP-style simulations and comparing their climate statistics against reference observations and conventional models.

For each model, 15 simulations span 5 ensemble members and 3 sea surface temperature (SST) forcing scenarios — historical baseline, +2 K, and +4 K uniform global SST perturbations — covering October 1978 through December 2024. Outputs are CMIP6-compliant NetCDF files at daily and monthly mean temporal resolution on a 1° × 1° global grid.

## Models

### ACE2.1-ERA5

ACE2.1-ERA5 is a version of the [Ai2 Climate Emulator (ACE)](https://github.com/ai2cm/ace) trained on ERA5 reanalysis data. The base model uses a Noise-Conditioned Spherical Fourier Neural Operator (NoiseConditionedSFNO) architecture on a 1° global horizontal grid with 8 native vertical levels. A secondary MLP decoder, added via fine-tuning with the base model frozen, produces the pressure-level diagnostic variables included in this dataset. The base model was trained on ERA5 data from 1979–2008 (training) and 2009–2014 (validation); four random seeds were trained and evaluated, and the best-performing seed was selected for inference.

### ACE2.2-ERA5

ACE2.2-ERA5 (CMIP `source_id` `ACE2-2-ERA5`) is a stochastic model with the same 1° grid, 8 native vertical levels, 6-hourly time step and input variables as ACE2.1-ERA5. It differs from ACE2.1-ERA5 as follows:

- It is trained with a CRPS and energy-score loss and a noise input, instead of a weighted mean squared error loss.
- Its network is wider (embedding dimension 512 instead of 384), with revised module hyperparameters.
- The 2 m temperature and humidity and the 10 m winds are outputs of the main network, but not inputs. ACE2.1-ERA5 predicts them with its secondary decoder.
- It was trained on ERA5 data from 1979–2013 except 1994, with 1994 and 2014 held out for validation.
- Training has three stages: single-step pretraining, multi-step fine-tuning, and fine-tuning of the secondary pressure-level decoder with the main network frozen.

Four random seeds were trained, and one was selected for inference from simulations over 1979–2014.

**Resources:**
- Model code: https://github.com/ai2cm/ace
- Training, inference and postprocessing workflow: https://github.com/ai2cm/ACE2.1-ERA5-AIMIP, archived for ACE2.1-ERA5 at https://doi.org/10.5281/zenodo.19831256 and for ACE2.2-ERA5 at https://doi.org/10.5281/zenodo.24515079
- Paper: Watt-Meyer et al. (2025). "ACE2: Accurately learning subseasonal to decadal atmospheric variability and forced responses." *npj Climate and Atmospheric Science*. https://doi.org/10.1038/s41612-025-01090-0
- Papers related to ACE2.2-ERA5: Perkins et al. (2026). "HiRO-ACE: Fast and skillful AI emulation and downscaling trained on a 3 km global storm-resolving model." https://arxiv.org/abs/2512.18224; McGibbon et al. (2026). "Stochastic climate emulation at reduced cost with ACE2.5-ERA5." https://arxiv.org/abs/2611.28315
- Model checkpoints: [allenai/ACE HuggingFace collection](https://huggingface.co/collections/allenai/ace-67327d822f0f0d8e0e5e6ca4)

## Experiment Design

### AIMIP Phase 1

AIMIP Phase 1 is an intercomparison project comparing AI weather/climate models through standardized multidecadal AMIP-style simulations. Its goals are to (1) systematically compare time-mean climate, trends, and variability across AI models trained on ERA5; (2) produce CMIP7-compatible outputs to enable evaluation by the broader climate science community; and (3) enhance the credibility of AI climate models through structured intercomparison. See the [AIMIP specification](https://github.com/ai2cm/AIMIP) for full details.

### Simulation Setup

| Parameter | Value |
|---|---|
| Simulation period | Oct 1, 1978 – Dec 31, 2024 |
| Spinup period | Oct–Dec 1978 (3 months, excluded from analysis) |
| Analysis period | Jan 1, 1979 – Dec 31, 2024 |
| Model timestep | 6-hourly (internal; outputs are daily and monthly averages) |
| Ensemble members | 5 (r1i1p1f1 – r5i1p1f1) |
| SST scenarios | 3 (baseline, +2K, +4K) |
| Total simulations | 15 per model |

### Forcing Data

Both models are forced with prescribed monthly sea surface temperature (SST) and sea-ice concentration (SIC) derived from ERA5 reanalysis (1978–2024), regridded to the models' native 1° global grid. Three scenarios are provided:

| Experiment ID | Description |
|---|---|
| `aimip` | Historical observed SST and sea-ice concentration (baseline) |
| `aimip-p2k` | Historical SST with a uniform global +2 K perturbation |
| `aimip-p4k` | Historical SST with a uniform global +4 K perturbation |

### Ensemble Members

Five simulations are run per scenario, labeled `r1i1p1f1` through `r5i1p1f1`. ACE2.1-ERA5 is deterministic, so its members are initialized from different atmospheric states on or around October 1, 1978. ACE2.2-ERA5 is stochastic, so its members share one initial state and differ by random seed.

## Variables

All variables follow CMIP6 naming and unit conventions. Files are CF-compliant NetCDF with standard coordinate attributes.

Two frequency tables are available:
- **`Amon`** (monthly means): full simulation period, Oct 1978 – Dec 2024
- **`day`** (daily means): two sub-periods only — Oct 1978–Dec 1979 (spinup assessment) and Jan–Dec 2024 (out-of-sample testing)

### 3D Atmospheric Fields at Standard Pressure Levels (grid label: `gr`)

These variables are provided at standard pressure levels, stacked along a single `plev` dimension:

- ACE2.1-ERA5: 13 levels (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, and 1000 hPa)
- ACE2.2-ERA5: the 7 AIMIP levels (50, 100, 250, 500, 700, 850, and 1000 hPa)

They are produced by a secondary MLP decoder added during fine-tuning. In ACE2.2-ERA5, 850 hPa temperature and 500 hPa geopotential height come from the main network instead.

| Variable | Long name | Units | Frequencies |
|---|---|---|---|
| `ta` | Air temperature | K | `day`, `Amon` |
| `hus` | Specific humidity | kg kg⁻¹ | `day`, `Amon` |
| `ua` | Eastward wind | m s⁻¹ | `day`, `Amon` |
| `va` | Northward wind | m s⁻¹ | `day`, `Amon` |
| `zg` | Geopotential height | m | `day`, `Amon` |

### 3D Atmospheric Fields at Native Model Levels (grid label: `gn`)

These variables are also provided on the models' native 8-level coordinate. The 8 layers are ordered from the top of the atmosphere downward (`model_layer` 0 is the uppermost) and represent the models' internal prognostic state.

| Variable | Long name | Units | Frequencies |
|---|---|---|---|
| `ta` | Air temperature | K | `day`, `Amon` |
| `hus` | Specific humidity | kg kg⁻¹ | `day`, `Amon` |
| `ua` | Eastward wind | m s⁻¹ | `day`, `Amon` |
| `va` | Northward wind | m s⁻¹ | `day`, `Amon` |

Note: `zg` is not available at native model levels.

### Surface / Single-Level Fields (grid label: `gn`)

| Variable | Long name | Units | Frequencies |
|---|---|---|---|
| `ps` | Surface pressure | Pa | `day`, `Amon` |
| `ts` | Surface temperature | K | `day`, `Amon` |
| `tas` | Near-surface (2 m) air temperature | K | `day`, `Amon` |
| `huss` | Near-surface (2 m) specific humidity | kg kg⁻¹ | `day`, `Amon` |
| `uas` | Near-surface (10 m) eastward wind | m s⁻¹ | `day`, `Amon` |
| `vas` | Near-surface (10 m) northward wind | m s⁻¹ | `day`, `Amon` |
| `pr` | Precipitation rate | kg m⁻² s⁻¹ | `day`, `Amon` |

## Data Format and File Organization

### Format

NetCDF4, CF-convention compliant, with CMIP6-standard variable names, units, and coordinate attributes.

### File Naming Convention

Files follow the CMIP6 Data Reference Syntax (DRS). Each model has its own folder, with a nested directory hierarchy inside it:

```
{experiment_id}/{variant_label}/{table_id}/{varname}/{grid_label}/{version}/{filename}.nc
```

where `{filename}` is:

```
{varname}_{table_id}_{source_id}_{experiment_id}_{variant_label}_{grid_label}_{start}-{end}.nc
```

| Model | Folder | `source_id` in file names | `version` |
|---|---|---|---|
| ACE2.1-ERA5 | `ACE2-1-ERA5/` | `ACE2-ERA5` | `v20251130` |
| ACE2.2-ERA5 | `ACE2-2-ERA5/` | `ACE2-2-ERA5` | `v20261107` |

| Field | Values |
|---|---|
| `table_id` | `Amon` (monthly mean) or `day` (daily mean) |
| `experiment_id` | `aimip`, `aimip-p2k`, or `aimip-p4k` |
| `variant_label` | `r1i1p1f1` through `r5i1p1f1` |
| `grid_label` | `gr` (standard pressure levels) or `gn` (native model grid) |
| `start`/`end` | `YYYYMM` format |

**Examples**, relative to each model's folder:
```
# ACE2.1-ERA5: monthly temperature at standard pressure levels (full period)
aimip/r1i1p1f1/Amon/ta/gr/v20251130/ta_Amon_ACE2-ERA5_aimip_r1i1p1f1_gr_197810-202412.nc

# ACE2.2-ERA5: monthly near-surface temperature (full period)
aimip/r1i1p1f1/Amon/tas/gn/v20261107/tas_Amon_ACE2-2-ERA5_aimip_r1i1p1f1_gn_197810-202412.nc

# ACE2.1-ERA5: daily temperature at pressure levels, +2K scenario (spinup sub-period)
aimip-p2k/r3i1p1f1/day/ta/gr/v20251130/ta_day_ACE2-ERA5_aimip-p2k_r3i1p1f1_gr_19781001-19791231.nc
```

### Dataset Size

Approximately 560 GB for ACE2.1-ERA5 and 360 GB for ACE2.2-ERA5 (daily + monthly, 5 ensemble members × 3 SST scenarios each).

## How to Use

The files can be opened with any NetCDF-compatible tool. A minimal Python example using [xarray](https://xarray.dev/), run from the `ACE2-1-ERA5/` folder (for ACE2.2-ERA5, use its folder, `source_id` and `version` from the table above):

```python
import xarray as xr
import glob

# Monthly near-surface temperature (full period, native grid)
ds = xr.open_dataset(
    "aimip/r1i1p1f1/Amon/tas/gn/v20251130/"
    "tas_Amon_ACE2-ERA5_aimip_r1i1p1f1_gn_197810-202412.nc"
)
tas = ds["tas"]  # shape: (time, lat, lon)

# Monthly temperature at standard pressure levels (full period)
# All pressure levels (13 for ACE2.1-ERA5, 7 for ACE2.2-ERA5) are stacked along the plev dimension
ds3d = xr.open_dataset(
    "aimip/r1i1p1f1/Amon/ta/gr/v20251130/"
    "ta_Amon_ACE2-ERA5_aimip_r1i1p1f1_gr_197810-202412.nc"
)
ta = ds3d["ta"]  # shape: (time, plev, lat, lon)
print(ta.plev.values)  # pressure levels in Pa

# Load all 5 ensemble members for the baseline scenario
files = sorted(glob.glob("aimip/r*i1p1f1/Amon/tas/gn/*/tas_*.nc"))
ds_ens = xr.open_mfdataset(files, concat_dim="member", combine="nested")

# Daily data is available for two sub-periods only
ds_day = xr.open_dataset(
    "aimip/r1i1p1f1/day/tas/gn/v20251130/"
    "tas_day_ACE2-ERA5_aimip_r1i1p1f1_gn_19781001-19791231.nc"  # spinup period
)
```

## Generation

The ACE2.1-ERA5 outputs were generated through the following steps:

1. **Training**: The ACE2.1-ERA5 base model was trained on ERA5 reanalysis (1979–2008 training period, 2009–2014 validation) across four random seeds.
2. **Seed selection**: Each trained seed was evaluated across multiple multi-year inference runs; the best-performing checkpoint was selected.
3. **Fine-tuning**: The selected checkpoint's base model weights were frozen, and a secondary MLP decoder was trained to produce the pressure-level diagnostic variables included in this dataset.
4. **Final selection**: Fine-tuned seeds were re-evaluated, and the best checkpoint was selected for inference.
5. **Inference**: 15 parallel 46-year simulations were run (5 initial conditions × 3 SST scenarios), starting October 1, 1978, with a 3-month spinup period.

The ACE2.2-ERA5 outputs were generated through the following steps:

1. **Training**: Four random seeds were trained on ERA5 reanalysis (1979–2013 except 1994 for training, 1994 and 2014 for validation), with single-step pretraining followed by multi-step fine-tuning.
2. **Seed selection**: Each seed was evaluated in simulations over 1979–2014, and one was selected.
3. **Fine-tuning**: With the selected seed's main network frozen, a secondary MLP decoder was trained to produce the pressure-level diagnostic variables.
4. **Inference**: 15 parallel 46-year simulations were run (5 random seeds × 3 SST scenarios) from one initial condition, starting October 1, 1978, with a 3-month spinup period.

For both models, raw 6-hourly inference outputs were subsequently postprocessed to produce the AIMIP-compliant daily and monthly mean files in this dataset, including stacking pressure levels along a `plev` dimension and adding required CF and CMIP6 metadata attributes.

## Citation

If you use this dataset, please cite the ACE2 paper, and for ACE2.2-ERA5 also McGibbon et al. (2026), https://arxiv.org/abs/2611.28315:

```bibtex
@article{watt-meyer2025ace2,
  author={Watt-Meyer, Oliver and others},
  title={{ACE2}: Accurately learning subseasonal to decadal atmospheric variability and forced responses},
  journal={npj Climate and Atmospheric Science},
  year={2025},
  doi={10.1038/s41612-025-01090-0},
  url={https://doi.org/10.1038/s41612-025-01090-0}
}
```

This dataset was generated for the [AIMIP Phase 1 intercomparison](https://github.com/ai2cm/AIMIP), which is described in the AIMIP Phase 1 evaluation paper (https://arxiv.org/abs/2605.06944).

## License

This dataset is licensed under [CC BY-4.0](LICENSE-DATASET). It is intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).

This dataset consists entirely of weather and climate model output derived from the ERA5 reanalysis dataset. It contains no private, personal, or otherwise sensitive data.
