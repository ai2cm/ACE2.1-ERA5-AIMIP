# ACE2.1-ERA5 AIMIP Inference Outputs

Daily and monthly mean atmospheric output from 15 multidecadal AMIP-style ensemble simulations using the ACE2.1-ERA5 machine learning atmospheric model, contributed to [AIMIP Phase 1](https://github.com/ai2cm/AIMIP).

## Overview

This dataset contains output from 15 parallel ~46-year atmospheric simulations produced by the ACE2.1-ERA5 model as a contribution to the AI Model Intercomparison Project (AIMIP) Phase 1. AIMIP systematically evaluates AI weather/climate models trained on ERA5 reanalysis by running standardized AMIP-style simulations and comparing their climate statistics against reference observations and conventional models.

The 15 simulations span 5 ensemble members (different initial atmospheric conditions) and 3 sea surface temperature (SST) forcing scenarios — historical baseline, +2 K, and +4 K uniform global SST perturbations — covering October 1978 through December 2024. Outputs are CMIP6-compliant NetCDF files at daily and monthly mean temporal resolution on a 1° × 1° global grid.

## Model

ACE2.1-ERA5 is a version of the [Ai2 Climate Emulator (ACE)](https://github.com/ai2cm/ace) trained on ERA5 reanalysis data. The base model uses a Noise-Conditioned Spherical Fourier Neural Operator (NoiseConditionedSFNO) architecture on a 1° global horizontal grid with 8 native vertical levels. A secondary MLP decoder, added via fine-tuning with the base model frozen, produces the pressure-level diagnostic variables included in this dataset. The base model was trained on ERA5 data from 1979–2008 (training) and 2009–2014 (validation); four random seeds were trained and evaluated, and the best-performing seed was selected for inference.

**Resources:**
- Code: https://github.com/ai2cm/ace
- Paper: Watt-Meyer et al. (2025). "ACE2: Accurately learning subseasonal to decadal atmospheric variability and forced responses." *npj Climate and Atmospheric Science*. https://doi.org/10.1038/s41612-025-01090-0
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
| Total simulations | 15 |

### Forcing Data

The model is forced with prescribed monthly sea surface temperature (SST) and sea-ice concentration (SIC) derived from ERA5 reanalysis (1978–2024), regridded to the ACE2.1-ERA5 native 1° global grid. Three scenarios are provided:

| Experiment ID | Description |
|---|---|
| `aimip` | Historical observed SST and sea-ice concentration (baseline) |
| `aimip-p2k` | Historical SST with a uniform global +2 K perturbation |
| `aimip-p4k` | Historical SST with a uniform global +4 K perturbation |

### Ensemble Members

Five simulations are run per scenario, each initialized from a different atmospheric state on or around October 1, 1978. Ensemble members are labeled `r1i1p1f1` through `r5i1p1f1`.

## Variables

All variables follow CMIP6 naming and unit conventions. Files are CF-compliant NetCDF with standard coordinate attributes.

Two frequency tables are available:
- **`Amon`** (monthly means): full simulation period, Oct 1978 – Dec 2024
- **`day`** (daily means): two sub-periods only — Oct 1978–Dec 1979 (spinup assessment) and Jan–Dec 2024 (out-of-sample testing)

### 3D Atmospheric Fields at Standard Pressure Levels (grid label: `gr`)

These variables are provided at 13 standard pressure levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, and 1000 hPa, stacked along a single `plev` dimension. They are produced by a secondary MLP decoder added during fine-tuning.

| Variable | Long name | Units | Frequencies |
|---|---|---|---|
| `ta` | Air temperature | K | `day`, `Amon` |
| `hus` | Specific humidity | kg kg⁻¹ | `day`, `Amon` |
| `ua` | Eastward wind | m s⁻¹ | `day`, `Amon` |
| `va` | Northward wind | m s⁻¹ | `day`, `Amon` |
| `zg` | Geopotential height | m | `day`, `Amon` |

### 3D Atmospheric Fields at Native Model Levels (grid label: `gn`)

These variables are also provided on the ACE2.1-ERA5 native 8-level model coordinate. The 8 levels are ordered from the surface upward and represent the model's internal prognostic state.

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

Files follow the CMIP6 Data Reference Syntax (DRS), organized in a nested directory hierarchy:

```
{experiment_id}/{variant_label}/{table_id}/{varname}/{grid_label}/{version}/{filename}.nc
```

where `{filename}` is:

```
{varname}_{table_id}_ACE2-ERA5_{experiment_id}_{variant_label}_{grid_label}_{start}-{end}.nc
```

| Field | Values |
|---|---|
| `table_id` | `Amon` (monthly mean) or `day` (daily mean) |
| `experiment_id` | `aimip`, `aimip-p2k`, or `aimip-p4k` |
| `variant_label` | `r1i1p1f1` through `r5i1p1f1` |
| `grid_label` | `gr` (standard pressure levels) or `gn` (native model grid) |
| `version` | e.g., `v20251130` |
| `start`/`end` | `YYYYMM` format |

**Examples:**
```
# Monthly temperature at standard pressure levels (full period)
aimip/r1i1p1f1/Amon/ta/gr/v20251130/ta_Amon_ACE2-ERA5_aimip_r1i1p1f1_gr_197810-202412.nc

# Monthly near-surface temperature (full period)
aimip/r1i1p1f1/Amon/tas/gn/v20251130/tas_Amon_ACE2-ERA5_aimip_r1i1p1f1_gn_197810-202412.nc

# Daily temperature at pressure levels, +2K scenario (spinup sub-period)
aimip-p2k/r3i1p1f1/day/ta/gr/v20251130/ta_day_ACE2-ERA5_aimip-p2k_r3i1p1f1_gr_19781001-19791231.nc
```

### Dataset Size

Approximately 500 GB total (daily + monthly, 5 ensemble members × 3 SST scenarios).

## How to Use

The files can be opened with any NetCDF-compatible tool. A minimal Python example using [xarray](https://xarray.dev/):

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
# All 13 pressure levels are stacked along the plev dimension
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

The dataset was generated through the following steps:

1. **Training**: The ACE2.1-ERA5 base model was trained on ERA5 reanalysis (1979–2008 training period, 2009–2014 validation) across four random seeds.
2. **Seed selection**: Each trained seed was evaluated across multiple multi-year inference runs; the best-performing checkpoint was selected.
3. **Fine-tuning**: The selected checkpoint's base model weights were frozen, and a secondary MLP decoder was trained to produce the pressure-level diagnostic variables included in this dataset.
4. **Final selection**: Fine-tuned seeds were re-evaluated, and the best checkpoint was selected for inference.
5. **Inference**: 15 parallel 46-year simulations were run (5 initial conditions × 3 SST scenarios), starting October 1, 1978, with a 3-month spinup period.

Raw 6-hourly inference outputs were subsequently postprocessed to produce the AIMIP-compliant daily and monthly mean files in this dataset, including stacking pressure levels along a `plev` dimension and adding required CF and CMIP6 metadata attributes.

## Citation

If you use this dataset, please cite the ACE2 paper:

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

This dataset was generated for the [AIMIP Phase 1 intercomparison](https://github.com/ai2cm/AIMIP). An AIMIP paper is forthcoming; this citation will be updated when it is available.

## License

This dataset is licensed under [CC BY-4.0](LICENSE-DATASET). It is intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).

This dataset consists entirely of weather and climate model output derived from the ERA5 reanalysis dataset. It contains no private, personal, or otherwise sensitive data.
