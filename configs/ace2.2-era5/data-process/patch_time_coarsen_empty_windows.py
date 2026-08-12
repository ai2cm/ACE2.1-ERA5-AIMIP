"""Patch scripts/data_process/time_coarsen.py for the pressure-level build.

Two fixes against ai2cm/ace 632ca493e (both still present on main, 2026-08-05):

1. window_names: [] — coarsen() unconditionally calls .coarsen(time=...) on
   ds[config.window_names]; with no window variables that selection has no
   time dimension and xarray raises. The pressure-level zarr contains only
   snapshot variables, so the window branch must be skipped.

2. Source-store encoding — the 2025-11-10 pressure-level zarr is v2-format
   with numcodecs Blosc compressors; those ride along in .encoding and
   zarr-python 3 rejects them when writing the v3 output store ("Expected a
   BytesBytesCodec"). Clear inherited encodings so zarr picks its defaults.

Run from the ace repo root before time_coarsen.py.
"""

PATH = "scripts/data_process/time_coarsen.py"

OLD = """    ds_window = (
        ds[config.window_names]
        .coarsen(time=config.factor, boundary="trim")
        .mean()
        .drop("time")
    )  # use time of snapshots
    ds_coarsened = xr.merge([ds_snapshot, ds_window, ds_constants])"""

NEW = """    if config.window_names:
        ds_window = (
            ds[config.window_names]
            .coarsen(time=config.factor, boundary="trim")
            .mean()
            .drop("time")
        )  # use time of snapshots
        ds_coarsened = xr.merge([ds_snapshot, ds_window, ds_constants])
    else:
        ds_coarsened = xr.merge([ds_snapshot, ds_constants])"""

OLD_ENC = """    ds_coarsened = coarsen(ds, config)
    if not dry_run:"""

NEW_ENC = """    ds_coarsened = coarsen(ds, config)
    for _name in list(ds_coarsened.variables):
        ds_coarsened[_name].encoding = {}
    if not dry_run:"""

src = open(PATH).read()
if NEW in src and NEW_ENC in src:
    print("time_coarsen.py already patched")
else:
    assert src.count(OLD) == 1, "unexpected time_coarsen.py contents (windows)"
    assert src.count(OLD_ENC) == 1, "unexpected time_coarsen.py contents (encoding)"
    src = src.replace(OLD, NEW).replace(OLD_ENC, NEW_ENC)
    open(PATH, "w").write(src)
    print("patched time_coarsen.py (empty window_names + encoding reset)")
