"""Driver: assemble per-year N-stream encoder states via ``streams.run_states``.

Wires the (previously caller-less) generic streamer registry into the pipeline.
Reads a ``states`` block from ``esk_desk_config.json`` listing covariate streams
(climate / land-use / HYDE as per-variable EMA streams; soil / elevation as static
streams), resolves each stream's variables/paths from disk, and writes
``<hist_dir>/yearly_states/state_{year}.npz`` (+ ``history_vectors.npy`` and the
``state_schema.json`` sidecar) for the contiguous model timeline. An EMA burn-in
of ``warmup`` years before ``first_year`` primes the smoothing without being
written (``run_states`` skips years < ``sample_start``).

    python -m src.data.combine.build_states
"""
import argparse
import glob
import os
import re

import numpy as np

from src.config_utils import load_config, load_data_config
from src.data.combine import streams
from src.data.preprocess.bbs import load_grid_reference
from src.temporal import load_timeline

_YEAR_TIF = re.compile(r"^(?P<var>.+)_(?P<year>\d{4})_grid\.tif$")


def discover_variables(grid_dir, level=None):
    """Distinct ``{var}`` tokens from ``{var}_{year}_grid.tif`` files in a dir.

    With ``level`` set (climate q10/q50/q90), keep only vars ending ``_{level}``.
    """
    vars_found = set()
    for f in glob.glob(os.path.join(grid_dir, "*_????_grid.tif")):
        m = _YEAR_TIF.match(os.path.basename(f))
        if m:
            vars_found.add(m.group("var"))
    if level:
        vars_found = {v for v in vars_found if v.endswith(f"_{level}")}
    return sorted(vars_found)


def resolve_spec(spec):
    """Fill in a stream spec's ``variables`` (per_variable) or ``paths`` (static)
    from disk when not given explicitly. Returns the spec (mutated copy)."""
    spec = dict(spec)
    grid_dir = spec["grid_dir"]
    if spec["type"] == "per_variable":
        if not spec.get("variables"):
            spec["variables"] = discover_variables(grid_dir, spec.get("level"))
            if not spec["variables"]:
                raise FileNotFoundError(
                    f"[{spec.get('name')}] no {{var}}_{{year}}_grid.tif in {grid_dir}"
                    f"{' for level ' + spec['level'] if spec.get('level') else ''}")
    elif spec["type"] == "static":
        if not spec.get("paths"):
            spec["paths"] = streams.static_paths(grid_dir, spec.get("suffix", "_grid.tif"))
            if not spec["paths"]:
                raise FileNotFoundError(f"[{spec.get('name')}] no static rasters in {grid_dir}")
    return spec


def default_specs(dr):
    """Fallback stream specs keyed to the standard preprocess output dirs."""
    return [
        {"name": "climate", "type": "per_variable",
         "grid_dir": os.path.join(dr, "climate_grid"), "ema_tau": 10},
        {"name": "landuse", "type": "per_variable",
         "grid_dir": os.path.join(dr, "luh3_grid"), "ema_tau": 10},
        {"name": "hyde", "type": "per_variable",
         "grid_dir": os.path.join(dr, "hyde35_grid"), "ema_tau": 10},
        {"name": "soil", "type": "static",
         "grid_dir": os.path.join(dr, "soilgrids_grid")},
        {"name": "elevation", "type": "static",
         "grid_dir": os.path.join(dr, "elevation"), "suffix": ".tif"},
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="states out dir (default: config paths.hist_dir)")
    ap.add_argument("--warmup", type=int, default=None, help="EMA burn-in years before first_year")
    ap.add_argument("--samples-per-year", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-workers", type=int, default=None,
                    help="processes compressing per-year npz in parallel (default ~cpu, cap 8; 1=serial)")
    ap.add_argument("--read-workers", type=int, default=None,
                    help="threads pre-reading rasters (I/O-bound; default 2*cpu, cap 32; 1=serial)")
    args = ap.parse_args()

    cfg = load_config()
    dcfg = load_data_config()
    dr = dcfg["datasets_root"]
    scfg = cfg.get("states", {})
    specs = [resolve_spec(s) for s in (scfg.get("streams") or default_specs(dr))]
    warmup = args.warmup if args.warmup is not None else int(scfg.get("warmup", 20))
    out = args.out or cfg["paths"]["hist_dir"]

    tl = load_timeline()
    first_year, end_year = tl["first_year"], tl["end_year"]

    mask_path = cfg.get("latent_cube", {}).get("water_mask_path") \
        or os.path.join(dr, "land_mask", "ocean_mask_25km.tif")
    land_mask, _, _, _, nx, ny = load_grid_reference(mask_path)

    print(f"[build_states] {len(specs)} streams -> {out}; "
          f"years {first_year - warmup}..{end_year} (sample from {first_year})", flush=True)
    streams.run_states(
        specs, out_dir=out,
        start_year=first_year - warmup, end_year=end_year,
        mask=land_mask, sample_start=first_year,
        samples_per_year=args.samples_per_year,
        rng=np.random.default_rng(args.seed),
        write_workers=args.write_workers,
        read_workers=args.read_workers,
    )
    print(f"[build_states] done -> {out}/yearly_states + state_schema.json", flush=True)


if __name__ == "__main__":
    main()
