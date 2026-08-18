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


def load_grid_manifest(grid_dir):
    """A grid dir's ``manifest.json`` (authoritative channel order), or None.

    ``climate_grid_monthly`` and ``bui_grid`` write one; LUH-3/HYDE dirs have none
    and fall back to sorted-glob discovery.
    """
    path = os.path.join(grid_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    import json
    with open(path) as fh:
        return json.load(fh)


def discover_variables(grid_dir, level=None):
    """Distinct ``{var}`` tokens from ``{var}_{year}_grid.tif`` files in a dir.

    With ``level`` set (climate q10/q50/q90), keep only vars ending ``_{level}``.

    When the dir carries a ``manifest.json``, its ``variables`` list is the
    authoritative ORDER and is validated against what is actually on disk. This
    matters because channel identity is otherwise positional: the returned order
    becomes the channel order in ``state_*.npz``, and the per-channel ``mu``/``sd``
    saved in ``desk_meta.npz`` are indexed by that position. A stray or missing
    raster would silently renumber every channel after it, applying the wrong
    normalization to the wrong variable with no error raised anywhere.
    """
    vars_found = set()
    for f in glob.glob(os.path.join(grid_dir, "*_????_grid.tif")):
        m = _YEAR_TIF.match(os.path.basename(f))
        if m:
            vars_found.add(m.group("var"))
    if level:
        vars_found = {v for v in vars_found if v.endswith(f"_{level}")}

    manifest = load_grid_manifest(grid_dir)
    if manifest and manifest.get("variables"):
        expected = [v for v in manifest["variables"]
                    if not level or v.endswith(f"_{level}")]
        missing = sorted(set(expected) - vars_found)
        extra = sorted(vars_found - set(expected))
        if missing or extra:
            raise SystemExit(
                f"{grid_dir} disagrees with its manifest.json"
                + (f"\n  missing from disk ({len(missing)}): {missing[:8]}" if missing else "")
                + (f"\n  not in manifest ({len(extra)}): {extra[:8]}" if extra else "")
                + "\nChannel order is positional and feeds the saved mu/sd, so this "
                  "is refused rather than silently reindexed. Re-run "
                  "src.data.preprocess.climate_grid, or remove the stray rasters.")
        return expected
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
    ap.add_argument("--read-chunk-years", type=int, default=8,
                    help="years pre-read per batch; caps resident rasters (default 8). "
                         "Lower on tight memory, raise for fewer prefetch stalls")
    args = ap.parse_args()

    cfg = load_config()
    dcfg = load_data_config()
    dr = dcfg["datasets_root"]
    scfg = cfg.get("states", {})
    # states.streams is REQUIRED, with no built-in fallback. There used to be a
    # default_specs() here, and it had gone stale in the two ways that matter most: it
    # named climate_grid (the annual scheme-v1 dir that climate_grid.assert_no_legacy_rasters
    # now refuses to write into) instead of climate_grid_monthly, and it set ema_tau=10 where
    # the config sets 2. A stale fallback for the covariate set is the one kind of residue
    # that produces wrong NUMBERS rather than an error, so it fails loudly instead.
    if not scfg.get("streams"):
        raise SystemExit(
            "states.streams is missing from the ESK/DESK config. It is the authoritative "
            "covariate-stream registry (name/type/grid_dir/ema_tau per stream); there is no "
            "default, because a default that drifts from the config would silently build a "
            "different covariate set than the one the encoder was normalized against.")
    specs = [resolve_spec(s) for s in scfg["streams"]]
    warmup = args.warmup if args.warmup is not None else int(scfg.get("warmup", 20))
    out = args.out or cfg["paths"]["hist_dir"]

    tl = load_timeline()
    first_year, end_year = tl["first_year"], tl["end_year"]

    # Grid geometry belongs to data_config.json; ``cfg`` is the separate
    # ESK/DESK config and intentionally has no top-level ``grid`` block.
    res_km = dcfg["grid"]["target_res_m"] // 1000
    mask_path = cfg.get("latent_cube", {}).get("water_mask_path") \
        or os.path.join(dr, "land_mask", f"ocean_mask_{res_km}km.tif")
    land_mask, _, _, _, nx, ny = load_grid_reference(mask_path)

    print(f"[build_states] {len(specs)} streams -> {out}; "
          f"years {first_year - warmup}..{end_year} (sample from {first_year})", flush=True)
    # Log each stream's resolved width and channel-order endpoints. Channel count
    # is a silent breaking change for any saved mu/sd or trained checkpoint, so it
    # must be visible in the job log rather than only inside state_schema.json.
    total_dim = 0
    for spec in specs:
        if spec["type"] == "per_variable":
            v = spec["variables"]
            total_dim += len(v)
            print(f"[build_states]   {spec['name']:9s} {len(v):4d} ch  "
                  f"[{v[0]} .. {v[-1]}]", flush=True)
        else:
            n = len(spec["paths"])
            total_dim += n
            print(f"[build_states]   {spec['name']:9s} {n:4d} ch  (static)", flush=True)
    print(f"[build_states]   {'TOTAL':9s} {total_dim:4d} ch", flush=True)
    streams.run_states(
        specs, out_dir=out,
        start_year=first_year - warmup, end_year=end_year,
        mask=land_mask, sample_start=first_year,
        samples_per_year=args.samples_per_year,
        rng=np.random.default_rng(args.seed),
        write_workers=args.write_workers,
        read_workers=args.read_workers,
        read_chunk_years=args.read_chunk_years,
    )
    print(f"[build_states] done -> {out}/yearly_states + state_schema.json", flush=True)


if __name__ == "__main__":
    main()
