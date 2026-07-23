"""Config-driven covariate streamers for building yearly encoder states.

Generalizes the old hardcoded PRISM+BUI pair (see states.py) into a registry of
named streams, each yielding ``(year, state)`` in lockstep so the combine step
can assemble any covariate set. Two generic streamers cover the new products:

- ``PerVariableYearStreamer`` — a set of variables each stored as
  ``{var}_{year}_grid.tif`` (what preprocess/luh3.py and hyde.py write). Per
  year it stacks the variables into an (H, W, n_var) state, filling each variable
  from its nearest available year (so sparse HYDE time points and annual LUH-3
  interoperate), then EMA-smooths across years. This is the land-use stream.
- ``StaticStreamer`` — time-invariant rasters (SoilGrids); stacks them once and
  yields the same state every year (no EMA). This is the soil stream.

The monthly bio-year climate stream still lives in states.py (PrismStreamer);
its continental replacement (climr output) plugs in here as another streamer once
that acquire step has produced grid rasters.
"""
import functools
import glob
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import rasterio

from src.processing import regrid


def ema_alpha(tau):
    """EMA weight for a time constant of ``tau`` years (matches states.py)."""
    return 1.0 - np.exp(-1.0 / tau)


@functools.lru_cache(maxsize=None)
def _read_grid(path):
    """Read a single-band grid raster as (H, W) float32 with nodata -> NaN.

    Cached: nearest-year fill re-requests the SAME file across many years (every
    warmup year maps to the earliest available; decadal products repeat within a
    decade), so without caching one raster is re-read from Lustre dozens of times.
    The array is returned read-only (shared cache entry) -- callers np.stack/EMA
    into fresh arrays and never mutate it in place."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype(np.float32)
    out = arr.filled(np.nan)
    out.flags.writeable = False
    return out


def _write_state_npz(path, arrays):
    """Compress + write one year's per-stream arrays (runs in a worker process)."""
    np.savez_compressed(path, **arrays)
    return path


def _prewarm_reads(paths, workers):
    """Read all unique rasters concurrently to warm the ``_read_grid`` cache.

    The per-year EMA loop is sequential and reads ~10k small GeoTIFFs from Lustre;
    done serially that is the whole runtime (single core, ~30-50 ms per open). A
    thread pool parallelizes the opens (rasterio releases the GIL during read), so
    the subsequent sequential loop hits a warm cache and the reads no longer bound
    the wall clock."""
    paths = list(paths)
    if not paths or workers <= 1:
        for p in paths:
            _read_grid(p)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_read_grid, paths))


class _EmaStreamer:
    """Base: iterate model years, EMA-smoothing whatever ``_year_state`` returns.

    ``_year_state(year)`` returns the raw (H, W, C) stack for that year or None
    (no data yet). The EMA state carries forward across None years, exactly like
    the original PRISM/BUI streamers.
    """

    def __init__(self, start_year, end_year, alpha):
        self.years = range(start_year, end_year + 1)
        self.alpha = alpha
        self.state = None

    def _year_state(self, year):
        raise NotImplementedError

    def __iter__(self):
        for year in self.years:
            curr = self._year_state(year)
            if curr is not None:
                self.state = curr if self.state is None else \
                    self.alpha * curr + (1.0 - self.alpha) * self.state
            yield year, self.state


class PerVariableYearStreamer(_EmaStreamer):
    """Stack ``{var}_{year}_grid.tif`` variables per year, nearest-year fill + EMA."""

    def __init__(self, grid_dir, variables, start_year, end_year, alpha,
                 name="stream"):
        super().__init__(start_year, end_year, alpha)
        self.grid_dir = grid_dir
        self.variables = list(variables)
        self.name = name
        # Per variable, map available year -> file path.
        self.avail = {}
        for var in self.variables:
            pat = re.compile(rf"^{re.escape(var)}_(\d{{4}})_grid\.tif$")
            yrs = {}
            for f in glob.glob(os.path.join(grid_dir, f"{var}_*_grid.tif")):
                m = pat.match(os.path.basename(f))
                if m:
                    yrs[int(m.group(1))] = f
            if not yrs:
                raise FileNotFoundError(
                    f"[{name}] no {var}_*_grid.tif in {grid_dir}")
            self.avail[var] = yrs
        print(f"[{name}] {len(self.variables)} vars, "
              f"{{{min(min(v) for v in self.avail.values())}.."
              f"{max(max(v) for v in self.avail.values())}}} yrs -> {grid_dir}")

    def _nearest_year(self, years_available, year):
        """Nearest available year (ties -> the earlier/past year, no peeking bias)."""
        return min(years_available, key=lambda y: (abs(y - year), y > year))

    def _year_state(self, year):
        bands = []
        for var in self.variables:
            yrs = self.avail[var]
            path = yrs.get(year) or yrs[self._nearest_year(yrs.keys(), year)]
            bands.append(_read_grid(path))
        return np.stack(bands, axis=-1)

    def prefetch_paths(self, years):
        """Every distinct raster this streamer will read over ``years`` (after
        nearest-year fill) -- for parallel cache pre-warming."""
        out = set()
        for var in self.variables:
            yrs = self.avail[var]
            for y in years:
                out.add(yrs.get(y) or yrs[self._nearest_year(yrs.keys(), y)])
        return out


class StaticStreamer:
    """Time-invariant stream: stack rasters once, yield the same state every year."""

    def __init__(self, paths, start_year, end_year, name="static"):
        self.years = range(start_year, end_year + 1)
        self.name = name
        if not paths:
            raise FileNotFoundError(f"[{name}] no rasters given")
        self.state = np.stack([_read_grid(p) for p in paths], axis=-1)
        print(f"[{name}] {self.state.shape[-1]} static bands, shape {self.state.shape}")

    def __iter__(self):
        for year in self.years:
            yield year, self.state


def static_paths(grid_dir, suffix="_grid.tif"):
    """Sorted list of static rasters in a dir (deterministic band order)."""
    return sorted(glob.glob(os.path.join(grid_dir, f"*{suffix}")))


def build_streamer(spec, start_year, end_year):
    """Construct one streamer from a config spec.

    spec keys: ``type`` ("per_variable" | "static"), ``name``, ``grid_dir``,
    and for per_variable: ``variables``, ``ema_tau`` (default 10).
    """
    stype = spec["type"]
    name = spec.get("name", stype)
    if stype == "per_variable":
        alpha = ema_alpha(spec.get("ema_tau", 10.0))
        return PerVariableYearStreamer(spec["grid_dir"], spec["variables"],
                                       start_year, end_year, alpha, name=name)
    if stype == "static":
        paths = spec.get("paths") or static_paths(spec["grid_dir"])
        return StaticStreamer(paths, start_year, end_year, name=name)
    raise ValueError(f"unknown stream type: {stype!r}")


def run_states(specs, out_dir, start_year, end_year, mask, sample_start,
               samples_per_year=20000, rng=None, write_workers=None, read_workers=None):
    """Lockstep-iterate all streams; write per-year npz + a training-vector bag.

    ``mask`` is a boolean (H, W) land mask (True = sample here). Each per-year npz
    holds one array per stream (named by ``spec['name']``); the training bag
    concatenates per-stream pixel vectors, and an offsets dict records each
    stream's channel slice. A ``state_schema.json`` sidecar persists those offsets
    (+ per-stream dims/variables) so consumers can split ``history_vectors.npy``
    and normalize per stream without re-deriving the layout. Returns (bag, offsets).

    The EMA is sequential (cheap arithmetic), but the two costs -- reading rasters
    (cached; see ``_read_grid``) and the per-year zlib compression -- are not: the
    per-year ``savez_compressed`` runs in a ``write_workers`` process pool while the
    main loop keeps computing the next year, overlapping compute with compression.
    ``write_workers`` defaults to ~cpu count (capped 8); pass 1 to write serially.
    """
    os.makedirs(out_dir, exist_ok=True)
    states_dir = os.path.join(out_dir, "yearly_states")
    os.makedirs(states_dir, exist_ok=True)
    rng = rng or np.random.default_rng()
    write_workers = write_workers if write_workers is not None else min(os.cpu_count() or 1, 8)
    # Reads are I/O-bound (Lustre small-file opens), so oversubscribe cores.
    read_workers = read_workers if read_workers is not None else min((os.cpu_count() or 1) * 2, 32)

    names = [s.get("name", s["type"]) for s in specs]
    streamers = [build_streamer(s, start_year, end_year) for s in specs]
    valid_rows, valid_cols = np.where(mask)
    n_valid = len(valid_rows)
    print(f"[states] {n_valid} valid cells; streams: {names}; "
          f"read_workers={read_workers} write_workers={write_workers}")

    # Warm the read cache in parallel before the sequential EMA loop, so ~10k small
    # Lustre opens don't serialize on one core (the dominant cost otherwise).
    all_years = range(start_year, end_year + 1)
    prefetch = set()
    for s in streamers:
        if hasattr(s, "prefetch_paths"):
            prefetch |= s.prefetch_paths(all_years)
    if prefetch:
        print(f"[states] pre-reading {len(prefetch)} unique rasters ({read_workers} threads)...",
              flush=True)
        _prewarm_reads(prefetch, read_workers)

    # fork: no open rasterio handles or threads in the parent at pool creation
    # (reads use `with rasterio.open`), and fork avoids re-importing __main__.
    ex = None
    if write_workers > 1:
        try:
            ex = ProcessPoolExecutor(
                max_workers=write_workers,
                mp_context=multiprocessing.get_context("fork"),
            )
        except (NotImplementedError, PermissionError, OSError) as exc:
            # Restricted macOS/container environments may deny POSIX semaphore
            # discovery. Compression is an optimization, not a correctness
            # requirement, so degrade explicitly to serial writes.
            print(f"[states] process writer unavailable ({exc}); writing serially")
    pending = []                                         # bounded in-flight write futures

    def _submit_write(year, arrays):
        path = os.path.join(states_dir, f"state_{year}.npz")
        if ex is None:
            _write_state_npz(path, arrays)
            return
        pending.append(ex.submit(_write_state_npz, path, arrays))
        while len(pending) >= 2 * write_workers:         # backpressure: cap memory in flight
            pending.pop(0).result()

    bag, offsets = [], None
    try:
        for tick in zip(*streamers):
            years = {y for y, _ in tick}
            assert len(years) == 1, f"stream desync: {years}"
            year = years.pop()
            states = {name: s for name, (_, s) in zip(names, tick)}
            if any(s is None for s in states.values()) or year < sample_start:
                continue

            # Deterministic per-year sample of valid cells.
            k = min(samples_per_year, n_valid)
            idx = rng.choice(n_valid, k, replace=False)
            r, c = valid_rows[idx], valid_cols[idx]

            vecs, offs, pos = [], {}, 0
            for name in names:
                v = states[name][r, c]                   # (k, C)
                offs[name] = (pos, pos + v.shape[1])
                pos += v.shape[1]
                vecs.append(v)
            combined = np.concatenate(vecs, axis=1)
            keep = ~np.isnan(combined).any(axis=1)       # strict NaN filter
            if keep.any():
                bag.append(combined[keep])
            offsets = offs

            # A per-year EMA array is a fresh object (reassigned, never mutated), and
            # static arrays are read-only, so it is safe to hand them to a worker
            # while the loop advances.
            _submit_write(year, {name: states[name] for name in names})

        for f in pending:                                # drain remaining writes
            f.result()
    finally:
        if ex is not None:
            ex.shutdown()

    if offsets is not None:
        spec_by_name = {s.get("name", s["type"]): s for s in specs}
        schema = {
            "streams": [
                {
                    "name": name,
                    "start": int(offsets[name][0]),
                    "end": int(offsets[name][1]),
                    "dim": int(offsets[name][1] - offsets[name][0]),
                    "type": spec_by_name[name]["type"],
                    "variables": list(spec_by_name[name].get("variables", [])),
                }
                for name in names
            ],
            "total_dim": int(offsets[names[-1]][1]),
            "start_year": int(start_year),
            "end_year": int(end_year),
            "sample_start": int(sample_start),
        }
        with open(os.path.join(out_dir, "state_schema.json"), "w") as fh:
            json.dump(schema, fh, indent=2)

    if bag:
        full = np.concatenate(bag, axis=0).astype(np.float32)
        np.save(os.path.join(out_dir, "history_vectors.npy"), full)
        print(f"[states] saved {len(full)} training vectors; offsets={offsets}")
    else:
        print("[states] WARNING: no vectors sampled")
    return (np.concatenate(bag, axis=0) if bag else np.empty((0, 0))), offsets
