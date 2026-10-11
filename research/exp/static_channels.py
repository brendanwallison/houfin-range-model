"""E038: run any DESK pipeline command with some covariate channels held STATIC -- each cell's own mean over a span of
years, in every year -- so they still describe WHERE a place is but cannot drive change through time.

    python research/exp/static_channels.py --channels temperature [--span 1966 2025] -- scripts/run_encoder.py desk
    python research/exp/static_channels.py --channels temperature -- research/exp/cache.py --out <dir>

Works by replacing covariate_io.load_state_stack for this process before the command runs (every DESK and validation
path loads states through it), so no core code changes. Families: temperature (Tave/Tmax/Tmin, degree-days, frost-free
days, reference evaporation), moisture (precipitation, moisture deficit/index, humidity, snow), climate (both), luh,
hyde, bui. Masks are kept: a value that is missing in a year stays missing.
"""
import argparse
import os
import runpy
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
TEMP = ("Tave", "Tmax", "Tmin", "DD5", "DD18", "DDsub0", "DDsub18", "NFFD", "Eref")


def family_columns(schema, families):
    cols = []
    for s in schema["streams"]:
        start = int(s["start"])
        if s["name"] == "climate":
            for j, v in enumerate(s.get("variables", [])):
                fam = "temperature" if v.split("_")[0] in TEMP else "moisture"
                if fam in families or "climate" in families:
                    cols.append(start + j)
        elif {"landuse": "luh"}.get(s["name"], s["name"]) in families:
            cols += list(range(start, int(s["end"])))
    return np.array(sorted(cols), dtype=int)


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: static_channels.py --channels FAMILY [...] [--span Y0 Y1] -- SCRIPT [ARGS...]")
    k = sys.argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--span", type=int, nargs=2, default=(1966, 2025))
    ap.add_argument("--means-cache", default=None, help="npy file: reused if present, written if not")
    a = ap.parse_args(sys.argv[1:k])
    rest = sys.argv[k + 1:]
    from src.config_utils import load_config
    from src.community_encoder.train_DESK import covariate_io as cio
    cfg = load_config()
    states_dir = os.path.join(cfg["paths"]["hist_dir"], "yearly_states")
    schema = cio.load_schema(states_dir)
    cols = family_columns(schema, set(a.channels))
    orig = cio.load_state_stack
    if a.means_cache and os.path.exists(a.means_cache):
        means = np.load(a.means_cache)
    else:
        s = cnt = None
        for y in range(a.span[0], a.span[1] + 1):
            try:
                x = orig(y, states_dir, schema)[..., cols].astype("float64")
            except FileNotFoundError:
                continue
            ok = np.isfinite(x)
            if s is None:
                s, cnt = np.zeros_like(x), np.zeros_like(x)
            s += np.where(ok, x, 0.0)
            cnt += ok
        means = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype("float32")
        if a.means_cache:
            os.makedirs(os.path.dirname(os.path.abspath(a.means_cache)), exist_ok=True)
            np.save(a.means_cache, means)
    print(f"[static-channels] {len(cols)} channels ({', '.join(a.channels)}) held at their {a.span[0]}-{a.span[1]} "
          f"per-cell mean", flush=True)

    def patched(year, states_dir_, schema_):
        x = orig(year, states_dir_, schema_)
        x = x.copy()
        cur = x[..., cols]
        x[..., cols] = np.where(np.isfinite(cur), means, np.nan)
        return x
    cio.load_state_stack = patched
    sys.argv = rest
    runpy.run_path(rest[0], run_name="__main__")


if __name__ == "__main__":
    main()
