"""E075 data step: per-cell LAND-COVER change 1920 -> 1966 and the Great Plains zones, on the model grid.

    python research/exp/e075_cover_change_map.py --out cover_change_1920_1966.npz

Cover = E073's physical cover (anchored_delta.COVER: the LUH2 crop / pasture / rangeland / urban fractions, forest =
primf + secdf, natural non-forest = primn + secdn), from the yearly states (DESK's transforms). Change per cell =
Euclidean distance between the 1918-1922 and 1964-1968 mean cover vectors, each channel scaled by its SD over all
valid cells in 1964-1968 (so no channel dominates by units). Saved: change (H, W; NaN where invalid), zones (H, W;
1 west, 2 Great Plains, 3 east, 0 outside), and tercile / decile thresholds over valid cells.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from anchored_delta import COVER, STATES, ZONES_TIF  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import rasterio
    from src.community_encoder.train_DESK import covariate_io as cio
    schema = cio.load_schema(STATES)
    ydir = os.path.join(STATES, "yearly_states")
    names, stream = [], []
    for s in schema["streams"]:
        v = s.get("variables") or [f"{s['name']}_{i}" for i in range(int(s["dim"]))]
        v = list(v) if len(v) == int(s["dim"]) else [f"{s['name']}_{i}" for i in range(int(s["dim"]))]
        names += v; stream += [s["name"]] * int(s["dim"])
    names, stream = np.array(names), np.array(stream)
    lu = lambda n: int(np.where((stream == "landuse") & (names == n))[0][0])

    def cover(y):
        s = cio.load_state_stack(y, ydir, schema)
        ch = []
        for c in COVER:
            if c == "forest":
                ch.append(s[..., lu("primf")] + s[..., lu("secdf")])
            elif c == "natnonforest":
                ch.append(s[..., lu("primn")] + s[..., lu("secdn")])
            else:
                ch.append(s[..., lu(c)])
        return np.stack(ch, -1).astype("float64")
    early = np.mean([cover(y) for y in range(1918, 1923)], 0)
    late = np.mean([cover(y) for y in range(1964, 1969)], 0)
    valid = np.isfinite(early).all(-1) & np.isfinite(late).all(-1)
    sd = late[valid].std(0); sd = np.where(sd > 1e-9, sd, 1.0)
    change = np.where(valid, np.sqrt((((late - early) / sd) ** 2).sum(-1)), np.nan)
    with rasterio.open(ZONES_TIF) as src:
        zones = src.read(1)
    assert zones.shape == change.shape, (zones.shape, change.shape)
    q = np.nanquantile(change[valid], [1 / 3, 2 / 3, 0.9])
    np.savez_compressed(a.out, change=change.astype("float32"), zones=zones.astype("int16"),
                        terciles=q[:2], decile90=q[2], channels=np.array(COVER))
    for z, nm in ((1, "west"), (2, "great_plains"), (3, "east")):
        m = valid & (zones == z)
        print(f"[E075] {nm}: {int(m.sum())} cells, median change {np.median(change[m]):.2f}, share in top tercile "
              f"{(change[m] > q[1]).mean():.2f}")
    print(f"[E075] terciles {q[:2].round(3)}, p90 {q[2]:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
