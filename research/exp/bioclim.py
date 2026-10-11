"""E039c: the 19 WorldClim BIOCLIM variables (the standard SDM climate inputs) at every key of a research cache,
from the same yearly states DESK reads, smoothed over years with DESK's own input EMA so they match F_cov_raw.

    ESK_DESK_CONFIG=<overlay> python research/exp/bioclim.py --cache <research cache> --out <dir>

Monthly bands are the states' 12 bands (August of the previous year to July of the survey year), median (q50)
values of Tave, Tmax, Tmin and PPT. Definitions (WorldClim): BIO1 mean of monthly Tave; BIO2 mean of monthly
(Tmax - Tmin); BIO3 = 100 BIO2 / BIO7; BIO4 = 100 sd(monthly Tave); BIO5 max monthly Tmax; BIO6 min monthly Tmin;
BIO7 = BIO5 - BIO6; BIO8-11 mean Tave of the wettest / driest / warmest / coldest quarter (3 consecutive months,
circular); BIO12 annual PPT; BIO13 / BIO14 wettest / driest month; BIO15 = 100 sd / (1 + mean) of monthly PPT; BIO16-19
PPT of the wettest / driest / warmest / coldest quarter. Also writes the 12 LUH class fractions with the same EMA
(climate + land cover: the usual SDM input set). EMA: x_ema(y) = (1 - a) x_ema(y-1) + a x(y), a = 1 - 2^(-1/hl),
started at 1940's value (DESK's warm-up start), hl = the cache's ema_half_life.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

LUH_CLASSES = ("c3ann", "c3nfx", "c3per", "c4ann", "c4per", "pastr", "primf", "primn", "range", "secdf", "secdn",
               "urban")
BIO_NAMES = [f"BIO{i}" for i in range(1, 20)]


def bioclim(tave, tmax, tmin, ppt):
    """``(n, 12)`` monthly arrays -> ``(n, 19)`` BIOCLIM. Pure."""
    q = lambda m: np.stack([np.roll(m, -k, axis=1)[:, :3].sum(1) for k in range(12)], 1)   # circular quarters
    tq = q(tave) / 3.0
    pq = q(ppt)
    pick = lambda vals, idx: np.take_along_axis(vals, idx[:, None], 1)[:, 0]
    b = np.empty((tave.shape[0], 19))
    b[:, 0] = tave.mean(1)
    b[:, 1] = (tmax - tmin).mean(1)
    b[:, 4] = tmax.max(1)
    b[:, 5] = tmin.min(1)
    b[:, 6] = b[:, 4] - b[:, 5]
    b[:, 2] = 100.0 * b[:, 1] / np.where(np.abs(b[:, 6]) > 1e-6, b[:, 6], np.nan)
    b[:, 3] = 100.0 * tave.std(1)
    b[:, 7] = pick(tq, pq.argmax(1))
    b[:, 8] = pick(tq, pq.argmin(1))
    b[:, 9] = tq.max(1)
    b[:, 10] = tq.min(1)
    b[:, 11] = ppt.sum(1)
    b[:, 12] = ppt.max(1)
    b[:, 13] = ppt.min(1)
    b[:, 14] = 100.0 * ppt.std(1) / (1.0 + ppt.mean(1))
    b[:, 15] = pq.max(1)
    b[:, 16] = pq.min(1)
    b[:, 17] = pick(pq, tq.argmax(1))
    b[:, 18] = pick(pq, tq.argmin(1))
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--start", type=int, default=1940)
    a = ap.parse_args()
    from src.config_utils import load_config
    from src.community_encoder.train_DESK import covariate_io as cio
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config()
    states_dir = os.path.join(cfg["paths"]["hist_dir"], "yearly_states")
    schema = cio.load_schema(states_dir)
    meta = json.load(open(os.path.join(a.cache, "meta.json"), encoding="utf-8"))
    hl = float(meta["ema"]["ema_half_life"])
    alpha = 1.0 - 2.0 ** (-1.0 / hl)
    keys = np.load(os.path.join(a.cache, "keys.npy"))
    cells, kc = np.unique(keys[:, :2], axis=0, return_inverse=True)
    kc = kc.ravel()
    cl = [s for s in schema["streams"] if s["name"] == "climate"][0]
    names = cl["variables"]
    col = lambda var: np.array([int(cl["start"]) + names.index(f"{var}_b{b:02d}m{m:02d}_q50")
                                for b, m in zip(range(1, 13), [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7])])
    cT, cX, cN, cP = col("Tave"), col("Tmax"), col("Tmin"), col("PPT")
    lu = [s for s in schema["streams"] if s["name"] == "landuse"][0]
    cL = np.array([int(lu["start"]) + lu["variables"].index(v) for v in LUH_CLASSES])
    years = np.arange(a.start, int(keys[:, 2].max()) + 1)
    out_b = np.full((len(keys), 19), np.nan)
    out_l = np.full((len(keys), 12), np.nan)
    ema_b = ema_l = None
    for y in years:
        x = cio.load_state_stack(int(y), states_dir, schema)[cells[:, 0], cells[:, 1]].astype("float64")
        b = bioclim(x[:, cT], x[:, cX], x[:, cN], x[:, cP])
        l = x[:, cL]
        if ema_b is None:
            ema_b, ema_l = b.copy(), l.copy()
        else:
            # a cell's EMA starts at its first finite value; a missing year leaves it unchanged
            step = lambda e, v: np.where(np.isnan(e), v, np.where(np.isfinite(v), (1 - alpha) * e + alpha * v, e))
            ema_b, ema_l = step(ema_b, b), step(ema_l, l)
        sel = keys[:, 2] == y
        out_b[sel] = ema_b[kc[sel]]
        out_l[sel] = ema_l[kc[sel]]
        if y in (years[0], 1966, 2025):
            print(f"[bioclim] {y}: BIO1 (annual mean temp) median {np.nanmedian(b[:, 0]):.1f}, BIO12 (annual ppt) "
                  f"median {np.nanmedian(b[:, 11]):.0f}, BIO5 {np.nanmedian(b[:, 4]):.1f}, BIO6 {np.nanmedian(b[:, 5]):.1f}",
                  flush=True)
    np.save(os.path.join(a.out, "bioclim_ema.npy"), out_b.astype("float32"))
    np.save(os.path.join(a.out, "luh_classes_ema.npy"), out_l.astype("float32"))
    json.dump({"cache": a.cache, "half_life": hl, "start": a.start, "names": BIO_NAMES, "luh": list(LUH_CLASSES),
               "finite_share": float(np.isfinite(out_b).all(1).mean())},
              open(os.path.join(a.out, "bioclim_meta.json"), "w", encoding="utf-8"), indent=1)
    print(f"[bioclim] {len(keys):,} keys, finite share {np.isfinite(out_b).all(1).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
