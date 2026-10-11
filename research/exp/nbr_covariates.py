"""E039e: neighbourhood covariates -- each key's covariates averaged over its 5 x 5-cell neighbourhood (DESK's
convolution footprint), built exactly like the suite's raw-matched covariates (F_cov_raw): each year's state, DESK's
own normalization (desk_meta mu / sd), no output EMA; the mean is masked (invalid cells excluded, as DESK's masked
convolution does).

    ESK_DESK_CONFIG=<overlay of the cache's model> python research/exp/nbr_covariates.py --cache <cache> --out <dir>

Writes <out>/F_nbr.npy, (n_keys, C) float32, in the cache's key order.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--size", type=int, default=5)
    a = ap.parse_args()
    from scipy.ndimage import uniform_filter
    from src.config_utils import load_config
    from src.community_encoder.train_DESK import covariate_io as cio
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config()
    states_dir = os.path.join(cfg["paths"]["hist_dir"], "yearly_states")
    schema = cio.load_schema(states_dir)
    dm = np.load(os.path.join(cfg["paths"]["desk_output_dir"], "desk_meta.npz"), allow_pickle=True)
    mu, sd = dm["mu"].astype("float32"), dm["sd"].astype("float32")
    keys = np.load(os.path.join(a.cache, "keys.npy"))
    out = np.full((len(keys), len(mu)), np.nan, "float32")
    for y in np.unique(keys[:, 2]):
        covn, mask = cio.norm_grid(cio.load_state_stack(int(y), states_dir, schema), mu, sd)
        m = mask.astype("float64")
        den = uniform_filter(m, size=a.size, mode="constant")
        num = uniform_filter(covn.astype("float64") * m[..., None], size=(a.size, a.size, 1), mode="constant")
        nb = np.where(den[..., None] > 1e-9, num / np.maximum(den[..., None], 1e-9), np.nan)
        sel = keys[:, 2] == y
        out[sel] = nb[keys[sel, 0], keys[sel, 1]]
    np.save(os.path.join(a.out, "F_nbr.npy"), out)
    print(f"[nbr] {len(keys):,} keys x {out.shape[1]} channels, {a.size}x{a.size} masked mean; finite share "
          f"{np.isfinite(out).all(1).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
