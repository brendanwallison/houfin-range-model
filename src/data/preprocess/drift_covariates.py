"""Raw-covariate panel for the age model's covariate drift (population_model.covariate_drift, E057 P2).

DESK's Z carries the covariates only through a community-trained encoder. The drift term asks a narrower,
interpretable question -- has the house finch's relationship to a NAMED covariate changed over time? -- so
it reads raw covariates directly. Each is gridded on the 27 km model grid (HPC ``$HOUFIN_DATA``), read at
the model's land cells, interpolated linearly between available years (ends held), transformed, and
standardised by its pooled land-cell mean/sd over the model years.

Output ``drift_covariates.npz`` next to the model inputs:
    x      (time, N_land, C) float32   standardised covariates
    names  (C,)                         covariate names (the config selects among them)
    years, mean, sd                     model years and the standardisation constants
Cells a source does not cover (BUI outside CONUS) are 0 (= the pooled mean) in every year.

    python -m src.data.preprocess.drift_covariates
"""
import os
import pickle

import numpy as np
import rasterio

# name -> (source dir, [variables summed], transform). Sums are taken BEFORE the transform.
PANEL = {
    "log_popd":      ("hyde35_grid", ["population_density"], "log1p"),
    "log_urban_pop": ("hyde35_grid", ["urban_population"], "log1p"),
    "cropland":      ("luh3_grid", ["c3ann", "c4ann", "c3per", "c4per", "c3nfx"], "none"),
    "pasture":       ("luh3_grid", ["pastr"], "none"),
    "rangeland":     ("luh3_grid", ["range"], "none"),
    "urban_land":    ("luh3_grid", ["urban"], "none"),
    "forest":        ("luh3_grid", ["primf", "secdf"], "none"),
    "tmin_q10":      ("climate_grid", ["Tmin_q10"], "none"),
    "ppt_q50":       ("climate_grid", ["PPT_q50"], "log1p"),
    "snow_q50":      ("climate_grid", ["PAS_q50"], "log1p"),
}


def _years_of(src_dir, var):
    out = []
    for n in os.listdir(src_dir):
        if n.startswith(var + "_") and n.endswith("_grid.tif"):
            tail = n[len(var) + 1:-len("_grid.tif")]
            if tail.isdigit():
                out.append(int(tail))
    return sorted(out)


def _read_series(src_dir, var, rows, cols):
    yrs = _years_of(src_dir, var)
    if not yrs:
        raise FileNotFoundError(f"no {var}_<year>_grid.tif in {src_dir}")
    vals = np.zeros((len(yrs), rows.size))
    for i, y in enumerate(yrs):
        with rasterio.open(os.path.join(src_dir, f"{var}_{y}_grid.tif")) as src:
            vals[i] = src.read(1)[rows, cols]
    return np.asarray(yrs, dtype=float), vals


def _interp_years(sy, v, years):
    out = np.zeros((years.size, v.shape[1]))
    for j, y in enumerate(years):
        if y <= sy[0]:
            out[j] = v[0]
        elif y >= sy[-1]:
            out[j] = v[-1]
        else:
            i = int(np.searchsorted(sy, y)) - 1
            f = (y - sy[i]) / (sy[i + 1] - sy[i])
            out[j] = (1 - f) * v[i] + f * v[i + 1]
    return out


def build(data_dir, input_dir, out_path, panel=PANEL):
    with open(os.path.join(input_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    rows, cols = np.asarray(meta["land_rows"]), np.asarray(meta["land_cols"])
    years = np.asarray(meta["years"]).astype(int)
    names, xs, mus, sds = [], [], [], []
    for name, (src, variables, transform) in panel.items():
        total, sy = None, None
        for var in variables:
            y_v, v = _read_series(os.path.join(data_dir, src), var, rows, cols)
            v = _interp_years(y_v, np.nan_to_num(v, nan=0.0), years)
            total = v if total is None else total + v
        if transform == "log1p":
            total = np.log1p(np.maximum(total, 0.0))
        mu, sd = float(total.mean()), float(total.std())
        if not sd > 0:
            raise ValueError(f"{name}: no variation over land cells and years")
        names.append(name); mus.append(mu); sds.append(sd)
        xs.append(((total - mu) / sd).astype(np.float32))
        print(f"[drift_covariates] {name}: mean {mu:.4g} sd {sd:.4g}")
    x = np.stack(xs, axis=-1)
    np.savez_compressed(out_path, x=x, names=np.asarray(names), years=years,
                        mean=np.asarray(mus), sd=np.asarray(sds))
    print(f"[drift_covariates] {out_path}: x {x.shape}")
    return out_path


def main():
    from src.config_utils import load_age_model_config
    input_dir = load_age_model_config()["input_dir"]
    build(os.environ["HOUFIN_DATA"], input_dir, os.path.join(input_dir, "drift_covariates.npz"))


if __name__ == "__main__":
    main()
