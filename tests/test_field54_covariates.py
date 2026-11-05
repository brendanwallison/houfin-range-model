"""Tests for the FIELD54 native-coarse covariate switches (all default-off).

* netcdf_grid.parent_weighted_reproject: a per-parent rate aggregates as
  sum(rate*parent)/sum(parent), falls back to the plain average where the parent is 0,
  and the plain output equals the default rule exactly.
* netcdf_grid.attach_parents: items without a parent mapping are returned unchanged.
* climate_regroup: regrouping per-sub-point output to a 2x coarser grid, with a parent
  straddling two chunks, equals quantile_aggregate over the union of the children's
  sub-points.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.acquire.climatena import quantile_aggregate
from src.data.preprocess import climate_regroup as cr
from src.data.preprocess import netcdf_grid as ncg
from src.data.preprocess.build_ref_grid import build_ref_grid
from src.processing import regrid


def _geo(arr, lon0=-130.0, lat0=55.0, step=1.0):
    ny, nx = arr.shape
    lon = lon0 + step * (np.arange(nx) + 0.5)
    lat = lat0 - step * (np.arange(ny) + 0.5)
    da = xr.DataArray(arr.astype("float32"), dims=("lat", "lon"),
                      coords={"lat": lat, "lon": lon})
    return da.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")


def test_parent_weighted_rule_and_fallback():
    with tempfile.TemporaryDirectory() as d:
        # 2-degree target cells over a 1-degree source: each target = 2x2 source pixels.
        build_ref_grid((-130.0, 47.0, -120.0, 55.0), "EPSG:4326", 2.0,
                       os.path.join(d, "ref.tif"))
        ref = rioxarray.open_rasterio(os.path.join(d, "ref.tif"))
        ny, nx = 8, 10
        rate = np.zeros((ny, nx))
        parent = np.zeros((ny, nx))
        # target (0,0): one source pixel with parent 0.4 and rate 10, rest parent 0 rate 0
        parent[0, 0], rate[0, 0] = 0.4, 10.0
        # target (0,1): parent everywhere 0, rate 3 everywhere -> fallback = plain = 3
        rate[0:2, 2:4] = 3.0
        # target (1,0): two pixels with parent 0.2 / 0.6 and rate 1 / 5 -> (0.2+3)/0.8 = 4
        parent[2, 0], rate[2, 0] = 0.2, 1.0
        parent[2, 1], rate[2, 1] = 0.6, 5.0
        r, a = _geo(rate), _geo(parent)

        def rp(da):
            return regrid.reproject_to_ref(da, ref, resampling="average")

        w, plain, n_fb = ncg.parent_weighted_reproject(r, a, ref, rp)
        w = np.asarray(w.values).squeeze()
        p = np.asarray(plain.values).squeeze()
        np.testing.assert_allclose(p, np.asarray(rp(r).values).squeeze())
        assert np.isclose(w[0, 0], 10.0) and np.isclose(p[0, 0], 2.5)
        assert np.isclose(w[0, 1], 3.0) and np.isclose(p[0, 1], 3.0)
        assert np.isclose(w[1, 0], 4.0) and np.isclose(p[1, 0], 1.5)
        assert n_fb > 0


def test_attach_parents_default_off():
    items = [dict(var="urban", year=1900, out_path="/x/urban_1900_grid.tif")]
    assert ncg.attach_parents(items, {}, {}) == items
    idx = {"c3ann": ("/f.nc", "time", "lon", "lat", {1900: 7})}
    out = ncg.attach_parents([dict(var="irrig_c3ann", year=1900, out_path="/x/a.tif")] + items,
                             {"irrig_c3ann": ["c3ann"]}, idx, plain_dir="/p")
    assert out[0]["parents"][0]["tindex"] == 7 and out[0]["plain_out_path"] == "/p/a.tif"
    assert "parents" not in out[1]


def test_climate_regroup_union_quantiles_with_straddling_parent():
    rng = np.random.default_rng(0)
    H, W, g = 4, 4, 3  # fine grid 4x4 cells, 3x3 sub-points each; coarse 2x2
    rows, cols, ids = [], [], []
    k = 0
    for r in range(H):
        for c in range(W):
            n = g * g - (r + c) % 3  # ragged: some sub-points dropped (fine-land filter)
            for _ in range(n):
                rows.append(r); cols.append(c); ids.append(k); k += 1
    sp = pd.DataFrame({"id": ids, "row": rows, "col": cols})
    sp["parent_id"] = sp["row"] * W + sp["col"]
    periods = [1901, 1902]
    pts = pd.DataFrame([(i, "cru.gpcc", p) for i in ids for p in periods],
                       columns=["id", "DATASET", "PERIOD"])
    pts["Tmax_01"] = rng.normal(size=len(pts))
    pts["PPT_07"] = rng.gamma(2.0, size=len(pts))
    with tempfile.TemporaryDirectory() as d:
        ch = os.path.join(d, "_chunks")
        os.makedirs(ch)
        # tile boundary at fine row 1|2 would be clean; cut at row 0|1 so coarse row 0 straddles
        tiles = [sp[sp["row"] < 1], sp[sp["row"] >= 1]]
        for i, t in enumerate(tiles):
            t.to_csv(os.path.join(ch, f"chunk_{i:03d}.csv"), index=False)
            od = os.path.join(ch, f"out_{i:03d}")
            os.makedirs(od)
            pts[pts["id"].isin(t["id"])].to_csv(os.path.join(od, "climate_points.csv"),
                                                index=False)
        out = os.path.join(d, "coarse")
        summ = cr.regroup(ch, 2, W // 2, out, workers=1)
        assert summ["n_boundary_parents"] == 2
        idp = pd.DataFrame({"id": sp["id"],
                            "parent_id": cr.coarse_parent(sp["row"], sp["col"], 2, W // 2)})
        ref = quantile_aggregate(pts, idp)
        for lvl, exp in ref.items():
            got = pd.read_csv(os.path.join(out, f"climate_{lvl}.csv"))
            exp = exp.sort_values(["id", "PERIOD"]).reset_index(drop=True)
            assert list(got.columns) == list(exp.columns)
            np.testing.assert_array_equal(got[["id", "PERIOD"]].to_numpy(),
                                          exp[["id", "PERIOD"]].to_numpy())
            np.testing.assert_allclose(got.iloc[:, 2:].to_numpy(), exp.iloc[:, 2:].to_numpy(),
                                       rtol=0, atol=1e-12)


def test_climate_regroup_union_across_two_chunk_dirs():
    """A supplementary run (second chunk dir) is unioned with the original one."""
    rng = np.random.default_rng(1)
    H, W = 4, 4
    sp = pd.DataFrame([(r * W + c, r, c) for r in range(H) for c in range(W)],
                      columns=["cell", "row", "col"]).loc[lambda d: d.index.repeat(4)]
    sp = sp.reset_index(drop=True)
    sp["id"] = np.arange(len(sp))
    sp["parent_id"] = sp["row"] * W + sp["col"]
    pts = pd.DataFrame([(i, "cru.gpcc", p) for i in sp["id"] for p in (1901, 1902)],
                       columns=["id", "DATASET", "PERIOD"])
    pts["Tmin_01"] = rng.normal(size=len(pts))
    extra = sp["row"].isin([1]) & sp["col"].isin([0, 3])   # 'missing' fine cells
    with tempfile.TemporaryDirectory() as d:
        dirs = []
        for k, part in enumerate([sp[~extra], sp[extra]]):
            ch = os.path.join(d, f"run{k}", "_chunks")
            os.makedirs(os.path.join(ch, "out_000"))
            part.drop(columns="cell").to_csv(os.path.join(ch, "chunk_000.csv"), index=False)
            pts[pts["id"].isin(part["id"])].to_csv(
                os.path.join(ch, "out_000", "climate_points.csv"), index=False)
            dirs.append(ch)
        out = os.path.join(d, "coarse")
        summ = cr.regroup(dirs, 2, W // 2, out, workers=1)
        assert summ["n_subpoints"] == len(sp) and summ["n_boundary_parents"] == 2
        idp = pd.DataFrame({"id": sp["id"],
                            "parent_id": cr.coarse_parent(sp["row"], sp["col"], 2, W // 2)})
        exp = quantile_aggregate(pts, idp)["q50"].sort_values(["id", "PERIOD"])
        got = pd.read_csv(os.path.join(out, "climate_q50.csv"))
        np.testing.assert_allclose(got.iloc[:, 2:].to_numpy(), exp.iloc[:, 2:].to_numpy(),
                                   rtol=0, atol=1e-12)
