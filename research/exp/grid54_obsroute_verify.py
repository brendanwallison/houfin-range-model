"""Verify the 54 km observer-route input against the revision-2 54 km input and the 27 km obsroute npz.

  * every non-observer metadata array and the Z / Z_disp tensors are identical to numpyro_input_s64_grid54
    (same obs order, counts, cells, kernels, holdout geometry inputs);
  * the observer arrays survive coarsening + snapping unchanged (27 km npz real rows == 54 km npz real rows),
    pseudo-zero rows carry -1 / -1 / 0, and the ingested ids are a pure relabelling of the npz ids;
  * every route and every observer-route maps to exactly one coarse cell (and one fine cell).

    python research/exp/grid54_obsroute_verify.py
"""
import json
import os
import pickle

import numpy as np

P = "/work/07980/bwa386/ls6/houfin/processed/grid54"
REV2 = f"{P}/model_inputs/numpyro_input_s64_grid54"
OBS = f"{P}/model_inputs/numpyro_input_s64_grid54_obsroute"
FINE = "/home/breallis/houfin/scratch/houfin/data/bbs_2026_release/bbs_data_obsroute_for_python.npz"
COARSE = f"{P}/bbs/bbs_data_obsroute_for_python.npz"
KEYS = ("obs_route_idx", "obs_obsroute_idx", "obs_first_year")


def meta(d):
    with open(os.path.join(d, "metadata.pkl"), "rb") as fh:
        return pickle.load(fh)


def cells_per_id(ids, rows, cols):
    keep = ids >= 0
    pairs = np.unique(np.stack([ids[keep], rows[keep], cols[keep]], 1), axis=0)
    _, n = np.unique(pairs[:, 0], return_counts=True)
    return {"n_ids": int(n.size), "ids_spanning_2plus_cells": int((n > 1).sum())}


def main():
    out = {}
    a, b = meta(REV2), meta(OBS)
    diff = []
    for k in sorted(set(a) | set(b)):
        if k in KEYS or k in ("ingest_id", "z_gathered_path", "z_disp_gathered_path", "population_model_spec"):
            continue
        va, vb = a.get(k), b.get(k)
        same = (np.array_equal(np.asarray(va), np.asarray(vb)) if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray)
                else va == vb)
        if not same:
            diff.append(k)
    out["metadata_keys_differing_from_rev2"] = diff
    out["observer_keys_in_metadata"] = [k for k in KEYS if k in b]
    for key, shape in (("z_gathered_path", (a["time"], a["N_land"], a["M"])),
                       ("z_disp_gathered_path", (a["time"], a["N_land"], a["K"], a["M"]))):
        x = np.memmap(os.path.join(REV2, a[key]), dtype="float32", mode="r", shape=shape)
        y = np.memmap(os.path.join(OBS, b[key]), dtype="float32", mode="r", shape=shape)
        out[f"{key}_identical"] = bool(all(np.array_equal(x[t], y[t]) for t in range(shape[0])))

    fn, cn = np.load(FINE), np.load(COARSE, allow_pickle=True)
    npf, npc = int(fn["N_pseudo"]), int(cn["N_pseudo"])
    pres = {}
    for k in KEYS:
        pres[k] = {"real_rows_identical": bool(np.array_equal(fn[k][npf:], cn[k][npc:])),
                   "pseudo_fill": np.unique(cn[k][:npc]).tolist()}
    out["npz_observer_arrays"] = pres
    out["npz_other_real_arrays_identical"] = {
        k: bool(np.array_equal(fn[k][npf:], cn[k][npc:])) for k in ("obs_year", "observed_results", "obs_quality")}
    for tag, d, n0, f in (("fine_27km", fn, npf, 1), ("coarse_54km", cn, npc, 1)):
        r, c = d["obs_rows"][n0:], d["obs_cols"][n0:]
        out[f"{tag}_cells_per_id"] = {k: cells_per_id(np.asarray(d[k][n0:]), r, c)
                                      for k in ("obs_route_idx", "obs_obsroute_idx")}
    # parents of the fine cells: a route split across 27 km cells would show here; the coarse npz check
    # above also covers snapping (applied per fine cell, so it cannot split one).
    # ingested ids: bijective relabelling of the npz ids on the kept observations
    rel = {}
    for k in ("obs_route_idx", "obs_obsroute_idx"):
        if k not in b:
            continue
        m = np.asarray(b[k]); real = m >= 0
        src = np.asarray(cn[k])  # ingest keeps every obs (no year/bounds drops here) but resamples pseudo rows
        src_real = src[npc:]
        m_real = m[-src_real.size:]
        pairs = np.unique(np.stack([src_real, m_real], 1), axis=0)
        rel[k] = {"n_ids": int(np.unique(m[real]).size), "max_id_plus_1": int(m.max()) + 1,
                  "bijective": bool(np.unique(pairs[:, 0]).size == pairs.shape[0] == np.unique(pairs[:, 1]).size),
                  "pseudo_rows_minus1": bool((m[:m.size - src_real.size] == -1).all())}
        obs_cells = cells_per_id(m, np.asarray(b["obs_rows"]), np.asarray(b["obs_cols"]))
        rel[k]["ingested_ids_spanning_2plus_cells"] = obs_cells["ids_spanning_2plus_cells"]
    fy = np.asarray(b["obs_first_year"])
    rel["obs_first_year"] = {"real_identical": bool(np.array_equal(fy[-(fn["observed_results"].size - npf):], fn["obs_first_year"][npf:])),
                             "share_first_year_real": float(fn["obs_first_year"][npf:].mean())}
    out["ingested_observer_arrays"] = rel
    with open(f"{P}/grid54_obsroute_verify.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
