"""54 km summer-64 Z cube from the backcast40 field_d_interp_s1 cube (block mean = kernel mean embedding).

Mirrors coarsen_model_grid.coarsen_cube (research/experiments/GRID54_INPUTS_2026-10-29.md) with one change:
the source is a raw 27 km cube (133x224x128 Z_latent_{year}.npy) instead of a 27 km numpyro input's gathered Z.
  * select the SUMMER block, columns 0-63 (exact coordinate selection; linear, so it commutes with both the
    block mean and the per-feature path integration that builds Z_disp afterwards);
  * gather the 27 km land cells (row-major np.where order) and apply the SAVED 54 km averaging operator
    (grid54/land_mask/averaging_operator.npz: mean over LAND children; asserted equal to a fresh rebuild);
  * write Z_latent_{year}.npy on the 67x112 grid (NaN off land) and cube_meta.json.

The source cube has no cube_meta.json (research/exp/z_assim.py export writes none). Its contract is the control
cube's (cubes/backcast40/control_s1/cube_meta.json: the arm overwrites the control's land cells in the control's
teacher spaces), amended with the field_d_interp provenance; that amended 27 km contract is written next to the
54 km cube as source_cube_meta_27km.json and embedded in cube_meta.json.

    research/bin/rx py research/exp/grid54_fdinterp_cube.py
    ... --dims 128 --out <dir>   (FIELD54 plan rule 1: ALL 128 dims, so the averaged arm has the same dimensional
                                  access as the native 54 km field when both are scored with one readout)
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from src.data.masks import read_land_mask  # noqa: E402
from src.processing.coarsen_model_grid import CUBE_NOTE, averaging_operator  # noqa: E402
from src.community_encoder.cube_contract import validate_cube_contract  # noqa: E402

ROOT = _REPO / ".research_runtime/community_z_repair_20261026"
SRC = ROOT / "cubes/backcast40/field_d_interp_s1"
CONTROL = ROOT / "cubes/backcast40/control_s1"
G = "/work/07980/bwa386/ls6/houfin/processed/grid54"
FINE_MASK = "/home/breallis/houfin/scratch/houfin/data/land_mask/ocean_mask_27km.tif"
F = 2
LO, HI = 0, 64


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_contract(years, file_sha):
    """Control cube contract amended for the field_d_interp arm (27 km, 128-d)."""
    ctrl = json.loads((CONTROL / "cube_meta.json").read_text())
    done = json.loads((SRC / "ASSIM_DONE.json").read_text())
    c = copy.deepcopy(ctrl)
    if [int(y) for y in c["years"]] != years:
        raise ValueError("control cube years differ from the source cube's files")
    c["source_cube_dir"] = str(SRC)
    c["contract_origin"] = {
        "copied_from": str(CONTROL / "cube_meta.json"),
        "copied_from_sha256": sha256(CONTROL / "cube_meta.json"),
        "why": "research/exp/z_assim.py export writes no cube_meta.json; the arm overwrites the control cube's "
               "land cells with predictions in the SAME teacher spaces (dims 0-63 summer teacher, 64-127 winter "
               "teacher), so the control's contract carries over with this provenance added",
    }
    c["assim_arm"] = {
        "arm": "field_d_interp", "export": "z_assim.export (research/exp/z_assim.py)",
        "assim_done": done,
        "z_assim_py_sha256_now": sha256(_REPO / "research/exp/z_assim.py"),
        "record": "research/experiments/ASSIM_FIELD_2026-10-27.md (In-era interpolation test)",
        "model": "Z(s,t) = Dbar(s) + u(s) + A.d(s,t) + R(s,t): control DESK covariate change recalibrated, plus a "
                 "GMRF level offset and a rank-16 space-time field (128 RBF at 200 km x cubic B-spline, 10-yr knots)",
    }
    c["evaluation_build"] = {
        "task": "backcast40",
        "spatial_holdouts_masked": True,
        "note": "EVALUATION build, not a production cube: the control DESK and the field were fitted with the "
                "backcast40 spatial holdout panel (and its buffer) and evaluation records withheld; summer sources "
                "are training points (1996+ plus pre-1996 rows outside the holdout/buffer for the *_interp arms) "
                "and training-circle winter coordinates. Values at withheld places are predictions.",
        "house_finch_excluded": True,
        "sealed_species_excluded": True,
    }
    c["source_file_sha256"] = file_sha
    validate_cube_contract(c, 128)
    return c


def main():
    global HI
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{G}/encoder/cube_fdinterp_s64_grid54")
    ap.add_argument("--dims", type=int, choices=[64, 128], default=64,
                    help="64 = summer block only (default, the ladder input); 128 = both seasonal blocks")
    a = ap.parse_args()
    HI = a.dims
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; refusing to overwrite")

    land_f = read_land_mask(FINE_MASK).astype(bool)
    land_c = read_land_mask(f"{G}/land_mask/ocean_mask_54km.tif").astype(bool)
    A, parent, n_child = averaging_operator(land_f, land_c, F)
    A_saved = sp.load_npz(f"{G}/land_mask/averaging_operator.npz")
    if abs(A_saved - A).max() > 0:
        raise ValueError("saved averaging operator differs from a fresh rebuild")
    fr, fc = np.where(land_f)
    cr, cc = np.where(land_c)

    files = sorted(SRC.glob("Z_latent_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
    years = [int(p.stem.split("_")[-1]) for p in files]
    if years != list(range(1902, 2026)):
        raise ValueError(f"source years {years[0]}-{years[-1]} ({len(years)}) != 1902-2025")
    out.mkdir(parents=True, exist_ok=True)
    file_sha = {}
    for p, y in zip(files, years):
        file_sha[p.name] = sha256(p)
        g = np.load(p)
        if g.shape != land_f.shape + (128,):
            raise ValueError(f"{p.name}: shape {g.shape}")
        if not np.isfinite(g[land_f]).all():
            raise ValueError(f"{p.name}: non-finite values on 27 km land")
        z = np.asarray(g[fr, fc, LO:HI], dtype=np.float64)
        zc = A @ z
        grid = np.full(land_c.shape + (HI - LO,), np.nan, dtype=np.float32)
        grid[cr, cc] = zc.astype(np.float32)
        np.save(out / f"Z_latent_{y}.npy", grid)

    src_c = source_contract(years, file_sha)
    (out / "source_cube_meta_27km.json").write_text(json.dumps(src_c, indent=2))
    if HI == 128:
        contract = copy.deepcopy(src_c)
        contract["coarsened_from"] = {
            "method": f"mean over 27 km LAND children of each {F}x{F} block (sparse averaging operator), all 128 dims",
            "factor": F, "operator": f"{G}/land_mask/averaging_operator.npz",
            "operator_sha256": sha256(f"{G}/land_mask/averaging_operator.npz"),
            "source_cube_dir": str(SRC),
            "source_file_sha256_digest": hashlib.sha256(
                "".join(f"{k}:{v}\n" for k, v in sorted(file_sha.items())).encode()).hexdigest(),
            "source_grid_shape": list(land_f.shape), "grid_shape": list(land_c.shape),
            "transform": [54000.0, 0.0, -2976931.1039, 0.0, -54000.0, 2067815.4047]}
        contract["coarsening_note"] = CUBE_NOTE
        contract["kernel_contract"] = "per block: Z(x) dot Z(x') ~= uncentered Ruzicka [block mean embedding; see note]"
        contract["field54_use"] = ("FIELD54 plan rule 1: the averaged 27 km field_d_interp scored on the native 54 km "
                                   "twin panel; dims stay in the 27 km teacher spaces (species readouts only)")
        validate_cube_contract(contract, 128)
        (out / "cube_meta.json").write_text(json.dumps(contract, indent=2))
        print(json.dumps({"out": str(out), "dims": 128, "n_years": len(years), "coarse_land": int(land_c.sum()),
                          "children_hist": {str(k): int((n_child == k).sum()) for k in range(1, 5)}}, indent=2))
        return
    summer = [b for b in src_c["feature_blocks"] if b["name"] == "summer"]
    if len(summer) != 1 or (summer[0]["start"], summer[0]["stop"]) != (LO, HI):
        raise ValueError("source contract has no summer block at columns 0-63")
    contract = {
        "schema_version": 2,
        "kernel": "ruzicka", "centered": False, "latent_dim": HI - LO,
        "feature_blocks": [{**copy.deepcopy(summer[0]), "start": 0, "stop": HI - LO}],
        "temporal_output": src_c["temporal_output"],
        "esk_basis_dir": summer[0]["esk_basis_dir"],
        "kernel_contract": "Z(x) dot Z(x') ~= uncentered Ruzicka(x,x') [block mean embedding; see note]",
        "years": years,
        "ensemble_method": None, "ensemble_members": None,
        "desk_checkpoint": src_c.get("desk_checkpoint"),
        "desk_meta": src_c.get("desk_meta"),
        "source_cube": "backcast40 field_d_interp_s1 (z_assim field arm on the masked control_s1 cube)",
        "evaluation_build": src_c["evaluation_build"],
        "seasonal_selection": {
            "season": "summer", "source_columns": list(range(LO, HI)),
            "note": "dims 0-63 of the 128-d source are the SUMMER teacher space (backcast40 summer ESK basis); "
                    "the winter half (64-127) is dropped. Matches the current ladder input "
                    "numpyro_input_s64_grid54 (summer block of the clean-v2 cube, research/exp/final_seasonal_inputs.py), "
                    "except that selection here PRECEDES path integration; both steps are linear and per-feature, "
                    "so the order does not matter",
            "operation": "exact_coordinate_selection_before_block_mean_and_path_integration",
        },
        "coarsened_from": {
            "method": f"mean over 27 km LAND children of each {F}x{F} block (sparse averaging operator)",
            "factor": F,
            "operator": f"{G}/land_mask/averaging_operator.npz",
            "operator_sha256": sha256(f"{G}/land_mask/averaging_operator.npz"),
            "source_cube_dir": str(SRC),
            "source_file_sha256_digest": hashlib.sha256(
                "".join(f"{k}:{v}\n" for k, v in sorted(file_sha.items())).encode()).hexdigest(),
            "source_contract_27km": "source_cube_meta_27km.json (this directory)",
            "source_grid_shape": list(land_f.shape),
            "grid_shape": list(land_c.shape),
            "transform": [54000.0, 0.0, -2976931.1039, 0.0, -54000.0, 2067815.4047],
        },
        "coarsening_note": CUBE_NOTE,
    }
    validate_cube_contract(contract, HI - LO)
    (out / "cube_meta.json").write_text(json.dumps(contract, indent=2))
    print(json.dumps({"out": str(out), "years": [years[0], years[-1]], "n_years": len(years),
                      "coarse_land": int(land_c.sum()), "fine_land": int(land_f.sum()),
                      "children_hist": {str(k): int((n_child == k).sum()) for k in range(1, 5)},
                      "source_digest": contract["coarsened_from"]["source_file_sha256_digest"]}, indent=2))


if __name__ == "__main__":
    main()
