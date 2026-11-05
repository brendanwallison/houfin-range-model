"""FIELD54 Stage F helpers: the age-model overlay for a 54 km s64 habitat cube, and the input check.

    overlay : grid54_age_obsroute.json with ONLY the habitat paths changed (input_dir, raw_z_dir, path_features,
              path_diagnostics_dir, run_names.map / vis_map), as research/experiments/GRID54_INPUTS_FDINTERP_2026-10-29.md
              built grid54_age_fdinterp_obsroute.json.
    check   : a new numpyro input vs a reference grid54 obsroute input: every metadata key outside the habitat-derived
              ones identical; Z_gathered / Z_disp_gathered differences, with Z_disp differences located on the
              land-share-floor pairs reported by path_feature_meta.

    python research/exp/field54_stage_f.py overlay --name N --root G --base grid54_age_obsroute.json --out OVERLAY
    python research/exp/field54_stage_f.py check --new INPUT --ref INPUT --zdisp ZDISP_DIR --out check.json
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

ZKEYS = ("ingest_id", "z_gathered_path", "z_disp_gathered_path", "z_kernel_contract", "path_feature_meta")


def overlay(base, name, root):
    c = json.loads(Path(base).read_text())
    c["input_dir"] = f"{root}/model_inputs/numpyro_input_{name}_s64_grid54_obsroute"
    c["raw_z_dir"] = f"{root}/latent_avian_paths_{name}_s64"
    c["path_features"] = {"input_dir": f"{root}/encoder/cube_{name}_s64", "output_dir": c["raw_z_dir"]}
    c["path_diagnostics_dir"] = f"{root}/latent_avian_path_diagnostics_{name}"
    for k in ("map", "vis_map"):
        if c["run_names"][k] != "age_map_{precision}_shared_clean_s64_grid54_obsroute":
            raise ValueError(f"unexpected base run name {c['run_names'][k]}")
        c["run_names"][k] = "age_map_{precision}_" + name + "_s64_grid54_obsroute"
    c["_field54_stage_f_comment"] = (
        "FIELD54 Stage F (research/experiments/FIELD54_FIELD_2026-10-30.md): grid54_age_obsroute.json with ONLY the "
        f"habitat cube changed to {name} (summer 64 dims of a 54 km 128-d cube; Z_disp with cell-integrated kernels and "
        "the land-share floor). Observations, holdout, pseudo-zeros, founders and every prior identical.")
    return c


def meta(d):
    with open(os.path.join(d, "metadata.pkl"), "rb") as fh:
        return pickle.load(fh)


def zmem(d, m, disp=False):
    if disp:
        return np.memmap(os.path.join(d, m["z_disp_gathered_path"]), dtype="float32", mode="r",
                         shape=(m["time"], m["N_land"], m["K"], m["M"]))
    return np.memmap(os.path.join(d, m["z_gathered_path"]), dtype="float32", mode="r",
                     shape=(m["time"], m["N_land"], m["M"]))


def check(new, ref, zdisp):
    mN, mR = meta(new), meta(ref)
    diff, same = [], []
    for k in sorted(set(mN) | set(mR)):
        if k in ZKEYS:
            continue
        a, b = mN.get(k), mR.get(k)
        eq = (np.array_equal(np.asarray(a), np.asarray(b)) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray)
              else a == b)
        (same if eq else diff).append(k)
    out = {"metadata_keys_identical": len(same), "metadata_keys_differing": diff,
           "z_kernel_contract_keys_differing": sorted(k for k in set(mN["z_kernel_contract"]) | set(mR["z_kernel_contract"])
                                                      if mN["z_kernel_contract"].get(k) != mR["z_kernel_contract"].get(k)),
           "shapes": {k: [int(mN[k]), int(mR[k])] for k in ("time", "N_land", "K", "M")}}
    zN, zR = zmem(new, mN), zmem(ref, mR)
    out["Z_gathered_max_abs_diff"] = float(np.abs(np.asarray(zN, np.float64) - zR).max())
    dN, dR = zmem(new, mN, True), zmem(ref, mR, True)
    pair = np.zeros((int(mN["N_land"]), int(mN["K"])))
    for t in range(int(mN["time"])):
        pair = np.maximum(pair, np.abs(np.asarray(dN[t], np.float64) - dR[t]).max(-1))
    pm = json.loads((Path(zdisp)/"path_feature_meta.json").read_text())
    floor = pm.get("land_share_floor")
    out["Z_disp"] = {"pairs_total": int(pair.size), "pairs_diff_gt_1e-5": int((pair > 1e-5).sum()),
                     "max_abs_diff": float(pair.max()),
                     "max_abs_diff_outside_top_affected": float(np.sort(pair.ravel())[::-1][
                         floor["affected_cell_band_pairs"]] if floor else pair.max()),
                     "floor_meta": floor}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=["overlay", "check"])
    p.add_argument("--name"); p.add_argument("--root"); p.add_argument("--base"); p.add_argument("--out", type=Path)
    p.add_argument("--new"); p.add_argument("--ref"); p.add_argument("--zdisp")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res = overlay(a.base, a.name, a.root) if a.stage == "overlay" else check(a.new, a.ref, a.zdisp)
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res if a.stage == "check" else {"overlay": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
