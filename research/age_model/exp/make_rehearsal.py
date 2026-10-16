"""Write a synthetic model-input set + configs so the whole Phase 1 pipeline can be rehearsed on a
laptop CPU (MAP -> refine -> pinned warm-start profile -> decomposition -> worker queue).

    python research/age_model/exp/make_rehearsal.py <root>
    # then: source <root>/env.sh  (sets AGE_MODEL_CONFIG / DATA_CONFIG / HOUFIN_ALLOW_CPU)

Inputs come from tests/test_full_model_smoke.synthetic_inputs(), written in load_data's on-disk
format (metadata.pkl + float32 memmaps). The reference raster places the grid over ~125W-70W in
ESRI:102003 so every longitude region of loglik_decompose gets data. Configs: the run_23a overlay
with input_dir / results_dir redirected into <root>; a data-config overlay pointing ref_raster at
the synthetic raster.
"""
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))


def main(root):
    import rasterio
    from rasterio.transform import from_origin
    from test_full_model_smoke import synthetic_inputs
    T, Ny, Nx = 60, 12, 16
    d = synthetic_inputs(T=T, Ny=Ny, Nx=Nx)
    inp = os.path.join(root, "numpyro_input"); os.makedirs(inp, exist_ok=True)
    meta = {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in d.items()}
    for key, path_key, name in (("Z_gathered", "z_gathered_path", "Z_gathered_rehearsal.dat"),
                                ("Z_disp_gathered", "z_disp_gathered_path", "Z_disp_gathered_rehearsal.dat")):
        arr = np.asarray(meta.pop(key), dtype=np.float32)
        mm = np.memmap(os.path.join(inp, name), dtype="float32", mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        meta[path_key] = name
    meta.pop("habitat_center_occupied", None)       # load_data recomputes it from the inputs
    meta["years"] = np.arange(1966 - 20, 1966 - 20 + T)
    with open(os.path.join(inp, "metadata.pkl"), "wb") as fh:
        pickle.dump(meta, fh)
    # ~125W..70W, 30N..50N in ESRI:102003 (Albers): x from -2.3e6 to 2.2e6, y from 0.5e6 to 2.6e6
    cell = 4.5e6 / Nx
    ref = os.path.join(root, "ref_grid.tif")
    with rasterio.open(ref, "w", driver="GTiff", height=Ny, width=Nx, count=1, dtype="float32",
                       crs="ESRI:102003", transform=from_origin(-2.3e6, 2.6e6, cell, 2.1e6 / Ny)) as dst:
        dst.write(np.ones((1, Ny, Nx), dtype="float32"))
    base = json.load(open(os.path.join(REPO, "config/overlays/map_run23a_weak_allee_juv_ratio.json")))
    base["input_dir"] = inp
    base["results_dir"] = os.path.join(root, "model_results")
    base["run_names"] = {"map": "age_map_{precision}_rehearsal", "vis_map": "age_map_{precision}_rehearsal"}
    json.dump(base, open(os.path.join(root, "age_overlay.json"), "w"), indent=2)
    json.dump({"grid": {"ref_raster": ref}}, open(os.path.join(root, "data_overlay.json"), "w"), indent=2)
    with open(os.path.join(root, "env.sh"), "w") as fh:
        fh.write(f"export AGE_MODEL_CONFIG={root}/age_overlay.json DATA_CONFIG={root}/data_overlay.json "
                 f"HOUFIN_ALLOW_CPU=1 HOUFIN_MAP_PROFILE=quick90\n")
    print("rehearsal inputs ->", root)


if __name__ == "__main__":
    main(sys.argv[1])
