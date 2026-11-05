"""Gather diagnostic ensemble features at fixed cells from the completed final cubes.

This does not create an adopted mean cube or override final_mean_cube's alignment gate.
Coordinates are fixed by each season's shared ESK target; seed disagreement is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import shutil

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.community_encoder.cube_contract import compatible_contract, validate_cube_contract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder-root", required=True)
    p.add_argument("--cells", required=True)
    p.add_argument("--family", choices=["final", "final_v2", "final_clean_v1", "final_clean_v2"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = np.load(a.cells)
    if len(a.seeds) < 2 or len(set(a.seeds)) != len(a.seeds) or any(s < 1 for s in a.seeds):
        raise ValueError("supply at least two distinct positive seeds")
    cubes = [Path(a.encoder_root) / f"cube_desk_{a.family}_s{s}" for s in a.seeds]
    metas = [json.loads((c / "cube_meta.json").read_text()) for c in cubes]
    if any(compatible_contract(m) != compatible_contract(metas[0]) for m in metas[1:]):
        raise ValueError("seed feature contracts differ")
    years = metas[0]["years"]
    if any(m["years"] != years for m in metas):
        raise ValueError("seed timelines differ")
    width = metas[0]["latent_dim"]
    for name in ("holdout_cells.npy", "buffer_cells.npy"):
        paths = [Path(a.encoder_root) / f"desk_{a.family}_s{s}" / name for s in a.seeds]
        masks = [np.load(f) for f in paths]
        if any(not np.array_equal(masks[0], m) for m in masks[1:]):
            raise ValueError("seed spatial holdouts differ")
        shutil.copyfile(paths[0], out / name)
    mean = np.lib.format.open_memmap(out / "z_raw_all.npy", mode="w+", dtype="float32",
                                    shape=(len(cells), len(years), width))
    disagreement = []
    previous = None
    for iy, y in enumerate(years):
        values = []
        for c in cubes:
            z = np.load(c / f"Z_latent_{y}.npy", mmap_mode="r")
            validate_cube_contract(metas[0], z.shape[-1])
            values.append(z[cells[:, 0], cells[:, 1]])
        values = np.stack(values)
        mean[:, iy] = values.mean(0)
        disagreement.append({"year": y, "seed_sd_rms": float(np.sqrt(np.nanmean(values.var(0, ddof=1)))),
                             "step_rms": None if previous is None else
                             float(np.sqrt(np.nanmean((values.mean(0) - previous) ** 2)))})
        previous = values.mean(0)
    mean.flush()
    np.save(out / "cells.npy", cells)
    np.save(out / "years.npy", np.asarray(years, dtype="int32"))
    meta = {"family": a.family, "mode": "diagnostic_mean_features_not_adopted_cube",
            "feature_contract": compatible_contract(metas[0]),
            "members": [{"cube": str(c), "checkpoint": m["desk_checkpoint"]} for c, m in zip(cubes, metas)],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "disagreement": disagreement, "shape": list(mean.shape)}
    (out / "DONE.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
    print(json.dumps({"family": a.family, "shape": list(mean.shape), "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
