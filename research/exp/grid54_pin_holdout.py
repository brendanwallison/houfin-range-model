"""Write the 27 km s64 spatial-holdout block selection as a pinned list for coarser grids.

The fit code (GRID54_LADDER snapshot, src/model/data_loading.holdout_observation_mask) draws the withheld
blocks from the blocks that hold observations, so a 54 km input (different pseudo-zero sites) draws different
blocks. This reproduces the 27 km draw EXACTLY with that function and writes {"block_km", "blocks"}; the 54 km
config then sets observation_mask.holdout_blocks.pinned_blocks to it (block_cells 4 x 54 km = 8 x 27 km).

    PYTHONPATH=<snapshot> python research/exp/grid54_pin_holdout.py --snapshot <code dir> --out <json>
"""
import argparse
import json
import pickle
import sys

import numpy as np

SRC = "/work/07980/bwa386/ls6/houfin/processed/model_inputs/numpyro_input_fd_clean_v2_m2_shared_s64/metadata.pkl"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-cells", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=70)
    a = ap.parse_args()
    sys.path.insert(0, a.snapshot)
    from src.model.data_loading import holdout_observation_mask
    with open(SRC, "rb") as fh:
        m = pickle.load(fh)
    spec = {"holdout_blocks": {"block_cells": a.block_cells, "fraction": a.fraction, "seed": a.seed}}
    held = holdout_observation_mask(m, spec)
    b = a.block_cells
    rows, cols = np.asarray(m["obs_rows"]), np.asarray(m["obs_cols"])
    blocks = sorted({(int(r) // b, int(c) // b) for r, c in zip(rows[held], cols[held])})
    cell_km = float(m["path_feature_meta"]["cell_size_km"])
    out = {"block_km": b * cell_km, "blocks": [list(x) for x in blocks],
           "source": {"metadata": SRC, "block_cells": b, "fraction": a.fraction, "seed": a.seed,
                      "cell_size_km": cell_km, "n_obs_withheld_spatial": int(held.sum()),
                      "share_withheld_spatial": float(held.mean())}}
    # Round trip through the pinned path on the SAME input must reproduce the draw exactly.
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    pinned = holdout_observation_mask(m, {"holdout_blocks": {"block_cells": b, "pinned_blocks": a.out}})
    assert np.array_equal(pinned, held), "pinned selection does not reproduce the drawn one"
    print(json.dumps({k: v for k, v in out.items() if k != "blocks"} | {"n_blocks": len(blocks)}, indent=1))


if __name__ == "__main__":
    main()
