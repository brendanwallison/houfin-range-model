"""WHERE in community space do BBS route communities sit, that 16,000 landmarks do not reach?

basis_domain_gap.py refuted the scale explanation: landmark and BBS total abundance agree (24.1
vs 22.1) and sparsity is close (13 vs 11 nonzero of 96). The one number that does not agree is
the BEST-MATCHING landmark -- 0.645 for a landmark against its neighbours, 0.170 for a BBS
community -- and ||z||^2 tracks it (0.652 vs 0.150). So the landmark set does not cover the region
BBS occupies, and the question is which species make it a different region.

Ruzicka is Sum-min / Sum-max, and Sum-max = Sum-min + Sum|x - l|. R = 0.17 therefore means the
MISMATCHED mass is about 5x the shared mass. That decomposes per species and by sign, which says
whether BBS carries mass the landmarks lack, the landmarks carry mass BBS lacks, or both.

Run: cd $HOUFIN_REPO && python scripts/diagnostics/basis_coverage_gap.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.community_encoder.train_DESK.config_utils import load_config          # noqa: E402


def main():
    cfg = load_config(os.environ.get("ESK_DESK_CONFIG") or None)
    z_dir = cfg["desk"]["z_dir"]
    meta = json.load(open(os.path.join(z_dir, "meta.json"), encoding="utf-8"))
    L = np.load(os.path.join(z_dir, "esk_landmarks.npy")).astype("float64")
    rng = np.random.default_rng(0)

    from src.community_encoder.train_DESK.validate_bbs_routes import load_observed
    X_bbs, keys, bmeta, _raw = load_observed(cfg)
    B = np.asarray(X_bbs[rng.permutation(len(X_bbs))[:600]], dtype="float64")
    Lp = L[rng.permutation(len(L))[:600]]

    names = None
    csv = (cfg.get("desk", {}).get("trend", {}).get("community_trend_list")
           or cfg.get("data", {}).get("community_trend_list"))
    if csv and os.path.exists(csv):
        import pandas as pd
        names = [str(s) for s in pd.read_csv(csv)["species_code"].tolist()]
    if not names or len(names) != L.shape[1]:
        names = [f"sp{i:02d}" for i in range(L.shape[1])]

    print(f"landmarks {L.shape}  BBS {X_bbs.shape}  synthetic share "
          f"{meta.get('landmark_synthetic_share')}")

    # --- per-species presence and mass, both domains ---------------------------------------
    print("\n=== per-species occupancy (share of rows where the species is present) ===")
    pl, pb = (Lp > 0).mean(0), (B > 0).mean(0)
    ml, mb = Lp.mean(0), B.mean(0)
    order = np.argsort(-(np.abs(pl - pb)))
    print(f"  {'species':<14}{'occ_lm':>8}{'occ_bbs':>9}{'d_occ':>8}{'mass_lm':>9}{'mass_bbs':>9}")
    for i in order[:18]:
        print(f"  {names[i]:<14}{pl[i]:>8.3f}{pb[i]:>9.3f}{pb[i] - pl[i]:>+8.3f}"
              f"{ml[i]:>9.3f}{mb[i]:>9.3f}")

    # --- decompose the best match --------------------------------------------------------
    # For each BBS row, find its best landmark and split Sum|x - l| by species and by sign.
    print("\n=== where the mismatched mass sits, at each BBS row's BEST landmark ===")
    shared, only_x, only_l = np.zeros(len(B)), np.zeros(len(B)), np.zeros(len(B))
    per_sp_x, per_sp_l = np.zeros(L.shape[1]), np.zeros(L.shape[1])
    best_r = np.zeros(len(B))
    for i, b in enumerate(B):
        mn = np.minimum(b[None, :], L).sum(1)
        mx = np.maximum(b[None, :], L).sum(1)
        r = np.where(mx > 0, mn / np.maximum(mx, 1e-12), 0.0)
        j = int(np.argmax(r))
        best_r[i] = r[j]
        d = b - L[j]
        shared[i] = np.minimum(b, L[j]).sum()
        only_x[i] = np.clip(d, 0, None).sum()
        only_l[i] = np.clip(-d, 0, None).sum()
        per_sp_x += np.clip(d, 0, None)
        per_sp_l += np.clip(-d, 0, None)
    tot = shared + only_x + only_l
    print(f"  best-landmark R          : median {np.median(best_r):.4f}")
    print(f"  shared mass (Sum-min)    : {np.median(shared):8.3f}  "
          f"{100 * np.median(shared / tot):5.1f}% of Sum-max")
    print(f"  BBS-only mass            : {np.median(only_x):8.3f}  "
          f"{100 * np.median(only_x / tot):5.1f}%   <- mass no landmark has")
    print(f"  landmark-only mass       : {np.median(only_l):8.3f}  "
          f"{100 * np.median(only_l / tot):5.1f}%   <- mass BBS lacks")

    print("\n  species driving BBS-only mass (BBS has it, the best landmark does not):")
    for i in np.argsort(-per_sp_x)[:10]:
        print(f"    {names[i]:<14}{per_sp_x[i] / len(B):8.4f} per row  "
              f"(occ lm {pl[i]:.3f} -> bbs {pb[i]:.3f})")
    print("\n  species driving landmark-only mass (the landmark has it, BBS does not):")
    for i in np.argsort(-per_sp_l)[:10]:
        print(f"    {names[i]:<14}{per_sp_l[i] / len(B):8.4f} per row  "
              f"(occ lm {pl[i]:.3f} -> bbs {pb[i]:.3f})")

    # --- control: how well does BBS match OTHER BBS rows? --------------------------------
    # If BBS rows match each other at ~0.65 the way landmarks do, the geometry is fine and the
    # landmark SET is simply drawn from elsewhere. If BBS rows do not match each other either,
    # BBS communities are intrinsically more variable and no landmark set of this size would
    # cover them -- a different problem with a different fix.
    print("\n=== control: BBS against OTHER BBS rows (same computation, different reference) ===")
    Bref = np.asarray(X_bbs[rng.permutation(len(X_bbs))[:600]], dtype="float64")
    bb = np.zeros(len(B))
    for i, b in enumerate(B):
        mn = np.minimum(b[None, :], Bref).sum(1)
        mx = np.maximum(b[None, :], Bref).sum(1)
        r = np.where(mx > 0, mn / np.maximum(mx, 1e-12), 0.0)
        r[i] = 0.0                                   # never itself
        bb[i] = r.max()
    print(f"  best-BBS-neighbour R : median {np.median(bb):.4f}")
    print(f"  best-landmark R      : median {np.median(best_r):.4f}")
    print()
    if np.median(bb) > 0.5:
        print("  -> BBS rows DO find close neighbours among themselves. The geometry is fine;")
        print("     the landmark set is drawn from a different population. Fix: draw landmarks")
        print("     from (or include) the BBS route communities the oracle has to project.")
    else:
        print("  -> BBS rows do not closely match each other either. These communities are")
        print("     intrinsically more variable than the landmark population, so no landmark")
        print("     set of this size covers them; the community definition is the thing to fix.")


if __name__ == "__main__":
    main()
