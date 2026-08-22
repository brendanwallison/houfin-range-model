"""Why does the SAME ESK basis self-report ||z||^2 = 0.656 on its own fitting data and 0.146 on
BBS route communities?

That gap is not a metric nuisance: 0.146 against a contract of 1.0 means the basis barely
represents the objects the oracle projects, and every z-space comparison involving an observed
BBS community inherits it. Ruzicka is Sum-min / Sum-max, which is NOT scale-invariant, so a
systematically smaller vector is "dissimilar to everything" -- R(x, landmark) is small for every
landmark, the Nystrom estimate of R(x,x) collapses, and ||z||^2 falls even though the true
R(x,x) = 1 exactly. This measures whether that is what is happening.

Prints, for the trend points the basis was fitted on and for the BBS route communities:
  - total abundance per row (the scale Ruzicka is sensitive to)
  - nonzero species per row (sparsity)
  - R(x, landmark) over a landmark sample -- how close each domain sits to the basis
  - ||z||^2 from the saved projection, and the EXACT R(x,x) = 1 check

Run on TACC (needs the raw BBS release and the processed trend points):
    cd $HOUFIN_REPO && python scripts/diagnostics/basis_domain_gap.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.community_encoder.train_DESK.config_utils import load_config          # noqa: E402
from src.community_encoder.train_DESK.esk_kernel import project_points_to_z     # noqa: E402


def ruzicka_pairs(A, B):
    """Exact uncentered Ruzicka between every row of A and every row of B."""
    out = np.zeros((len(A), len(B)), dtype="float64")
    for i, a in enumerate(A):
        mn = np.minimum(a[None, :], B).sum(1)
        mx = np.maximum(a[None, :], B).sum(1)
        out[i] = np.where(mx > 0, mn / np.maximum(mx, 1e-12), 0.0)
    return out


def describe(name, X, landmarks, z_dir, ld, rng, n=600):
    take = rng.permutation(len(X))[:min(n, len(X))]
    S = np.asarray(X[take], dtype="float64")
    tot, nz = S.sum(1), (S > 0).sum(1)
    lm = landmarks[rng.permutation(len(landmarks))[:400]].astype("float64")
    R = ruzicka_pairs(S, lm)
    self_r = np.diag(ruzicka_pairs(S, S))
    z = project_points_to_z(np.asarray(X[take], dtype="float32"), z_dir, ld)
    z2 = (z ** 2).sum(1) if z is not None else np.array([np.nan])
    print(f"\n=== {name}  (n={len(S)} of {len(X)} rows, {S.shape[1]} species)")
    print(f"  total abundance per row : median {np.median(tot):8.3f}   "
          f"q10 {np.quantile(tot, .1):7.3f}  q90 {np.quantile(tot, .9):8.3f}")
    print(f"  nonzero species per row : median {np.median(nz):8.1f}   "
          f"q10 {np.quantile(nz, .1):7.1f}  q90 {np.quantile(nz, .9):8.1f}")
    print(f"  R(x, landmark)          : median {np.median(R):8.4f}   "
          f"max-over-landmarks median {np.median(R.max(1)):.4f}")
    print(f"  EXACT R(x,x)            : median {np.median(self_r):8.4f}   "
          f"(contract: exactly 1.0)")
    print(f"  ||z||^2 from projection : median {np.median(z2):8.4f}   "
          f"(contract: 1.0; Nystrom can only under-estimate)")
    return dict(total=float(np.median(tot)), nz=float(np.median(nz)),
                r_lm=float(np.median(R)), r_self=float(np.median(self_r)),
                z2=float(np.median(z2)))


def main():
    cfg = load_config(os.environ.get("ESK_DESK_CONFIG") or None)
    z_dir = cfg["desk"]["z_dir"]
    meta = json.load(open(os.path.join(z_dir, "meta.json"), encoding="utf-8"))
    ld = int(meta["latent_dim"])
    landmarks = np.load(os.path.join(z_dir, "esk_landmarks.npy"))
    rng = np.random.default_rng(0)
    print(f"basis   : {z_dir}")
    print(f"          n_species={meta['n_species']} n_weeks={meta['n_weeks']} "
          f"sigma={meta['sigma']} landmarks={landmarks.shape}")
    print(f"          self-reported annual ||z||^2 = {meta.get('median_z_norm2_annual')}")
    print(f"landmark total abundance: median {np.median(landmarks.sum(1)):.3f}  "
          f"q10 {np.quantile(landmarks.sum(1), .1):.3f}  "
          f"q90 {np.quantile(landmarks.sum(1), .9):.3f}")

    # 1. the communities the basis was FITTED on
    pts_dir = cfg.get("trend", {}).get("points_dir", "")
    Xt = None
    for cand in (os.path.join(pts_dir, "X.npy"), os.path.join(pts_dir, "trend_points_X.npy"),
                 os.path.join(z_dir, "X.npy")):
        if os.path.exists(cand):
            Xt = np.load(cand, mmap_mode="r")
            print(f"\ntrend points: {cand}  shape={Xt.shape}")
            break
    if Xt is None:
        print(f"\n!! no trend-point community array found under {pts_dir} or {z_dir}; "
              "listing what IS there so the right file can be named:")
        for d in (pts_dir, z_dir):
            if os.path.isdir(d):
                print(f"   {d}: {sorted(os.listdir(d))[:25]}")

    # 2. the BBS route communities the oracle projects
    from src.community_encoder.train_DESK.validate_bbs_routes import load_observed
    X_bbs, keys, bmeta = load_observed(cfg)
    print(f"\nBBS route communities: shape={X_bbs.shape}  "
          f"ruzicka_log1p={bmeta.get('ruzicka_log1p')}")

    res = {}
    if Xt is not None:
        res["trend_points"] = describe("TREND POINTS (basis fitted on these)",
                                       np.asarray(Xt), landmarks, z_dir, ld, rng)
    res["bbs_routes"] = describe("BBS ROUTE COMMUNITIES (the oracle projects these)",
                                 X_bbs, landmarks, z_dir, ld, rng)

    print("\n=== verdict ===")
    b = res["bbs_routes"]
    print(f"  BBS exact R(x,x) = {b['r_self']:.4f} (must be 1.0) but ||z||^2 = {b['z2']:.4f}")
    if "trend_points" in res:
        t = res["trend_points"]
        print(f"  total abundance  : trend {t['total']:.3f} vs BBS {b['total']:.3f}  "
              f"(ratio {b['total'] / max(t['total'], 1e-9):.3f})")
        print(f"  R(x, landmark)   : trend {t['r_lm']:.4f} vs BBS {b['r_lm']:.4f}")
        print(f"  ||z||^2          : trend {t['z2']:.4f} vs BBS {b['z2']:.4f}")
        print("\n  If BBS total abundance is much smaller AND R(x,landmark) is much smaller,")
        print("  the basis is being asked to represent vectors it has no nearby landmark for:")
        print("  a DOMAIN/SCALE gap, fixable by matching the scale, not by rescaling z.")
        print("  If the abundances match but ||z||^2 still collapses, it is the basis itself")
        print("  (rank/landmark coverage) and the fix is a different one.")
    json.dump(res, open("basis_domain_gap.json", "w"), indent=2)
    print("\n  wrote basis_domain_gap.json")


if __name__ == "__main__":
    main()
