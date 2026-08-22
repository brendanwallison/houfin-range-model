"""Why does the SAME ESK basis self-report ||z||^2 = 0.656 on its own data and 0.146 on BBS
route communities?

0.146 against a contract of 1.0 is not a scale nuisance to correct downstream: it says the basis
barely represents the objects being projected, and every z-space comparison involving an observed
BBS community inherits it. Ruzicka is Sum-min / Sum-max and is NOT scale-invariant, so a
systematically smaller vector is dissimilar to EVERY landmark: R(x, landmark) is small for all of
them, the Nystrom estimate of R(x,x) collapses, and ||z||^2 falls even though the EXACT R(x,x) is
still 1. This measures whether that is what is happening.

The reference domain is the LANDMARKS themselves. They are stored as communities -- (n_landmarks,
n_species) -- so they are the basis's domain by definition, and no separate trend-point array has
to be located. If BBS communities sit at a different abundance scale than the landmarks, the
collapse is a scale/domain gap and the fix is in the community construction. If they sit at the
SAME scale and ||z||^2 still collapses, the fix is the basis (landmark coverage or the rank
truncation).

Run on TACC (needs the raw BBS release):
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


def describe(name, X, lm_sample, z_dir, ld, rng, n=600):
    take = rng.permutation(len(X))[:min(n, len(X))]
    S = np.asarray(X[take], dtype="float64")
    tot, nz = S.sum(1), (S > 0).sum(1)
    R = ruzicka_pairs(S, lm_sample)
    self_r = np.diag(ruzicka_pairs(S, S))
    z = project_points_to_z(np.asarray(X[take], dtype="float32"), z_dir, ld)
    z2 = (z ** 2).sum(1) if z is not None else np.array([np.nan])
    print(f"\n=== {name}   n={len(S)} of {len(X)} rows, {S.shape[1]} species")
    print(f"  total abundance per row : median {np.median(tot):8.3f}  "
          f"q10 {np.quantile(tot, .1):8.3f}  q90 {np.quantile(tot, .9):8.3f}")
    print(f"  nonzero species per row : median {np.median(nz):8.1f}  "
          f"q10 {np.quantile(nz, .1):8.1f}  q90 {np.quantile(nz, .9):8.1f}")
    print(f"  R(x, landmark)          : median {np.median(R):8.4f}  "
          f"best-landmark median {np.median(R.max(1)):.4f}")
    print(f"  EXACT R(x,x)            : median {np.median(self_r):8.4f}   (contract: exactly 1.0)")
    print(f"  ||z||^2 from projection : median {np.median(z2):8.4f}   (contract: 1.0)")
    return dict(total=float(np.median(tot)), nz=float(np.median(nz)),
                r_lm=float(np.median(R)), r_best=float(np.median(R.max(1))),
                r_self=float(np.median(self_r)), z2=float(np.median(z2)))


def main():
    cfg = load_config(os.environ.get("ESK_DESK_CONFIG") or None)
    z_dir = cfg["desk"]["z_dir"]
    meta = json.load(open(os.path.join(z_dir, "meta.json"), encoding="utf-8"))
    ld = int(meta["latent_dim"])
    landmarks = np.load(os.path.join(z_dir, "esk_landmarks.npy"))
    rng = np.random.default_rng(0)
    print(f"basis: {z_dir}")
    print(f"       n_species={meta['n_species']} n_weeks={meta['n_weeks']} sigma={meta['sigma']} "
          f"landmarks={landmarks.shape}")
    print(f"       self-reported annual ||z||^2 = {meta.get('median_z_norm2_annual'):.4f}")
    lm_sample = landmarks[rng.permutation(len(landmarks))[:400]].astype("float64")

    res = {}
    # The landmarks ARE the basis domain. Projecting them is also the cleanest possible sanity
    # check on the projection itself: a landmark is in the span by construction, so if its own
    # ||z||^2 is far below 1 the shortfall is rank truncation and nothing to do with BBS.
    res["landmarks"] = describe("LANDMARKS (the basis domain, in-span by construction)",
                                landmarks, lm_sample, z_dir, ld, rng)

    from src.community_encoder.train_DESK.validate_bbs_routes import load_observed
    X_bbs, keys, bmeta, X_raw = load_observed(cfg)
    print(f"\nBBS route communities: shape={X_bbs.shape} ruzicka_log1p={bmeta.get('ruzicka_log1p')}")
    res["bbs_log1p"] = describe("BBS ROUTE COMMUNITIES, log1p (what the oracle projects)",
                                X_bbs, lm_sample, z_dir, ld, rng)
    # And the raw counts, to show which side of the log1p the scale gap (if any) lives on.
    res["bbs_raw"] = describe("BBS ROUTE COMMUNITIES, raw counts (for reference)",
                              X_raw, lm_sample, z_dir, ld, rng)

    lmk, b = res["landmarks"], res["bbs_log1p"]
    print("\n=== verdict ===")
    print(f"  exact R(x,x): landmarks {lmk['r_self']:.4f}  BBS {b['r_self']:.4f}   "
          "(both must be 1.0 -- the kernel is exact here)")
    print(f"  ||z||^2     : landmarks {lmk['z2']:.4f}  BBS {b['z2']:.4f}")
    print(f"  total abund : landmarks {lmk['total']:.3f}  BBS {b['total']:.3f}  "
          f"(ratio {b['total'] / max(lmk['total'], 1e-9):.3f})")
    print(f"  R(x,lm)     : landmarks {lmk['r_lm']:.4f}  BBS {b['r_lm']:.4f}")
    print(f"  best R(x,lm): landmarks {lmk['r_best']:.4f}  BBS {b['r_best']:.4f}")
    print()
    if lmk["z2"] < 0.8:
        print(f"  -> The LANDMARKS themselves project to {lmk['z2']:.3f}, and they are in the span")
        print("     by construction. So the shortfall is the rank-64 truncation, NOT anything")
        print("     about BBS, and it caps every predictor equally. Raising latent_dim is the")
        print("     lever; the BBS-specific gap is whatever remains beyond this.")
    # Scale and coverage are SEPARATE diagnoses with different fixes, so they get separate
    # tests. Reporting them through one branch (as the first version of this did) printed
    # "different scale" on a 0.92 abundance ratio, which is agreement, not a gap.
    scale_gap = b["total"] < 0.6 * lmk["total"] or b["total"] > 1.7 * lmk["total"]
    cover_gap = b["r_best"] < 0.6 * lmk["r_best"]
    if scale_gap:
        print(f"  -> SCALE gap: BBS total abundance {b['total']:.2f} vs landmarks "
              f"{lmk['total']:.2f} (ratio {b['total'] / max(lmk['total'], 1e-9):.2f}). Ruzicka is "
              "not scale-invariant, so this alone collapses ||z||^2. Fix the community "
              "construction -- do NOT rescale z.")
    else:
        print(f"  -> No scale gap: abundance ratio {b['total'] / max(lmk['total'], 1e-9):.2f}, "
              "so Ruzicka's scale sensitivity is NOT the explanation.")
    if cover_gap:
        print(f"  -> COVERAGE gap: the best of {len(landmarks):,} landmarks is only "
              f"{b['r_best']:.3f} similar to a BBS community, against {lmk['r_best']:.3f} for a "
              "landmark against its own neighbours. The landmark set does not reach the region "
              "BBS occupies. Run basis_coverage_gap.py to localise it by species.")
    json.dump(res, open("basis_domain_gap.json", "w"), indent=2)
    print("\n  wrote basis_domain_gap.json")


if __name__ == "__main__":
    main()
