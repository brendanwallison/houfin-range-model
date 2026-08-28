"""The ESK basis, examined three ways. No encoder, no training, no GPU.

Consolidated from what used to be two scripts, because all three questions need the same two
things -- the saved basis and some communities to project -- and splitting them meant loading the
same artifacts twice and having two places to look.

**1. Domain gap.** Why does the SAME ESK basis self-report ||z||^2 = 0.656 on its own data and
0.146 on BBS route communities? 0.146 against a contract of 1.0 is not a scale nuisance to correct
downstream: it says the basis barely represents the objects being projected, and every z-space
comparison involving an observed BBS community inherits it. Ruzicka is Sum-min / Sum-max and is NOT
scale-invariant, so a systematically smaller vector is dissimilar to EVERY landmark: R(x, landmark)
is small for all of them, the Nystrom estimate of R(x,x) collapses, and ||z||^2 falls even though
the EXACT R(x,x) is still 1.

The reference domain is the LANDMARKS themselves. They are stored as communities --
(n_landmarks, n_species) -- so they are the basis's domain by definition. If BBS communities sit at
a different abundance scale than the landmarks, the collapse is a scale/domain gap and the fix is in
the community construction. If they sit at the SAME scale and ||z||^2 still collapses, the fix is
the basis (landmark coverage or the rank truncation).

**2. Spectrum and adjacent gaps.** Measured 2026-08-27: the median adjacent eigenvalue ratio is
1.0295, only the top 2 components clear 1.85x, and the effective rank is ~6. That is what closed the
NestedLoRA question -- its ordering guarantee needs adjacent eigenvalues to be resolvable, and at
1.34x the recovery already collapses (tests/test_nested_lora.py). It also explains why every swept
configuration shows 21-30 spectrum inversions: the target ordering barely exists.

**3. Rank curve on HELD-OUT communities.** Does the basis tail carry real signal? The trainer's rank
curve is on DESK's z, so a flat result cannot distinguish a noisy basis tail from a real tail the
covariates cannot predict -- opposite conclusions (cut latent_dim vs the encoder is the ceiling).
ESK is an explicit eigenvalue-descending decomposition, so its own curve separates them. Same
estimand as the trainer's metric (MSE of dot(z_i,z_j) against exact Ruzicka), but on BBS route
communities rather than the trend-grid val pool, so read the SHAPE -- which rank wins -- and not the
absolute value against the sweep's k@24.

    CAVEAT, and read section 1 first: if ||z||^2 on BBS communities is still collapsed, the basis
    barely represents these points and the rank curve here is measuring that rather than the tail.

Run on TACC (needs the raw BBS release). load_observed reads ~6.9M species-route-years, so use a
compute node rather than a login node -- an interactive `idev -p vm-small -t 00:30:00` is enough,
and no GPU is involved anywhere in this script:
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



def spectrum(z_dir, projmat):
    """Eigenvalue spectrum and adjacent gaps, derived TWO ways and cross-checked.

    Neither is stored. (a) proj_mat = U * rsqrt(L) with U orthonormal, so a column's norm is
    1/sqrt(lambda_l). (b) Z's per-column second moment. Scale conventions differ, so they are
    compared on SHAPE -- log-log correlation and rank agreement. On disagreement this trusts NEITHER
    and says why: (a) rests on a source comment, not a stored value, and a spectrum quietly derived
    from a wrong assumption is worse than none.
    """
    n = np.linalg.norm(projmat, axis=0)
    lam_a = 1.0 / n ** 2 if np.all(np.isfinite(n)) and np.all(n > 0) else None
    lam_b = None
    zp = os.path.join(z_dir, "Z.npy")
    if os.path.exists(zp):
        zz = np.asarray(np.load(zp, mmap_mode="r")).reshape(-1, projmat.shape[1])
        zz = zz[np.isfinite(zz).all(axis=1)]
        lam_b = np.mean(zz.astype("float64") ** 2, axis=0)
    print("\n=== 2. SPECTRUM AND ADJACENT GAPS ===")
    if lam_a is not None and lam_b is not None:
        la, lb = np.log(np.maximum(lam_a, 1e-300)), np.log(np.maximum(lam_b, 1e-300))
        r = float(np.corrcoef(la, lb)[0, 1])
        agree = int(np.sum(np.argsort(-lam_a) == np.argsort(-lam_b)))
        print(f"  cross-check projmat vs Z: log-log r={r:+.4f}, {agree}/{len(lam_a)} components in "
              f"the same rank position")
        if r < 0.9:
            print("  DISAGREE -- trust NEITHER. The projmat identity is read off a source comment,\n"
                  "  not a stored value; confirm it against esk_kernel.py before reading the gaps.")
            return None
        if agree < len(lam_a) // 2:
            print(f"  NOTE only {agree}/{len(lam_a)} ranks match between two derivations of the SAME\n"
                  f"  basis. That IS the degeneracy, measured directly: near-tied eigenvalues get\n"
                  f"  ordered differently by two estimators.")
    lam = np.sort(lam_b if lam_b is not None else lam_a)[::-1]
    ratios = lam[:-1] / np.maximum(lam[1:], 1e-300)
    for g, note in ((1.85, "orders (min|cos| 0.96 in the toy)"),
                    (1.34, "COLLAPSES to 0.03"),
                    (1.08, "no ordering at all, 0.001")):
        k = 0
        while k < len(ratios) and ratios[k] >= g:
            k += 1
        print(f"  adjacent ratio >= {g:.2f}: leading {k:>2} gap(s) -> top {max(k + 1, 1):>2} "
              f"component(s)   [{note}]")
    p = lam / lam.sum()
    print(f"  median adjacent ratio {np.median(ratios):.4f} | effective rank "
          f"{1 / np.sum(p ** 2):.1f} of {len(lam)} | condition number "
          f"{lam[0] / max(lam[-1], 1e-300):.3g}")
    print(f"  component 0 holds {p[0]:.1%} of the variance")
    return lam


def rank_curve(name, X, z_dir, ld, rng, ranks=(8, 16, 24, 32, 48, 64), n=700):
    """MSE of dot(z_i[:r], z_j[:r]) against EXACT Ruzicka, over all pairs in a sample.

    Same estimand as the trainer's kernel metric (_pair_kernel_loss), so the SHAPE is comparable to
    the sweep's rank curve even though the point set differs. All off-diagonal pairs of one sample
    rather than sampled index pairs: the sample is the only randomness, so the curve across ranks
    moves only with rank.
    """
    # SUBSAMPLE FIRST, then project. Projecting all of X and slicing afterwards builds an
    # (len(X) x n_landmarks) kernel block -- 16,000 landmarks against 16,000 landmarks is a 1.0 GB
    # float32 tensor before the numerator/denominator copies, which OOMed on a login node. Only `n`
    # rows are ever used, so projecting the rest is pure waste: at n=700 the block is ~45 MB.
    # `describe` above already had this order right; this function did not.
    take = rng.permutation(len(X))[:min(n, len(X))]
    S = np.asarray(X[take], dtype="float64")
    z = project_points_to_z(np.asarray(X[take], dtype="float32"), z_dir, ld)
    if z is None:
        print(f"\n  {name}: no saved projection, rank curve unavailable")
        return None
    zt = np.asarray(z, dtype="float64")
    R = ruzicka_pairs(S, S)
    iu = np.triu_indices(len(S), k=1)
    r_true = R[iu]
    out = {}
    for r in ranks:
        rr = min(int(r), zt.shape[1])
        pred = (zt[:, :rr] @ zt[:, :rr].T)[iu]
        out[int(r)] = float(np.mean((pred - r_true) ** 2))
    best = min(out, key=out.get)
    full = max(out)
    pen = 100.0 * (out[full] / max(out[best], 1e-12) - 1.0)
    print(f"\n  {name}  ({len(S)} points, {len(r_true):,} pairs)")
    print("    " + "  ".join(f"r{k}={out[k]:.5f}" for k in sorted(out)))
    if pen > 1.0:
        print(f"    best rank {best}; keeping all {full} costs {pen:+.1f}% -- the tail is NOISE in "
              f"the basis itself,\n    so latent_dim is wider than the data supports and a low "
              f"per-run bestR is not an encoder failure.")
    else:
        print(f"    best rank {best}; keeping all {full} costs {pen:+.1f}% -- the tail is REAL in "
              f"the basis,\n    so a DESK bestR far below {full} means the ENCODER is not reaching "
              f"it, and latent_dim is not the problem.")
    return out


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
    print("\n=== 1. DOMAIN GAP ===")
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
    res["spectrum"] = None
    _pm = np.load(os.path.join(z_dir, "esk_projmat.npy"))
    _lam = spectrum(z_dir, _pm)
    if _lam is not None:
        res["spectrum"] = [float(v) for v in _lam]

    print("\n=== 3. RANK CURVE ON HELD-OUT COMMUNITIES ===")
    if b["z2"] < 0.5:
        print(f"  WARNING: BBS ||z||^2 is {b['z2']:.3f} against a contract of 1.0, so the basis "
              f"barely represents\n  these points. The BBS curve below is partly measuring THAT, "
              f"not the tail. The landmark curve\n  is the clean one -- but landmarks are the "
              f"Nystrom fitting set, so it is in-sample.")
    res["rank_curve_landmarks"] = rank_curve(
        "LANDMARKS (in-sample for the fit: an upper bound on what the tail can do)",
        landmarks, z_dir, ld, rng)
    res["rank_curve_bbs"] = rank_curve(
        "BBS ROUTE COMMUNITIES (out-of-sample for the fit)", X_bbs, z_dir, ld, rng)

    json.dump(res, open("basis_domain_gap.json", "w"), indent=2)
    print("\n  wrote basis_domain_gap.json")


if __name__ == "__main__":
    main()
