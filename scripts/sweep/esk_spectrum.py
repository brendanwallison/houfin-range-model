#!/usr/bin/env python
"""Print the ESK basis's eigenvalue spectrum and its ADJACENT GAPS.

Why this number matters. The NestedLoRA ordering guarantee can only separate component l from l+1
if their eigenvalues are distinguishable, and that resolution degrades fast as the basis widens.
Measured on synthetic data at a fixed condition number (tests/test_nested_lora.py,
test_the_ordering_guarantee_collapses_once_the_basis_is_wide): an adjacent ratio of 1.85x orders to
min |cos| 0.96, 1.34x collapses to 0.03, 1.08x to 0.001. So the question "how many of DESK's 64
components could be ordered in principle" is answered by where ESK's own adjacent ratios fall.

It also bears on positional truncation. Ingest keeps z[..., :24]; components inside a near-degenerate
block have no meaningful order, so truncating in the middle of one keeps an arbitrary subset of it.

    python scripts/sweep/esk_spectrum.py --dir $HOUFIN_PROCESSED/encoder/esk_balanced/spacetime

Two independent derivations are printed and CROSS-CHECKED, because neither is stored directly:
  (a) from esk_projmat: the projection is ``U * rsqrt(L)`` with U orthonormal, so a column's norm is
      ``1/sqrt(lambda_l)`` and ``lambda_l = 1 / ||col_l||^2``.
  (b) from Z: the projected coordinates' per-column second moment.
If they disagree the script says so and trusts NEITHER -- assumption (a) is read off a source comment,
not a stored value, and a spectrum quietly derived from a wrong assumption is worse than none.
"""
import argparse
import os

import numpy as np


def spectrum_from_projmat(pm):
    """``lambda_l = 1/||col_l||^2``. Returns (lambda, ok, why)."""
    n = np.linalg.norm(pm, axis=0)
    if not np.all(np.isfinite(n)) or np.any(n <= 0):
        return None, False, "projmat has a zero or non-finite column norm"
    return 1.0 / n ** 2, True, ""


def spectrum_from_z(z):
    """Per-column second moment of the projected coordinates."""
    return np.mean(np.asarray(z, dtype=np.float64) ** 2, axis=0)


def report(lam, label, gaps=(1.85, 1.34, 1.08)):
    lam = np.asarray(lam, dtype=np.float64)
    L = len(lam)
    desc = np.all(np.diff(lam) <= 0)
    print(f"\n--- {label}: {L} components, descending={desc} ---")
    print("  idx  eigenvalue   ratio to next")
    for i in range(L):
        r = (lam[i] / lam[i + 1]) if i + 1 < L and lam[i + 1] > 0 else float("nan")
        mark = ""
        if np.isfinite(r):
            mark = "  <-- orderable" if r >= 1.85 else ("  (marginal)" if r >= 1.34 else "")
        print(f"  {i:>3}  {lam[i]:>10.4g}  {r:>8.3f}{mark}")
        if i == 15 and L > 24:
            print(f"  ... (showing to {min(L, 32) - 1})")
            if L > 32:
                for j in range(16, 32):
                    rj = (lam[j] / lam[j + 1]) if j + 1 < L and lam[j + 1] > 0 else float("nan")
                    print(f"  {j:>3}  {lam[j]:>10.4g}  {rj:>8.3f}")
                print(f"  ... components 32..{L - 1} omitted")
            break
    ratios = lam[:-1] / np.maximum(lam[1:], 1e-300)
    print()
    for g in gaps:
        # the length of the LEADING run whose adjacent gaps all clear g -- a prefix, because
        # positional ordering is only meaningful from the top down
        k = 0
        while k < len(ratios) and ratios[k] >= g:
            k += 1
        print(f"  adjacent ratio >= {g:>5.2f} for the leading {k:>2} gap(s) "
              f"-> top {k + 1 if k else 1} component(s) separated at that level")
    print(f"  median adjacent ratio over all {len(ratios)}: {np.median(ratios):.4f}")
    print(f"  condition number lambda_0/lambda_{L-1}: {lam[0] / max(lam[-1], 1e-300):.3g}")
    return ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--z-rows", type=int, default=200000,
                    help="cap on rows read from Z for the second-moment estimate")
    a = ap.parse_args()
    d = os.path.expandvars(a.dir)

    pm_p = os.path.join(d, "esk_projmat.npy")
    pm = np.load(pm_p)
    print(f"esk_projmat {pm.shape} {pm.dtype}")
    lam_a, ok, why = spectrum_from_projmat(pm)
    if not ok:
        print(f"  cannot derive from projmat: {why}")

    lam_b = None
    z_p = os.path.join(d, "Z.npy")
    if os.path.exists(z_p):
        z = np.load(z_p, mmap_mode="r")
        print(f"Z {z.shape} {z.dtype}")
        zz = np.asarray(z).reshape(-1, z.shape[-1])
        fin = np.isfinite(zz).all(axis=1)
        zz = zz[fin]
        if len(zz) > a.z_rows:
            zz = zz[np.random.default_rng(0).choice(len(zz), a.z_rows, replace=False)]
        print(f"  second moment over {len(zz):,} finite rows")
        lam_b = spectrum_from_z(zz)

    if lam_a is not None and lam_b is not None and len(lam_a) == len(lam_b):
        # scale is convention-dependent, so compare SHAPE: correlation of logs and rank agreement
        la, lb = np.log(np.maximum(lam_a, 1e-300)), np.log(np.maximum(lam_b, 1e-300))
        r = float(np.corrcoef(la, lb)[0, 1])
        same_order = int(np.sum(np.argsort(-lam_a) == np.argsort(-lam_b)))
        print(f"\nCROSS-CHECK projmat vs Z: log-log correlation {r:+.4f}, "
              f"{same_order}/{len(lam_a)} components in the same rank position")
        if r < 0.9:
            print("  DISAGREE. The projmat derivation rests on proj_mat = U * rsqrt(L), read off a\n"
                  "  source comment rather than a stored value. Do not trust either spectrum below\n"
                  "  until that is confirmed against esk_kernel.py in this tree.")
        else:
            print("  agree in shape, so the derivation holds and the gaps below are meaningful")

    if lam_b is not None:
        report(np.sort(lam_b)[::-1], "spectrum from Z (second moment, as trained)")
    if lam_a is not None:
        report(np.sort(lam_a)[::-1], "spectrum from esk_projmat (1/||col||^2)")


if __name__ == "__main__":
    main()
