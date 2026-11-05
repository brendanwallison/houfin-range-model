"""R5 inverse diagnostic: which TIME SHAPE does a fixed habitat need?

Habitat frozen at its 1966-2025 mean, release only (West removed). A piecewise-linear multiplier m(t) on ONE eastern
vital-rate array (K or Fmax; knots below, log-parameterised) plus a constant Fmax scale is fitted by Nelder-Mead to:
arrival offsets by BBS cohort -> 0 (sd 3 yr), the 1959 / 1965 CBC p90 radius (153 / 267 km, log sd .3), and eastern
BBS-row abundance per era -> the reference (fitted) arm's (log sd .2; the fit's level stands in for the data's -- this
is a diagnostic of shape, not an estimate). The inferred m(t) shape is the readout: a step near 1960 points to a
behavioural change, a rise tracking the colonised area to a density/Allee-type mechanism, a smooth secular rise to real
capacity change.
Usage (inside sbx_env.sh): python inverse.py --field K|Fmax --out DIR [--maxiter 300]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from harness import Sandbox

KNOTS = [1940, 1955, 1965, 1975, 1985, 1995, 2010, 2025]
CBC = {1959: 153., 1965: 267.}


def arm_for(theta, field):
    vals = list(np.exp(theta[:len(KNOTS)])); fs = float(np.exp(theta[-1]))
    return dict(name='inv', region='east_only', habitat_alpha=0, fmax_scale=fs,
                rate_edits=[dict(field=field, region='east', form='spline', knots=KNOTS, values=vals)])


def loss_bbs(row):
    """R5c: minus the East BBS log-likelihood under the fit's own observation model (arrival timing is carried by the
    zero / positive rows), no unit conversion, no hand weights."""
    parts = {k: -v['loglik'] for k, v in row['abundance'].items() if k.startswith('east')}
    return float(sum(parts.values())), parts


def loss(row, cbc_curve='abs_0.01_birds_km2'):
    if cbc_curve == 'bbs':
        return loss_bbs(row)
    off = row['curves']['rel_10pct']['arrival_offset']['by_cohort']
    L = 0.; parts = {}
    for k, v in off.items():
        parts[k] = ((v if v is not None else 30.) / 3.) ** 2
    yrs = row['years']; R = row['curves'][cbc_curve]['radius90_km']
    for y, r in CBC.items():
        parts[f'cbc{y}'] = (np.log(max(R[yrs.index(y)], 10.) / r) / .3) ** 2
    for k, v in row['abundance'].items():
        if k.startswith('east'):
            m = v['model_over_reference']
            parts[k] = (np.log(max(m, 1e-3)) / .2) ** 2 if m is not None else 100.
    return float(sum(parts.values())), parts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--field', default='K'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--maxiter', type=int, default=300); p.add_argument('--fscale0', type=float, default=.4)
    p.add_argument('--cbc_curve', default='abs_0.01_birds_km2', help="or 'bbs' (R5c: East BBS log-lik only)"); p.add_argument('--step', type=float, default=.3)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import os
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))           # reference arm: thresholds + abundance scale
    L0, P0 = loss(ref, a.cbc_curve)
    log = (a.out / f'inverse_{a.field}.jsonl').open('w')
    log.write(json.dumps(dict(arm='fitted', loss=L0, parts=P0)) + '\n')
    best = dict(L=np.inf)
    def f(theta):
        row = sb.run(arm_for(theta, a.field)); L, parts = loss(row, a.cbc_curve)
        rec = dict(theta=list(map(float, theta)), loss=L, parts=parts, sha=row['final_state_sha256'][:12],
                   offsets=row['curves']['rel_10pct']['arrival_offset']['by_cohort'],
                   east_abund=[row['abundance'][k]['model_over_reference'] for k in row['abundance'] if k.startswith('east')])
        log.write(json.dumps(rec) + '\n'); log.flush()
        if L < best['L']:
            best.update(L=L, rec=rec)
        return L
    theta0 = np.r_[np.zeros(len(KNOTS)), np.log(a.fscale0)]
    t0 = time.time()
    # explicit simplex: scipy default perturbs zero entries by only 2.5e-4, which froze every knot in the first R5
    simplex = np.vstack([theta0] + [theta0 + a.step * np.eye(len(theta0))[i] for i in range(len(theta0))])
    res = minimize(f, theta0, method='Nelder-Mead', options=dict(maxiter=a.maxiter, xatol=1e-2, fatol=1e-2,
                                                                  initial_simplex=simplex, adaptive=True))
    out = dict(field=a.field, knots=KNOTS, fitted_reference_loss=L0, best_loss=best['L'],
               best_multiplier=list(np.exp(best['rec']['theta'][:len(KNOTS)])),
               best_fmax_scale=float(np.exp(best['rec']['theta'][-1])), best=best['rec'],
               n_eval=res.nfev, minutes=round((time.time() - t0) / 60, 1))
    (a.out / f'inverse_{a.field}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('field', 'fitted_reference_loss', 'best_loss', 'best_multiplier',
                                          'best_fmax_scale', 'n_eval', 'minutes')}))


if __name__ == '__main__':
    main()
