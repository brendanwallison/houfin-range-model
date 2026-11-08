"""population_model.growth_kernel readouts for one or more fitted parameter sets (run under the FLAG-ON overlay used to fit).

  python growth_kernel_readout.py label=params.npz [...] [--json OUT.json]

Per (year, land cell), from the learned kernel (growth_kernel.equilibrium_readout; lambda on the breeder census
lambda = s + Sj rho, Sj the model's juvenile survival):
  lambda0      potential growth as density -> 0 (includes any learned Allee deficit; = Sa_flat + Sj_flat * Fmax_flat)
  lambda_peak  the maximum over density on the equilibrium stage ray (the Allee-free analogue of the mechanistic
               potential Sa + Sj Fmax, but net of whatever crowding acts at that density)
  N*           the largest STABLE equilibrium (total density, route counts) -- the data's implied capacity
  status       share of cells with no positive equilibrium (lambda < 1 at every density) and with lambda > 1 at the top
               of the basis range (equilibrium not inside the observed density range)
Headlines: trough D = 1902-50 mean of [GP mean - ETF mean] (CEC L1 cells, as trough_D2.py) on lambda0 and lambda_peak;
ETF lambda inflation = median ETF 2010 / 1910 (both measures); N* vs the mechanistic departure K (K_flat); interaction
amplitudes (gk_*_wi) and the relative-axis habitat scale amplitude (gk_wk) when present.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402
from harness import Sandbox, ap  # noqa: E402
from src.model import growth_kernel as gkm  # noqa: E402

CELLS = '/mnt/e/Datasets/houfin/grid54/sandbox/ROUTEMARG/trough_cells_cecL1.npz'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--json', default=None)
    p.add_argument('--cells', default=CELLS)
    p.add_argument('--n_grid', type=int, default=60)
    a = p.parse_args()
    assert ap.GROWTH_KERNEL, 'run under the growth_kernel overlay the parameters were fitted with'
    sb = Sandbox(os.environ['SBX_REFINED'])
    cz = np.load(a.cells); GP, ETF = cz['gp'], cz['etf']
    yrs = np.asarray(sb.years).astype(int); T = len(yrs); early = np.flatnonzero((yrs >= 1902) & (yrs <= 1950))
    yi = lambda y: int(np.flatnonzero(yrs == y)[0])
    scale = float(sb.data['pop_scalar'])
    out = {}
    for arg in a.runs:
        lab, path = arg.split('=', 1)
        lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
        with np.load(path) as z:
            lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_')})
        tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(
            data=sb.data, prior_scale=1.)
        v = lambda k: None if k not in tr else jnp.asarray(tr[k]['value'])
        gk = (v('growth_eta_s'), v('growth_A_s'), v('growth_eta_r'), v('growth_A_r'), v('growth_k'),
              v('gk_s_d'), v('gk_r_d'), ap._GK_SPEC)
        Sj = jnp.asarray(tr['Sj_flat']['value']).reshape(T, -1)
        K = np.asarray(tr['K_flat']['value']).reshape(T, -1)
        years = sorted(set(early.tolist()) | {yi(y) for y in (1910, 1950, 1966, 1990, 2010)})
        ro = jax.jit(lambda g, sj: gkm.equilibrium_readout(g + (ap._GK_SPEC,), sj, n_grid=a.n_grid))
        R = {}
        for t in years:
            r = ro(gkm.year_slice(gk, t)[:-1], Sj[t])
            R[t] = {k: np.asarray(x) for k, x in r.items()}
        lam0 = np.asarray(tr['growth_lambda0']['value']).reshape(T, -1)
        D0 = float(np.mean([lam0[t, GP].mean() - lam0[t, ETF].mean() for t in early]))
        Dp = float(np.mean([R[t]['lambda_peak'][GP].mean() - R[t]['lambda_peak'][ETF].mean() for t in early]))
        med = lambda X, m: float(np.median(X[m]))
        res = dict(trough_D_lambda0=D0, trough_D_lambda_peak=Dp,
                   etf_inflation_lambda0=med(lam0[yi(2010)], ETF) / med(lam0[yi(1910)], ETF),
                   etf_inflation_lambda_peak=med(R[yi(2010)]['lambda_peak'], ETF) / med(R[yi(1910)]['lambda_peak'], ETF),
                   amplitudes={k: float(np.asarray(tr[k]['value'])) for k in ('gk_s_wi', 'gk_r_wi', 'gk_s_w0', 'gk_r_w0', 'gk_wk') if k in tr})
        for y in (1910, 1950, 1966, 1990, 2010):
            t = yi(y); r = R[t]
            ns = r['n_star'] * scale; ok = np.isfinite(ns)
            row = {}
            for reg, m in (('all', np.ones(lam0.shape[1], bool)), ('GP', np.isin(np.arange(lam0.shape[1]), GP)),
                           ('ETF', np.isin(np.arange(lam0.shape[1]), ETF))):
                row[reg] = dict(lambda0_med=med(lam0[t], m), lambda_peak_med=med(r['lambda_peak'], m),
                                share_no_equilibrium=float(np.mean(r['status'][m] == 1)),
                                share_above_range=float(np.mean(r['status'][m] == 2)),
                                share_allee_threshold=float(np.mean(np.isfinite(r['allee'][m]))),
                                nstar_med_route_counts=float(np.median(ns[m & ok])) if (m & ok).any() else None,
                                nstar_over_K_med=float(np.median(ns[m & ok] / (K[t][m & ok] * scale))) if (m & ok).any() else None)
            res[str(y)] = row
        out[lab] = res
        g = res
        print(f"{lab:14s} D(lambda0) {D0:+.3f}  D(peak) {Dp:+.3f} | ETF inflation lambda0 x{g['etf_inflation_lambda0']:.2f} "
              f"peak x{g['etf_inflation_lambda_peak']:.2f} | amplitudes {g['amplitudes']}", flush=True)
        for y in (1910, 1950, 2010):
            for reg in ('all', 'GP', 'ETF'):
                q = g[str(y)][reg]
                print(f"   {y} {reg:3s} lambda0 {q['lambda0_med']:.3f} peak {q['lambda_peak_med']:.3f} | no-eq {q['share_no_equilibrium']:.0%} "
                      f"above-range {q['share_above_range']:.0%} Allee-threshold {q['share_allee_threshold']:.0%} | N* med "
                      f"{q['nstar_med_route_counts'] if q['nstar_med_route_counts'] is None else round(q['nstar_med_route_counts'], 3)} rc, "
                      f"N*/K {q['nstar_over_K_med'] if q['nstar_over_K_med'] is None else round(q['nstar_over_K_med'], 2)}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
