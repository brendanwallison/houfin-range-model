"""Fair held-out scoring across route-effect scales (SIGU follow-up).

Held-out BLOCK rows sit on routes the fit never saw; the standard readout scores them at route effect 0 (the median
route), which flatters small sigma_u by construction. Here an unseen route's count is scored by its PREDICTIVE: the NB
mixed over the unknown route effect, log int NB(y | mu exp(sigma_u z), phi) N(z; 0, 1) dz (Gauss-Hermite, 32 nodes),
mu = the expected count with that route's effect removed. Rows on routes WITH training data (the 2021-25 hold-out) are
scored conditionally on their fitted effect, as before. Splits: pre-1994 blocks, post-1993 blocks, 2021-25; and the
conditional (eps-at-fit) score for comparison. Run through the takeover runner with the arm's overlay:
  TAKEOVER_CODE_ROOT=<wt> AGE_MODEL_CONFIG=<overlay> bash <wt>/research/bin/local_takeover_run.sh \
     /mnt/c/.../marginal_heldout.py --params NPZ --out JSON
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--synthetic_y', default=None, help='score held-out rows against synth_bbs.py counts (known-truth calibration)')
    a = p.parse_args()
    sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
    import jax.numpy as jnp
    from numpyro import handlers
    from scipy.special import gammaln, logsumexp
    from harness import Sandbox, ap
    from src.config_utils import load_age_model_config

    sb = Sandbox(os.environ['SBX_REFINED'])
    if a.synthetic_y:   # overnight 2026-11-07: does held-out favour eta 1 even when the truth is eta 0?
        _ys = np.load(a.synthetic_y)['y']; _yo = np.asarray(sb.data['observed_results'])
        assert _ys.shape == _yo.shape
        sb.data = dict(sb.data); sb.data['observed_results'] = jnp.asarray(np.where(np.isfinite(_yo), _ys, _yo), dtype=jnp.asarray(_yo).dtype)
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
        fitted_sigma = float(z['laplace_route_sigma']) if 'laplace_route_sigma' in z.files else None
        route_mean = float(z['laplace_route_mean']) if 'laplace_route_mean' in z.files else 0.   # laplace_fit --route_mean
        route_skew = float(np.ravel(z['laplace_route_skew'])[0]) if 'laplace_route_skew' in z.files else None   # laplace_fit --route_skew
    tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=lat), rng_seed=104)).get_trace(data=sb.data, prior_scale=1.)
    sigma_cfg = float(load_age_model_config()['population_model']['observation']['observer_effect']['sigma_u'])
    # laplace_fit.py stores eps = e* sigma_hat / sigma_cfg (so sigma_cfg * eps is the fitted effect) and sigma_hat itself
    sigma = fitted_sigma if fitted_sigma is not None else sigma_cfg
    ok = np.asarray(sb.obs_ok)
    y = np.asarray(sb.data['observed_results'])[ok].astype(float)
    mu = np.asarray(tr['expected_obs']['value'])[ok]
    phi = np.asarray(tr['obs_concentration']['value'])[ok]
    held = (np.asarray(sb.data['obs_weight']) <= 0)[ok]
    yr = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])][ok]
    idx = np.asarray(sb.data['obs_obsroute_idx'])[ok]
    eps = np.asarray(lat['observer_route_eps'])
    eff = np.where(idx >= 0, sigma_cfg * eps[np.maximum(idx, 0)], 0.)   # the effect the model applied to mu
    trained = np.zeros(eps.size, bool); trained[idx[(~held) & (idx >= 0)]] = True
    seen = (idx >= 0) & trained[np.maximum(idx, 0)]

    def nb_logpmf(yv, m, ph):
        m = np.maximum(m, 1e-12)
        return gammaln(yv + ph) - gammaln(ph) - gammaln(yv + 1) + ph * np.log(ph / (ph + m)) + yv * np.log(m / (ph + m))

    xg, wg = np.polynomial.hermite_e.hermegauss(32); lw = np.log(wg / wg.sum())
    if route_skew is not None:   # centred skew-normal route prior (mean 0, var 1, shape alpha): reweight the normal nodes
        from scipy.stats import norm
        dl = route_skew / np.sqrt(1. + route_skew ** 2); om = 1. / np.sqrt(1. - 2. * dl ** 2 / np.pi); xi = -om * dl * np.sqrt(2. / np.pi)
        lsn = np.log(2.) - np.log(om) + norm.logpdf((xg - xi) / om) + norm.logcdf(route_skew * (xg - xi) / om)
        lw = lw + lsn - norm.logpdf(xg); lw = lw - logsumexp(lw)
    cond = nb_logpmf(y, mu, phi)
    mu0 = mu * np.exp(-eff + np.where(idx >= 0, route_mean, 0.))   # this route's own effect removed; unseen routes centred on the fitted route mean
    marg = np.array([logsumexp(nb_logpmf(y[i], mu0[i] * np.exp(sigma * xg), phi[i]) + lw) for i in np.flatnonzero(held & ~seen)])
    score = cond.copy(); score[np.flatnonzero(held & ~seen)] = marg
    sets = dict(pre_blocks=held & (yr < 1994), post_blocks=held & (yr >= 1994) & (yr < 2021), y2021_25=held & (yr >= 2021))
    out = dict(params=str(a.params), sigma_u=sigma, n_unseen_heldout=int((held & ~seen).sum()), n_seen_heldout=int((held & seen).sum()),
               marginal_nll={k: float(-score[s].sum()) for k, s in sets.items()},
               conditional_nll={k: float(-cond[s].sum()) for k, s in sets.items()})
    out['marginal_total'] = sum(out['marginal_nll'].values()); out['conditional_total'] = sum(out['conditional_nll'].values())

    # ROUTEMARG: the correct predictive for an unseen route integrates ONE effect shared by all its rows (the per-row
    # mixture above treats each row's effect as independent). Per-route Gauss-Hermite; rows without a route id stay
    # per-row (they have no effect). Diagnostic scan: what (sigma, mean shift c) would the unseen routes prefer?
    un = np.flatnonzero(held & ~seen & (idx >= 0))
    ru, inv = np.unique(idx[un], return_inverse=True)

    def route_nll(sg, c=0., sel=None):
        rows = un if sel is None else un[sel]
        iv = inv if sel is None else inv[sel]
        L = nb_logpmf(y[rows, None], mu0[rows, None] * np.exp(c + sg * xg[None, :]), phi[rows, None])   # rows x nodes
        per_route = np.zeros((ru.size, xg.size)); np.add.at(per_route, iv, L)
        return -float(logsumexp(per_route + lw[None, :], axis=1).sum())
    era = {k: (yr[un] >= lo) & (yr[un] < hi) for k, (lo, hi) in dict(pre_blocks=(0, 1994), post_blocks=(1994, 2021), y2021_25=(2021, 9999)).items()}
    # an unseen route's rows can span eras; the per-era split integrates each era's rows of that route separately
    out['route_marginal_unseen_nll'] = {k: route_nll(sigma, 0., s) for k, s in era.items() if s.any()}
    out['route_marginal_unseen_total'] = route_nll(sigma)
    out['n_unseen_routes'] = int(ru.size)
    grid = [(sg, c, route_nll(sg, c)) for sg in (.6, .8, .97, 1.1, 1.25, 1.4) for c in (-.4, -.2, 0., .2, .4, .6)]
    best = min(grid, key=lambda t: t[2])
    out['unseen_route_scan'] = dict(best_sigma=best[0], best_shift=best[1], best_nll=best[2],
                                    at_fit=route_nll(sigma), grid=[[g[0], g[1], round(g[2], 1)] for g in grid])
    a.out.write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out))


if __name__ == '__main__':
    main()
