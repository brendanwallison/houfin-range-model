"""Known-truth bias test, step 1 (2026-11-06; user: "should I believe the final estimates will be unbiased?").

Draw synthetic BBS counts from a fitted model (the TRUTH): at the truth's state, per real BBS row the route-free mean
mu_i (expected_obs) times exp(sigma * e_r), e_r ~ N(0, 1) per observer x route unit (sigma = the truth's Laplace route
SD), then y ~ NB2(mean, phi_i) with the truth's own concentration. Route-free pseudo-zero rows and held-out rows are
drawn the same way (the fit masks them as usual). Output: npz with y (same length / order as observed_results) and the
truth's summary. Fit afterwards with laplace_fit.py --synthetic_y NPZ under any structure.
Usage (takeover runner, TRUTH overlay): synth_bbs.py --params NPZ --out NPZ [--seed 1]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    import jax.numpy as jnp
    from numpyro import handlers
    from harness import Sandbox, ap
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
        sigma = float(z['laplace_route_sigma']) if 'laplace_route_sigma' in z.files else 1.25
        rmean = float(z['laplace_route_mean']) if 'laplace_route_mean' in z.files else 0.
    lat['observer_route_eps'] = jnp.zeros_like(lat['observer_route_eps'])   # route-free mean; fresh route effects drawn below
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(data=sb.data, prior_scale=1.)
    mu = np.asarray(tr['expected_obs']['value'], float)
    conc = np.asarray(tr['obs_concentration']['value'], float)
    conc = np.broadcast_to(conc, mu.shape) if conc.size > 1 else np.full(mu.shape, float(conc))
    idx = np.asarray(sb.data['obs_obsroute_idx'])
    rng = np.random.default_rng(a.seed)
    n_route = int(idx.max()) + 1
    e = rng.standard_normal(n_route)
    mult = np.where(idx >= 0, np.exp(rmean + sigma * e[np.maximum(idx, 0)]), 1.0)
    m = np.maximum(mu * mult, 1e-12)
    lam = rng.gamma(shape=conc, scale=m / conc)       # NB2 as gamma-Poisson
    y = rng.poisson(lam).astype(float)
    y_obs = np.asarray(sb.data['observed_results'], float)
    y = np.where(np.isfinite(y_obs), y, np.nan)       # keep the real missingness pattern
    np.savez_compressed(a.out, y=y, truth_params=str(a.params), sigma=sigma, route_eps=e, seed=a.seed)
    ok = np.isfinite(y_obs) & np.isfinite(y)
    print(json.dumps(dict(rows=int(ok.sum()), mean_real=round(float(y_obs[ok].mean()), 3), mean_synth=round(float(y[ok].mean()), 3),
                          zero_share_real=round(float((y_obs[ok] == 0).mean()), 3), zero_share_synth=round(float((y[ok] == 0).mean()), 3),
                          sigma=sigma)))


if __name__ == '__main__':
    main()
