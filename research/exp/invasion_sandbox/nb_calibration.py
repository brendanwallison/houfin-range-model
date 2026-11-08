"""NB calibration (user: "is the NB dispersion becoming Poisson-like and the model over-fitting?").

[cur]'s BBS likelihood is NB2 with a density-dependent concentration phi(mu) = phi0 * mu^kappa (observation.conc_density_power)
-- none of the sandbox refits free phi0 / kappa. Readout at the fit, by expected-count bin, for TRAINING rows and the
observation_mask HELD-OUT rows (2021-25 + 31 blocks, never seen by the fit):
  mean Pearson residual^2 (y - mu)^2 / Var_NB  (1 = calibrated, > 1 = the NB is too tight -> over-confident fit),
  coverage of the central 90% NB interval, observed / expected zero share, and the implied variance / mean ratio.
Usage (inside sbx_env.sh): python nb_calibration.py --out FILE.json
"""
import argparse, json, os
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy import stats
from harness import Sandbox, ap


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--params', type=Path, default=None, help='score this fit (best_params.npz; route effects at their saved mode)')
    a = p.parse_args()
    sb = Sandbox(os.environ['SBX_REFINED'])
    if a.params is not None:
        sb.lat = dict(sb.lat)
        with np.load(a.params) as z:
            sb.lat.update({k: np.asarray(z[k]) for k in z.files if not k.startswith('laplace_')})
    tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=sb.lat), rng_seed=104)).get_trace(
        data=sb.data, prior_scale=1.)
    m = np.asarray(sb.obs_ok)
    y = np.asarray(sb.data['observed_results'])[m]
    mu = np.asarray(tr['expected_obs']['value'])[m]; phi = np.asarray(tr['obs_concentration']['value'])[m]
    ow = sb.data.get('obs_weight')
    held = ((np.asarray(ow) <= 0) if ow is not None else np.zeros(m.size, bool))[m]
    var = mu + mu ** 2 / phi
    pr2 = (y - mu) ** 2 / var
    # NB2(mean mu, conc phi) = scipy nbinom(n = phi, p = phi / (phi + mu))
    n_, p_ = phi, phi / (phi + mu)
    lo, hi = stats.nbinom.ppf(.05, n_, p_), stats.nbinom.ppf(.95, n_, p_)
    cov = (y >= lo) & (y <= hi)
    p0 = stats.nbinom.pmf(0, n_, p_)
    out = dict(phi0=float(np.ravel(np.asarray(sb.lat['concentration']))[0]),
               kappa=float(np.ravel(np.asarray(sb.lat.get('obs_conc_kappa', 0.)))[0]), bins={})
    edges = [0, .1, .5, 1, 3, 10, 30, 100, 1e9]
    for lab, sel in (('train', ~held), ('heldout', held)):
        rows = {}
        for e0, e1 in zip(edges[:-1], edges[1:]):
            s = sel & (mu >= e0) & (mu < e1)
            if s.sum() < 20:
                continue
            rows[f'{e0}-{e1}'] = dict(n=int(s.sum()), mean_mu=float(mu[s].mean()), mean_y=float(y[s].mean()),
                                      var_over_mean_model=float((var[s] / mu[s]).mean()),
                                      var_over_mean_obs=float(((y[s] - mu[s]) ** 2).mean() / mu[s].mean()),
                                      pearson_r2=float(pr2[s].mean()), pearson_r2_median=float(np.median(pr2[s])),
                                      cover90=float(cov[s].mean()), zero_obs=float((y[s] == 0).mean()), zero_model=float(p0[s].mean()))
        rows['all'] = dict(n=int(sel.sum()), pearson_r2=float(pr2[sel].mean()), cover90=float(cov[sel].mean()),
                           zero_obs=float((y[sel] == 0).mean()), zero_model=float(p0[sel].mean()))
        out['bins'][lab] = rows
    a.out.write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
