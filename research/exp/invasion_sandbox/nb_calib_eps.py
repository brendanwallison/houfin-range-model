"""NB calibration addendum 2: how much of the in-sample mean fit is carried by observer-route effects (eps)?"""
import json, os
import numpy as np
from numpyro import handlers
from harness import Sandbox, ap

sb = Sandbox(os.environ['SBX_REFINED'])
tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=sb.lat), rng_seed=104)).get_trace(data=sb.data, prior_scale=1.)
m = np.asarray(sb.obs_ok)
y = np.asarray(sb.data['observed_results'])[m]; mu = np.asarray(tr['expected_obs']['value'])[m]
held = (np.asarray(sb.data['obs_weight']) <= 0)[m]
yr = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])][m]
eps = np.asarray(sb.lat['observer_route_eps'])[np.asarray(sb.data['obs_obsroute_idx'])][m]
reg = {r: np.asarray(sb.region[r]) for r in ('east', 'plains', 'west')}
out = dict(eps_held_rows=dict(mean=float(eps[held].mean()), sd=float(eps[held].std()), share_zero=float((np.abs(eps[held]) < 1e-6).mean())),
           eps_train_rows=dict(mean=float(eps[~held].mean()), sd=float(eps[~held].std()),
                               count_weighted_mean=float((eps[~held] * y[~held]).sum() / max(y[~held].sum(), 1))))
mu0 = mu / np.exp(eps)   # expected count with this route's effect removed
for kn, ks in (('train', ~held), ('held_2021', held & (yr >= 2021)), ('held_blocks', held & (yr < 2021))):
    out[kn] = dict(obs_over_exp=float(y[ks].sum() / mu[ks].sum()), obs_over_exp_no_eps=float(y[ks].sum() / mu0[ks].sum()),
                   obs_over_exp_no_eps_x_lognormal=float(y[ks].sum() / (mu0[ks].sum() * np.exp(.5 * np.asarray(sb.lat['observer_route_eps']).var()))))
print(json.dumps(out, indent=1))
