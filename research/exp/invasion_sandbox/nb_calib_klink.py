"""K-link check (user: "it could be the transform -- softplus or exp for K?"). On [cur]:
(a) where the softplus argument of K sits (share of occupied cell-years in the sub-exponential regime; softplus/exp ratio);
(b) obs / expected with each row's route effect removed, by expected-count bin, train vs held-out blocks -- does the
    under-prediction GROW with density (softplus compression) or is it flat (a level / placement offset)?"""
import json, os
import numpy as np
from numpyro import handlers
from harness import Sandbox, ap, capture, to_land

sb = Sandbox(os.environ['SBX_REFINED'])
x, _ = capture(sb.data, sb.lat)
T = len(sb.years)
K = to_land(x['K_flat'], sb.rows, sb.cols, T, sb.Nx)
arg = np.log(np.expm1(np.maximum(K, 1e-12)))          # softplus^-1(K): the link argument
ratio = K / np.exp(arg)
occ = np.isfinite(sb.first_obs)
out = dict(K_arg_quantiles_occupied=dict(zip(['q10', 'q50', 'q90', 'q99'], np.quantile(arg[:, occ], [.1, .5, .9, .99]).round(3).tolist())),
           softplus_over_exp_quantiles_occupied=dict(zip(['q10', 'q50', 'q90', 'q99'], np.quantile(ratio[:, occ], [.1, .5, .9, .99]).round(3).tolist())))
tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=sb.lat), rng_seed=104)).get_trace(data=sb.data, prior_scale=1.)
m = np.asarray(sb.obs_ok)
y = np.asarray(sb.data['observed_results'])[m]; mu = np.asarray(tr['expected_obs']['value'])[m]
held = (np.asarray(sb.data['obs_weight']) <= 0)[m]
yr = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])][m]
eps = np.asarray(sb.lat['observer_route_eps'])[np.asarray(sb.data['obs_obsroute_idx'])][m]
mu0 = mu / np.exp(eps)
edges = [0, .1, .5, 1, 3, 10, 30, 1e9]
for kn, ks in (('train', ~held), ('held_blocks', held & (yr < 2021)), ('held_2021', held & (yr >= 2021))):
    rows = {}
    for e0, e1 in zip(edges[:-1], edges[1:]):
        s = ks & (mu0 >= e0) & (mu0 < e1)
        if s.sum() >= 30:
            rows[f'{e0}-{e1}'] = dict(n=int(s.sum()), obs_over_exp_no_eps=round(float(y[s].sum() / mu0[s].sum()), 3),
                                      mean_eps=round(float(eps[s].mean()), 3))
    out[kn] = rows
print(json.dumps(out, indent=1))
