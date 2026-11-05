"""NB calibration addendum: observed / expected and Pearson by held-out kind (2021-25 vs pinned blocks) x region x era."""
import json, os, sys
import numpy as np
from numpyro import handlers
from harness import Sandbox, ap

sb = Sandbox(os.environ['SBX_REFINED'])
tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=sb.lat), rng_seed=104)).get_trace(data=sb.data, prior_scale=1.)
m = np.asarray(sb.obs_ok)
y = np.asarray(sb.data['observed_results'])[m]; mu = np.asarray(tr['expected_obs']['value'])[m]
phi = np.asarray(tr['obs_concentration']['value'])[m]; var = mu + mu ** 2 / phi
held = (np.asarray(sb.data['obs_weight']) <= 0)[m]
yr = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])][m]
cell = np.asarray(sb.data['obs_cell_indices'])[m] if 'obs_cell_indices' in sb.data else None
out = {}
kinds = dict(train=~held, held_2021=held & (yr >= 2021), held_blocks=held & (yr < 2021))
for kn, ks in kinds.items():
    for y0, y1 in ((1966, 1993), (1994, 2009), (2010, 2020), (2021, 2025)):
        s = ks & (yr >= y0) & (yr <= y1)
        if s.sum() < 50:
            continue
        hi = s & (mu >= 10)
        out[f'{kn} {y0}-{y1}'] = dict(n=int(s.sum()), obs_over_exp=float(y[s].sum() / mu[s].sum()),
                                      pearson=float(((y[s] - mu[s]) ** 2 / var[s]).mean()),
                                      n_mu10=int(hi.sum()), obs_over_exp_mu10=(float(y[hi].sum() / mu[hi].sum()) if hi.sum() else None),
                                      pearson_mu10=(float(((y[hi] - mu[hi]) ** 2 / var[hi]).mean()) if hi.sum() else None))
print(json.dumps(out, indent=1))
