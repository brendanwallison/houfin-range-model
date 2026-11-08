"""population_model.growth_kernel integration checks on the real grid54 inputs (run through local_takeover_run.sh).

  off PARAMS REF.npz      flag-OFF overlay: trace expected_obs / Na at PARAMS and compare to REF (written by the pre-change
                          code) to 1e-10 -- the flag must leave the model byte-identical.
  on PARAMS               flag-ON overlay: trace at PARAMS (an init from growth_kernel_init.py), check finiteness, and take
                          jax.grad of log_density with respect to the gk_* sites (finite, nonzero).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402
from harness import Sandbox, ap  # noqa: E402

_SB = {}


def _sandbox():
    if 'sb' not in _SB:
        _SB['sb'] = Sandbox(os.environ['SBX_REFINED'])
    return _SB['sb']


def _latents(params):
    sb = _sandbox()
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_')})
    return lat


def trace_expected_obs(params):
    sb = _sandbox()
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=_latents(params)), 1)).get_trace(
        data=sb.data, prior_scale=1.)
    return dict(expected_obs=np.asarray(tr['expected_obs']['value']), Na=np.asarray(tr['Na_grid']['value']), trace=tr)


def check_off(params, ref):
    got = trace_expected_obs(params); r = np.load(ref)
    for k in ('expected_obs', 'Na'):
        a, b = got[k], r[k]
        rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
        print(f'{k}: shape {a.shape} max rel diff {rel:.3e} identical={np.array_equal(a, b)}', flush=True)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=0)
    print('FLAG-OFF OK', flush=True)


def check_on(params):
    assert ap.GROWTH_KERNEL, 'run with a growth_kernel.enabled overlay'
    sb = _sandbox(); lat = _latents(params)
    t0 = time.time()
    tr = trace_expected_obs(params)['trace']
    print(f'trace {time.time() - t0:.1f}s', flush=True)
    for k in ('expected_obs', 'Na_grid', 'growth_s_real', 'growth_rho_real', 'growth_lambda0', 'nest_F_real'):
        v = np.asarray(tr[k]['value'])
        print(f'{k}: shape {v.shape} finite {np.isfinite(v).all()} q01/50/99 {np.quantile(v, [.01, .5, .99])}', flush=True)
        assert np.isfinite(v).all(), k
    names = sorted(k for k, s in tr.items() if s['type'] == 'sample' and k.startswith('gk_'))
    print('gk sites:', {k: tuple(np.shape(tr[k]['value'])) for k in names}, flush=True)
    fixed = {k: v for k, v in lat.items() if k not in names}
    model = handlers.substitute(ap.build_model_2d, data=fixed)

    def lp(gk):
        return log_density(model, (), dict(data=sb.data, prior_scale=1.), gk)[0]
    gk0 = {k: jnp.asarray(tr[k]['value']) for k in names}
    t0 = time.time()
    v, g = jax.value_and_grad(lp)(gk0)
    print(f'log_density {float(v):.3f}; grad {time.time() - t0:.1f}s', flush=True)
    for k in names:
        gk_ = np.asarray(g[k])
        print(f'  d/d {k}: finite {np.isfinite(gk_).all()} |g|max {np.abs(gk_).max():.3e}', flush=True)
        assert np.isfinite(gk_).all(), k
    assert np.abs(np.asarray(g['gk_r_bi'])).max() > 0 and np.abs(np.asarray(g['gk_s_bi'])).max() > 0
    print('FLAG-ON OK', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'off':
        check_off(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'on':
        check_on(sys.argv[2])
    else:
        raise SystemExit(__doc__)
