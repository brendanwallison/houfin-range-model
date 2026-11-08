"""Distil a mechanistic fit's DYNAMICS into population_model.growth_kernel (a start for the real fit; run under the FLAG-ON
overlay). Only the gk_* sites move; every other site stays at INIT. No data likelihood is used.

  python growth_kernel_distill.py INIT.npz REF_TRACE.npz OUT.npz [--maxiter 60 --minutes 40 --n0 1e-3 --interaction]

Objective: sum over years and land cells of (log(N_gk + n0) - log(N_ref + n0))^2, N = adults + juveniles at the end of
each year (REF_TRACE from the mechanistic fit: arrays Na, Nj (T, Ny, Nx)), plus the gk sites' own priors (unit normals on
b0 / bi / bk, N(0, c_scale^2) on c / d). Why: least squares on log RATES at the realised states reproduces the rates
(R^2 > 0.99) but not the equilibria -- N* sits where lambda crosses 1, so a small rate error there moves it a lot, and
the replay then drifts several-fold from the mechanistic trajectory. Matching the trajectory directly targets that.
--interaction also frees the product-interaction block (gk_*_bi, gk_*_wi_raw); default: main effects (+ relative axis)
only, so the result keeps interaction 0 (a "neutral but dynamically matched" start).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from harness import Sandbox, ap  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('init'); p.add_argument('ref'); p.add_argument('out')
    p.add_argument('--maxiter', type=int, default=60)
    p.add_argument('--minutes', type=float, default=40.)
    p.add_argument('--n0', type=float, default=1e-3)
    p.add_argument('--interaction', action='store_true')
    a = p.parse_args()
    assert ap.GROWTH_KERNEL, 'run with a growth_kernel.enabled overlay'
    t0 = time.time()
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.init) as z:
        saved = {k: np.asarray(z[k]) for k in z.files}
    lat.update({k: jnp.asarray(v) for k, v in saved.items() if not k.startswith('laplace_')})
    ref = np.load(a.ref)
    rows, cols = jnp.asarray(sb.rows), jnp.asarray(sb.cols)
    target = jnp.log((jnp.asarray(ref['Na']) + jnp.asarray(ref['Nj']))[:, rows, cols] + a.n0)
    names = sorted(k for k in saved if k.startswith('gk_'))
    if not a.interaction:
        names = [k for k in names if not (k.endswith('_bi') or k.endswith('_wi_raw'))]
    fixed = {k: v for k, v in lat.items() if k not in names}
    c_scale = float(ap._GK_SPEC['c_scale'])
    shapes = {k: np.shape(saved[k]) for k in names}
    sizes = [int(np.prod(shapes[k])) for k in names]
    bnd = np.cumsum([0] + sizes)

    def unpack(x):
        return {k: jnp.reshape(x[bnd[i]:bnd[i + 1]], shapes[k]) for i, k in enumerate(names)}

    def objective(x):
        gk = unpack(x)
        tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data={**fixed, **gk}), 1)).get_trace(
            data=sb.data, prior_scale=1.)
        N = (tr['Na_grid']['value'] + tr['Nj_grid']['value'])[:, rows, cols]
        d = jnp.log(jnp.maximum(N, 0.) + a.n0) - target
        pri = 0.
        for k, v in gk.items():
            if k.endswith(('_b0', '_bi', '_bk')):
                pri = pri + .5 * jnp.sum(v ** 2)
            elif k.endswith(('_c', '_d')):
                pri = pri + .5 * jnp.sum((v / c_scale) ** 2)
        return jnp.sum(d ** 2) + pri, jnp.sqrt(jnp.mean(d ** 2))
    vg = jax.jit(jax.value_and_grad(lambda x: objective(x)[0]))
    rms = jax.jit(lambda x: objective(x)[1])
    x0 = np.concatenate([np.ravel(saved[k]).astype(np.float64) for k in names])
    f0, _ = vg(jnp.asarray(x0))
    print(json.dumps(dict(stage='start', objective=float(f0), rms_log=float(rms(jnp.asarray(x0))), n=int(x0.size),
                          sites=names, sec=round(time.time() - t0))), flush=True)
    best = dict(f=float(f0), x=x0)
    it = {'n': 0}

    def fun(x):
        v, g = vg(jnp.asarray(x)); v = float(v); g = np.asarray(g, np.float64)
        if not (np.isfinite(v) and np.isfinite(g).all()):
            return 1e30, np.zeros_like(x)
        if v < best['f']:
            best.update(f=v, x=np.array(x))
        return v, g

    class Stop(Exception):
        pass

    def cb(xk, *args):
        it['n'] += 1
        print(json.dumps(dict(iteration=it['n'], objective=best['f'], rms_log=float(rms(jnp.asarray(best['x']))),
                              sec=round(time.time() - t0))), flush=True)
        if (time.time() - t0) / 60. > a.minutes:
            raise Stop()
    try:
        minimize(fun, x0, jac=True, method='L-BFGS-B', callback=cb, options=dict(maxiter=a.maxiter, maxcor=50))
    except Stop:
        print(json.dumps(dict(stage='time_budget_reached')), flush=True)
    out = dict(saved)
    out.update({k: np.asarray(v) for k, v in unpack(jnp.asarray(best['x'])).items()})
    np.savez_compressed(a.out, **out)
    print(json.dumps(dict(stage='done', objective=best['f'], rms_log=float(rms(jnp.asarray(best['x']))), out=a.out,
                          sec=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
