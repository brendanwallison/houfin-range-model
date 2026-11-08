"""Joint MAP by trust-region Newton-Krylov with exact Hessian-vector products (STIFF follow-up).

Same objective and coordinates as Codex's local_joint_fit.py --objective posterior (negative constrained-site log
posterior, no Jacobian; MapCoordinates with the inactive Veit logistic departure sites excluded) -- only the optimizer
differs: scipy trust-krylov with hessp = jvp of the gradient. STIFF showed the problem is ill-conditioned (top curvature
~1e6 in the shared habitat-factor direction vs flat directions < 10), which L-BFGS cannot cross in a few thousand steps.
Writes trace.jsonl, best_params.npz, result.json (objective, data NLL split, gradient norm / max, iterations).
Run through the takeover runner (read-only on the worktree):
  TAKEOVER_CODE_ROOT=<worktree> AGE_MODEL_CONFIG=<overlay> bash <root>/research/bin/local_takeover_run.sh \
     /mnt/c/.../newton_fit.py --init NPZ --out DIR [--maxiter 200]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--maxiter', type=int, default=200)
    p.add_argument('--gtol', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--method', choices=['trust-krylov', 'lbfgs'], default='trust-krylov')
    p.add_argument('--maxcor', type=int, default=200)
    p.add_argument('--jitter', type=float, default=0.)
    p.add_argument('--fix', default='', help='comma list of sites held at their --init values (dropped from the coordinates)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    root = Path(os.environ['TAKEOVER_CODE_ROOT'])
    sys.path.insert(0, str(root / 'research' / 'exp'))
    import jax, jax.numpy as jnp
    from numpyro import handlers
    from numpyro.infer.util import log_density
    from scipy.optimize import minimize
    from harness import Sandbox, ap
    from src.model import hmc_common as hc
    from local_map_coordinates import MapCoordinates

    t0 = time.time()
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.init) as saved:
        lat.update({k: jnp.asarray(saved[k]) for k in saved.files})
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), a.seed)).get_trace(data=sb.data, prior_scale=1.)
    excluded = ['dispersal_logit_intercept', 'dispersal_logit_slope'] if ap.VEIT_DEPARTURE else []
    fixed = [s for s in a.fix.split(',') if s]
    coords = MapCoordinates(tr, exclude=excluded + fixed)
    del tr
    arrays, static = hc.split_data(sb.data)
    array_model = hc.array_model(ap.build_model_2d, static)

    def parts(x):
        lp, trace = log_density(array_model, (), {'arrays': arrays}, coords.constrain(x))
        data = []
        for name in ('obs', 'nest_loglik'):
            s = trace[name]; v = s['fn'].log_prob(s['value'])
            data.append(jnp.sum(v * (s['scale'] if s.get('scale') is not None else 1.)))
        return -lp, -jnp.stack(data)

    vg = jax.jit(jax.value_and_grad(lambda x: parts(x)[0]))
    dparts = jax.jit(lambda x: parts(x)[1])
    g_only = jax.jit(jax.grad(lambda x: parts(x)[0]))
    hvp = jax.jit(lambda x, v: jax.jvp(g_only, (x,), (v,))[1])
    x0 = np.asarray(coords.x0, dtype=np.float64)
    if a.jitter:   # dispersed start in the optimizer coordinates (prior-SD units for Normal sites)
        x0 = x0 + np.random.default_rng(a.seed).normal(0., a.jitter, x0.shape)
    trace = (a.out / 'trace.jsonl').open('x')
    best = {'objective': float('inf')}
    n = {'f': 0, 'h': 0, 'it': 0}

    def save(x):
        tmp = a.out / 'best_params.tmp.npz'
        np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in coords.constrain(jnp.asarray(x)).items()})
        tmp.replace(a.out / 'best_params.npz')

    def fun(x):
        v, g = vg(jnp.asarray(x)); v, g = float(v), np.asarray(g, np.float64)
        if not (np.isfinite(v) and np.isfinite(g).all()):
            return np.inf, np.zeros_like(x)
        n['f'] += 1
        if v < best['objective']:
            best.update(objective=v, x=np.array(x), gnorm=float(np.linalg.norm(g)), gmax=float(np.abs(g).max()))
        return v, g

    def hessp(x, v):
        n['h'] += 1
        return np.asarray(hvp(jnp.asarray(x), jnp.asarray(v)), np.float64)

    def callback(xk, *args):
        n['it'] += 1
        d = np.asarray(dparts(jnp.asarray(xk)))
        row = dict(iteration=n['it'], objective=best['objective'], data_nll=float(d.sum()), obs=float(d[0]), nest=float(d[1]),
                   grad_l2=best.get('gnorm'), grad_max=best.get('gmax'), f_evals=n['f'], hvps=n['h'], sec=round(time.time() - t0, 1))
        trace.write(json.dumps(row) + '\n'); trace.flush(); print(json.dumps(row), flush=True)
        save(best['x'])

    f0, _ = fun(x0)
    print(json.dumps(dict(stage='start', objective=f0, n=int(x0.size), sec=round(time.time() - t0, 1))), flush=True)
    if a.method == 'lbfgs':
        res = minimize(fun, x0, jac=True, method='L-BFGS-B', callback=callback,
                       options=dict(maxiter=a.maxiter, maxcor=a.maxcor, maxls=40, ftol=1e-12, gtol=1e-4))
    else:
        res = minimize(fun, x0, jac=True, hessp=hessp, method='trust-krylov', callback=callback,
                       options=dict(maxiter=a.maxiter, gtol=a.gtol))
    save(best['x'])
    d = np.asarray(dparts(jnp.asarray(best['x'])))
    out = dict(objective=best['objective'], data_nll=dict(obs=float(d[0]), nest_loglik=float(d[1])), gradient_l2=best['gnorm'],
               gradient_max=best['gmax'], iterations=n['it'], f_evals=n['f'], hvps=n['h'], message=str(res.message),
               success=bool(res.success), stationarity_demonstrated=bool(best['gmax'] < 1e-2), start_objective=f0,
               init=str(a.init), config=os.environ.get('AGE_MODEL_CONFIG'), method=a.method, fixed_sites=fixed,
               seconds=round(time.time() - t0))
    (a.out / 'result.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
