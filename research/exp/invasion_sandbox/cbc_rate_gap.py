"""How hard would the CBC RATE term push a fit? (user, 2026-11-06)

At a fit's state, the CBC rate NLL (circle effects N(0,1) Laplace-integrated; s, p, log phi optimised) is compared with
the same term when the model's temporal shape is freed: log A_ct + delta_year (free year offsets, East circles share
them). The gap = the most the term could gain by reshaping the eastern trajectory = the pressure it puts on the process.
Also split: circles < 150 km from the release vs beyond (Weiss 1997: winter counts outrun breeding counts at new fronts).
Usage (takeover runner, fit overlay): cbc_rate_gap.py --params NPZ --cbc NPZ
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--cbc', type=Path, required=True)
    a = p.parse_args()
    import jax, jax.numpy as jnp
    from jax.scipy.special import gammaln
    from scipy.optimize import minimize
    from numpyro import handlers
    from harness import Sandbox, ap
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(data=sb.data, prior_scale=1.)
    N = (np.asarray(tr['Na_grid']['value']) + np.asarray(tr['Nj_grid']['value'])) * float(sb.data['pop_scalar'])
    c = np.load(a.cbc); rows, cols = np.asarray(sb.data['land_rows']), np.asarray(sb.data['land_cols'])
    Nl = N[:, rows, cols]
    A = (Nl[c['t'][:, None], c['src'][c['circle']]] * c['w'][c['circle']]).sum(1)
    yrs = np.asarray(sb.years).astype(int)[c['t']]; uy, yi = np.unique(yrs, return_inverse=True)
    y = jnp.asarray(c['count'], float); le = jnp.asarray(c['log_effort']); ci = jnp.asarray(c['circle']); nc = int(c['circle'].max()) + 1
    logA = jnp.asarray(np.log(np.maximum(A, 1e-12))); yij = jnp.asarray(yi)

    def nll(theta, free, mask):
        s, pw, lphi = theta[0], theta[1], theta[2]; ph = jnp.exp(lphi)
        base = s + pw * le + logA + (theta[3:][yij] if free else 0.)

        def ll(u):
            mu = jnp.exp(base + u)
            return mask * (gammaln(y + ph) - gammaln(ph) - gammaln(y + 1.) + ph * (lphi - jnp.log(ph + mu)) + y * (jnp.log(mu) - jnp.log(ph + mu)))
        f = lambda u: jnp.sum(ll(u)); d1 = jax.grad(f); d2 = lambda u: jax.jvp(d1, (u,), (jnp.ones_like(u),))[1]
        seg = lambda v: jax.ops.segment_sum(v, ci, num_segments=nc)

        def step(cc, _):
            u = cc[ci]; g = seg(d1(u)) - cc; H = jnp.maximum(-seg(d2(u)), 0.) + 1.
            return cc + jnp.clip(g / H, -1., 1.), None
        cc, _ = jax.lax.scan(step, jnp.zeros(nc), None, length=40)
        H = jnp.maximum(-seg(d2(cc[ci])), 0.)
        used = seg(mask) > 0
        return -f(cc[ci]) + .5 * jnp.sum(jnp.where(used, cc ** 2 + jnp.log1p(H), 0.))

    out = dict(params=str(a.params), cbc=str(a.cbc), rows=int(y.size), circles=nc)
    near = np.asarray(c['d_release_km'])[np.asarray(c['circle'])] < 150
    for lab, msk in (('all', np.ones(y.size)), ('near150', near.astype(float)), ('far', (~near).astype(float))):
        mk = jnp.asarray(msk)
        r = {}
        for free in (False, True):
            k = 3 + (uy.size if free else 0)
            f = jax.jit(lambda th: nll(th, free, mk)); g = jax.jit(jax.grad(lambda th: nll(th, free, mk)))
            th0 = np.zeros(k); th0[0] = np.log(max(float(np.sum(np.asarray(y) * msk)), 1.) / max(float(np.sum(A * msk)), 1e-12)); th0[1] = .6
            res = minimize(lambda th: (float(f(jnp.asarray(th))), np.asarray(g(jnp.asarray(th)), float)), th0, jac=True, method='L-BFGS-B',
                           options=dict(maxiter=2000))
            r['free_trend' if free else 'model_trend'] = dict(nll=float(res.fun), p=float(res.x[1]), phi=float(np.exp(res.x[2])),
                                                              **({'delta_by_year': {int(u): round(float(v - res.x[3:].mean()), 3) for u, v in zip(uy, res.x[3:])}} if free else {}))
        r['gap'] = r['model_trend']['nll'] - r['free_trend']['nll']; r['n_rows'] = int(msk.sum())
        out[lab] = r
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
