"""Where is the fit stiff? Hessian curvature at a fitted endpoint, by parameter site, data vs prior.

User (2026-11-05): Codex measured sharp curvature along the gradient at the compact-onset endpoint but not WHERE.
This reproduces local_joint_fit.py's posterior objective exactly (same MapCoordinates, same excluded inactive sites,
negative constrained-site log posterior, no Jacobian) and uses Hessian-vector products (jvp of the gradient):
  - Lanczos (full reorthogonalisation) for the top curvature eigenpairs of the TOTAL objective;
  - for each eigenvector and for the normalised gradient: Rayleigh quotient split into DATA (BBS + NestWatch NLL)
    and PRIOR (everything else) curvature, and the share of the vector's squared norm on each site;
  - per-site curvature: Rayleigh quotient of the total Hessian restricted to each site's block along that block's
    gradient (where that is non-zero) -- the stiffness each site would show on its own.
Coordinates are the optimizer's (Normal sites in prior-SD units, constrained sites through their bijections), so
eigenvalues are directly the curvature L-BFGS sees.
Run through the takeover runner (read-only on its worktree):
  TAKEOVER_CODE_ROOT=<worktree> AGE_MODEL_CONFIG=<overlay> bash <root>/research/bin/local_takeover_run.sh \
     /mnt/c/.../stiffness_hvp.py --params NPZ --out JSON [--k 12]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--k', type=int, default=12, help='Lanczos steps (top eigenpairs reported)')
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--start', choices=['random', 'gradient'], default='random', help='Lanczos start vector')
    p.add_argument('--skip_per_site', action='store_true')
    p.add_argument('--save_vectors', type=Path, default=None, help='npz path for the top Ritz vectors')
    a = p.parse_args()
    root = Path(os.environ['TAKEOVER_CODE_ROOT'])
    sys.path.insert(0, str(root / 'research' / 'exp'))
    import jax, jax.numpy as jnp
    from numpyro import handlers
    from numpyro.infer.util import log_density
    from harness import Sandbox, ap
    from src.model import hmc_common as hc
    from local_map_coordinates import MapCoordinates

    t0 = time.time()
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as saved:
        lat.update({k: jnp.asarray(saved[k]) for k in saved.files})
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), a.seed)).get_trace(data=sb.data, prior_scale=1.)
    excluded = ['dispersal_logit_intercept', 'dispersal_logit_slope'] if ap.VEIT_DEPARTURE else []
    coords = MapCoordinates(tr, exclude=excluded)
    del tr
    arrays, static = hc.split_data(sb.data)
    array_model = hc.array_model(ap.build_model_2d, static)
    blocks, off = [], 0
    for name in coords.names:
        n = int(np.asarray(coords.initial[name]).size); blocks.append((name, off, off + n)); off += n
    x0 = jnp.asarray(coords.x0, dtype=jnp.float64)

    def parts(x):
        lp, trace = log_density(array_model, (), {'arrays': arrays}, coords.constrain(x))
        data = 0.
        for name in ('obs', 'nest_loglik'):
            s = trace[name]; v = s['fn'].log_prob(s['value'])
            data = data + jnp.sum(v * (s['scale'] if s.get('scale') is not None else 1.))
        return -lp, -data            # total objective, data NLL

    total = lambda x: parts(x)[0]
    data_f = lambda x: parts(x)[1]
    g_total = jax.jit(jax.grad(total)); g_data = jax.jit(jax.grad(data_f))
    hvp_total = jax.jit(lambda x, v: jax.jvp(g_total, (x,), (v,))[1])
    hvp_data = jax.jit(lambda x, v: jax.jvp(g_data, (x,), (v,))[1])
    f0, d0 = [float(v) for v in jax.jit(parts)(x0)]
    g = np.asarray(g_total(x0)); gn = float(np.linalg.norm(g))
    print(json.dumps(dict(stage='loaded', objective=f0, data_nll=d0, grad_norm=gn, n=int(x0.size), sec=round(time.time() - t0))), flush=True)

    def split(v):   # Rayleigh quotients (v unit): total, data, prior; site shares of |v|^2
        v = jnp.asarray(v / np.linalg.norm(v))
        ht = float(jnp.dot(v, hvp_total(x0, v))); hd = float(jnp.dot(v, hvp_data(x0, v)))
        vv = np.asarray(v) ** 2
        share = sorted(((nm, float(vv[s:e].sum())) for nm, s, e in blocks), key=lambda t: -t[1])
        return dict(total=ht, data=hd, prior=ht - hd, top_sites=[(nm, round(sh, 4)) for nm, sh in share[:6] if sh > .005])

    out = dict(params=str(a.params), config=os.environ.get('AGE_MODEL_CONFIG'), objective=f0, data_nll=d0, grad_norm=gn,
               n_coordinates=int(x0.size), gradient_site_share=None, along_gradient=None, lanczos=None, per_site=None)
    gg = g ** 2
    out['gradient_site_share'] = [(nm, round(float(gg[s:e].sum() / gg.sum()), 4)) for nm, s, e in
                                  sorted(blocks, key=lambda b: -gg[b[1]:b[2]].sum())[:8]]
    out['along_gradient'] = split(g)
    print(json.dumps(dict(stage='gradient', **out['along_gradient'], sec=round(time.time() - t0))), flush=True)

    # Lanczos with full reorthogonalisation on the total Hessian
    rng = np.random.default_rng(a.seed)
    q = g.copy() if a.start == 'gradient' else rng.normal(size=x0.size)   # a random start mostly samples flat directions
    q /= np.linalg.norm(q)
    Q, alpha, beta = [q], [], []
    for j in range(a.k):
        w = np.asarray(hvp_total(x0, jnp.asarray(Q[-1])))
        al = float(w @ Q[-1]); alpha.append(al)
        for qq in Q:
            w = w - (w @ qq) * qq
        b = float(np.linalg.norm(w)); beta.append(b)
        print(json.dumps(dict(stage='lanczos', step=j, alpha=al, beta=b, sec=round(time.time() - t0))), flush=True)
        if b < 1e-10:
            break
        Q.append(w / b)
    m = len(alpha); T = np.diag(alpha) + np.diag(beta[:m - 1], 1) + np.diag(beta[:m - 1], -1)
    ev, U = np.linalg.eigh(T); order = np.argsort(-ev)
    Qm = np.stack(Q[:m], 1)
    eig = []
    for i in list(order[:min(6, m)]) + [order[-1]]:   # top six and the most NEGATIVE / smallest Ritz pair
        v = Qm @ U[:, i]
        eig.append(dict(ritz=float(ev[i]), **split(v)))
        print(json.dumps(dict(stage='eigvec', ritz=float(ev[i]), sec=round(time.time() - t0))), flush=True)
    out['lanczos'] = dict(steps=m, ritz_values=sorted([float(x) for x in ev], reverse=True), top=eig)
    if a.save_vectors:   # top-k Ritz vectors (optimizer coordinates) + site blocks, for eigenvector-overlap checks
        V = np.stack([Qm @ U[:, i] for i in order[:min(8, m)]], 1)
        np.savez_compressed(a.save_vectors, vectors=V, ritz=np.asarray([ev[i] for i in order[:min(8, m)]]),
                            names=np.asarray([b[0] for b in blocks]), starts=np.asarray([b[1] for b in blocks]), ends=np.asarray([b[2] for b in blocks]))

    # per-site self-curvature along each site's own gradient block
    per = []
    for nm, s, e in ([] if a.skip_per_site else blocks):
        v = np.zeros(x0.size); v[s:e] = g[s:e]
        if np.linalg.norm(v) == 0:
            continue
        r = split(v); per.append(dict(site=nm, size=e - s, grad_share=float(gg[s:e].sum() / gg.sum()), total=r['total'], data=r['data'], prior=r['prior']))
    out['per_site'] = sorted(per, key=lambda d: -d['total'])
    out['seconds'] = round(time.time() - t0)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(stage='done', top_ritz=out['lanczos']['ritz_values'][:3], sec=out['seconds'])), flush=True)


if __name__ == '__main__':
    main()
