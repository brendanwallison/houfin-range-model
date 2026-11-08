"""ROUTEMARG check: does laplace_fit's row likelihood equal the model's obs term, and are a fit's route effects at their
conditional mode? Evaluates, at a params file's theta: the model's obs NLL with (a) the file's eps, (b) eps = 0, (c) eps = the
Newton conditional mode e*, (d) the penalised value l + sum(eps^2)/2 for (a) and (c), with laplace_fit's own row_ll beside the
model's for (a)-(c). Run through the takeover runner with the arm's overlay:  laplace_check.py --params NPZ
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--newton', type=int, default=60)
    a = p.parse_args()
    sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
    import jax, jax.numpy as jnp
    from jax.scipy.special import gammaln
    from numpyro import handlers
    from harness import Sandbox, ap
    from src.config_utils import load_age_model_config
    sig = float(load_age_model_config()['population_model']['observation']['observer_effect']['sigma_u'])
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if k != 'laplace_route_sigma'})
    eps_file = np.asarray(lat['observer_route_eps'])

    def model_obs(eps):
        L = {**lat, 'observer_route_eps': jnp.asarray(eps)}
        tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=L), 1)).get_trace(data=sb.data, prior_scale=1.)
        s = tr['obs']; v = s['fn'].log_prob(s['value']) * s['scale']
        return -float(jnp.sum(v)), tr
    o_file, _ = model_obs(eps_file)
    o_zero, tr0 = model_obs(np.zeros_like(eps_file))
    y_np = np.asarray(sb.data['observed_results'], float); w_np = np.asarray(sb.data['obs_weight'], float)
    train = np.isfinite(w_np) & (w_np > 0) & np.isfinite(y_np)
    w = jnp.asarray(np.where(train, w_np, 0.)); y = jnp.asarray(np.where(train, y_np, 0.)); trn = jnp.asarray(train)
    idx = np.asarray(sb.data['obs_obsroute_idx']); on = jnp.asarray(idx >= 0); idc = jnp.asarray(np.maximum(idx, 0)); n = eps_file.size
    kap = float(tr0['obs_conc_kappa']['value']) if 'obs_conc_kappa' in tr0 else 0.
    mu0 = jnp.where(trn, tr0['expected_obs']['value'], 1.)
    phib = jnp.where(trn, tr0['obs_concentration']['value'], 1.) / jnp.power(jnp.maximum(mu0, 1e-3), kap)

    def row_ll(u):
        mu = mu0 * jnp.exp(u); ph = phib * jnp.power(jnp.maximum(mu, 1e-3), kap)
        return gammaln(y + ph) - gammaln(ph) - gammaln(y + 1.) + ph * (jnp.log(ph) - jnp.log(ph + mu)) + y * (jnp.log(jnp.maximum(mu, 1e-300)) - jnp.log(ph + mu))
    f = jax.jit(lambda u: jnp.sum(w * row_ll(u)))
    U = lambda e: jnp.where(on, sig * jnp.asarray(e)[idc], 0.)
    d1 = jax.grad(f); d2 = lambda u: jax.jvp(d1, (u,), (jnp.ones_like(u),))[1]
    seg = lambda v: jax.ops.segment_sum(jnp.where(on, v, 0.), idc, num_segments=n)

    @jax.jit
    def solve(e):
        def step(e, _):
            u = U(e); g = sig * seg(d1(u)) - e; H = sig ** 2 * jnp.maximum(-seg(d2(u)), 0.) + 1.
            return e + jnp.clip(g / H, -1., 1.), jnp.max(jnp.abs(g))
        return jax.lax.scan(step, e, None, length=a.newton)
    e0, gtr = solve(jnp.zeros(n)); ef, gtr_f = solve(jnp.asarray(eps_file))
    o_star, _ = model_obs(np.asarray(e0))
    out = dict(sigma=sig, kappa=kap,
               model_obs=dict(file=o_file, zero=o_zero, estar=o_star),
               own_ll=dict(file=-float(f(U(eps_file))), zero=-float(f(U(np.zeros(n)))), estar=-float(f(U(e0))), estar_from_file=-float(f(U(ef)))),
               penalised_model=dict(file=o_file + float((eps_file ** 2).sum() / 2), estar=o_star + float((np.asarray(e0) ** 2).sum() / 2)),
               newton_grad_max_trace_from0=[float(x) for x in np.asarray(gtr)[[0, 4, 9, 19, 39, -1]]],
               newton_grad_max_trace_fromfile=[float(x) for x in np.asarray(gtr_f)[[0, 4, 9, 19, 39, -1]]],
               eps_mean=dict(file=float(eps_file.mean()), estar=float(np.asarray(e0).mean())),
               eps_sd=dict(file=float(eps_file.std()), estar=float(np.asarray(e0).std())),
               max_abs_diff_estar_two_starts=float(np.abs(np.asarray(e0) - np.asarray(ef)).max()))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
