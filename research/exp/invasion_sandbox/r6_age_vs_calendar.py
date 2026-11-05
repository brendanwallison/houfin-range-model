"""R6: does the fixed-habitat birth-rate rise follow LOCAL colonisation age or CALENDAR year?

Setup as grad_inverse.py (step 3, habitat frozen at the 1966-2025 mean, release only, loss = East BBS log-lik under
the fit's own observation model, L-BFGS with exact JAX gradients). The eastern Fmax multiplier is piecewise linear in
  cal: calendar year (knots CAL_KNOTS), or
  age: years since the cell was colonised (knots AGE_KNOTS); colonised = first year the cell's density reaches 1% of
       the reference arm's modern occupied median (unit-free); uncolonised -> age 0.
Colonisation years depend on the state, so 'age' is solved by fixed-point iteration: fit given the years, re-simulate,
update, refit (warm start) until under 2% of eastern cells change or --rounds is reached.
Usage (inside sbx_env.sh): python r6_age_vs_calendar.py --mode cal|age --out DIR [--maxiter 200] [--rounds 5]
"""
import argparse, json, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.optimize import minimize
from harness import Sandbox, capture, edit_habitat, edit_args, region_masks, c_of, to_land, ap

CAL_KNOTS = [1940, 1955, 1965, 1975, 1985, 1995, 2010, 2025]
AGE_KNOTS = [0, 2, 5, 10, 15, 25, 40, 85]


def interp_basis(v, knots):
    """Piecewise-linear 'hat' weights of values v (any shape) on knots -> v.shape + (len(knots),); clamped ends."""
    return np.stack([np.interp(v, knots, np.eye(len(knots))[i]) for i in range(len(knots))], -1)


class Problem:
    def __init__(self, sb):
        self.sb = sb
        self.d = edit_habitat(sb.data, sb.years, 0., [1966, 2025])
        x0, self.fwd = capture(self.d, sb.lat)
        self.x0 = edit_args(x0, dict(region='east_only'), sb.lon, sb.years, sb.rows, sb.cols)
        self.east = region_masks(sb.lon)['east']
        self.shape = np.asarray(self.x0['Fmax_flat']).shape
        self.east_rows = jnp.asarray(np.flatnonzero(sb.obs_ok)[sb.region['east'][sb.o_cell]])
        self.y = jnp.asarray(sb.data['observed_results'])
        self.lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
        self.T, self.N = len(sb.years), len(sb.lon)

    def args(self, theta, B):
        """B: (T, N, knots) basis; West cells get multiplier 1 (they are empty in release-only arms anyway)."""
        m = B @ jnp.exp(theta[:-1])
        mult = jnp.where(jnp.asarray(self.east)[None, :], m, 1.).reshape(self.shape)
        x = dict(self.x0)
        F = jnp.asarray(self.x0['Fmax_flat']) * mult * jnp.exp(theta[-1])
        Sa = jnp.minimum(jnp.asarray(x['Sa_flat']), .995); Sj = jnp.minimum(jnp.asarray(x['Sj_flat']), Sa)
        x.update(Fmax_flat=F, Sa_flat=Sa, Sj_flat=Sj, c_flat=c_of(F, Sa, Sj, x))
        return x

    def make_vg(self, B):
        B = jnp.asarray(B)
        def neg_ll(theta):
            out = self.fwd(**self.args(theta, B))
            original = ap.forward_sim_age_structured
            ap.forward_sim_age_structured = lambda *aa, **kw: out
            try:
                tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=self.lat),
                                                  rng_seed=104)).get_trace(data=self.d, prior_scale=1.)
            finally:
                ap.forward_sim_age_structured = original
            return -jnp.sum(tr['obs']['fn'].log_prob(self.y)[self.east_rows])
        return jax.jit(jax.value_and_grad(neg_ll))

    def multiplier(self, theta, B):
        return np.asarray(B @ np.exp(theta[:-1])) * np.exp(theta[-1])

    def colonisation_age(self, theta, B, thr):
        out = self.fwd(**self.args(jnp.asarray(theta), jnp.asarray(B)))
        tot = to_land(out[0], self.sb.rows, self.sb.cols, self.T, self.N)
        occ = tot >= thr
        col = np.where(occ.any(0), self.sb.years[np.argmax(occ, 0)], np.nan)
        age = np.where(np.isfinite(col)[None], np.maximum(self.sb.years[:, None] - col[None], 0.), 0.)
        return age, col


def fit(vg, theta0, maxiter, log, tag, t0):
    n = [0]
    def f(theta):
        v, g = vg(jnp.asarray(theta)); v, g = float(v), np.asarray(g, float); n[0] += 1
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            v, g = 1e12, np.zeros_like(theta)
        log.write(json.dumps(dict(tag=tag, eval=n[0], theta=list(map(float, theta)), nll=v,
                                  grad_norm=float(np.linalg.norm(g)), sec=round(time.time() - t0, 1))) + '\n')
        log.flush()
        return v, g
    res = minimize(f, theta0, jac=True, method='L-BFGS-B', bounds=[(-4., 3.)] * len(theta0),
                   options=dict(maxiter=maxiter))
    return res, n[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['cal', 'age'], required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--maxiter', type=int, default=200); p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--fscale0', type=float, default=.4)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    ref_nll = -sum(v['loglik'] for k, v in ref['abundance'].items() if k.startswith('east'))
    thr = .01 * sb.base_med
    P = Problem(sb); t0 = time.time()
    log = (a.out / f'r6_{a.mode}.jsonl').open('w')
    log.write(json.dumps(dict(arm='fitted', nll=ref_nll, threshold_model_units=thr)) + '\n')
    knots = CAL_KNOTS if a.mode == 'cal' else AGE_KNOTS
    theta = np.r_[np.zeros(len(knots)), np.log(a.fscale0)]
    rounds = []
    if a.mode == 'cal':
        B = np.broadcast_to(interp_basis(sb.years.astype(float), knots)[:, None, :], (P.T, P.N, len(knots)))
        res, ne = fit(P.make_vg(B), theta, a.maxiter, log, 'cal', t0); theta = res.x
        rounds.append(dict(round=0, nll=float(res.fun), n_eval=ne, converged=bool(res.success)))
    else:
        # round-0 colonisation years: from the constant-multiplier start (theta0)
        B = interp_basis(np.zeros((P.T, P.N)), knots)
        age, col = P.colonisation_age(theta, B, thr)
        for r in range(a.rounds):
            B = interp_basis(age, knots)
            res, ne = fit(P.make_vg(B), theta, a.maxiter, log, f'age_r{r}', t0); theta = res.x
            age_new, col_new = P.colonisation_age(theta, B, thr)
            e = P.east
            changed = float(np.mean((np.nan_to_num(col_new[e], nan=-1) != np.nan_to_num(col[e], nan=-1))))
            rounds.append(dict(round=r, nll=float(res.fun), n_eval=ne, converged=bool(res.success),
                               changed_share=changed, colonised_east=int(np.isfinite(col_new[e]).sum())))
            print(json.dumps(rounds[-1]), flush=True)
            age, col = age_new, col_new
            if changed < .02:
                break
        B = interp_basis(age, knots)   # final basis consistent with the final state (one more fixed-point step)
    vg = P.make_vg(B); final_nll = float(vg(jnp.asarray(theta))[0])
    mult = P.multiplier(theta, B)
    best = sb.run(dict(name=f'r6_{a.mode}', region='east_only', habitat_alpha=0, rate_array=dict(Fmax=mult)))
    readout_nll = -sum(v['loglik'] for k, v in best['abundance'].items() if k.startswith('east'))
    out = dict(mode=a.mode, knots=knots, fitted_reference_nll=ref_nll, final_nll=final_nll, readout_nll=readout_nll,
               rounds=rounds,
               knot_multiplier=list(np.exp(theta[:-1]) * np.exp(theta[-1])), fmax_scale=float(np.exp(theta[-1])),
               minutes=round((time.time() - t0) / 60, 1), sha=best['final_state_sha256'][:12],
               offsets=best['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               east_obs_over_expected={k: v['obs_over_expected'] for k, v in best['abundance'].items()
                                       if k.startswith('east')},
               east_loglik={k: v['loglik'] for k, v in best['abundance'].items() if k.startswith('east')},
               east_zero_share={k: [v['zero_share_obs'], v['zero_share_pred']] for k, v in best['abundance'].items()
                                if k.startswith('east')})
    np.savez_compressed(a.out / f'r6_{a.mode}_multiplier.npz', mult=mult.astype(np.float32), years=sb.years)
    (a.out / f'r6_{a.mode}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
