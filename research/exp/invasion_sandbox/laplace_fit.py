"""Joint MAP with the BBS observer x route effects MARGINALISED by a per-route Laplace approximation (ROUTEMARG).

Why: the route effects (19,127 observer_route_eps, ZeroSumNormal(1) x pinned sigma_u) are a flat ridge at the joint mode
-- starts within 8-9 objective units differ by up to ~765 marginal held-out NLL and their route SD ranges .63-1.13
(CAND multistarts, SANDBOX_INVASION_SHAPE 2026-11-05), and sigma_u cannot be estimated at a joint mode (age_priors
observer_log_offset docstring). Each route effect enters ONLY its own rows' NB means and nothing feeds back into the
dynamics, so the route block of the Hessian is DIAGONAL: the Laplace marginal (TMB / glmmTMB style) is exact in
structure and cheap. Per route r, with e ~ N(0,1), u = sigma e:
    -log p(y_r | theta, sigma) ~= -[sum_i w_i l_i(eta_i + sigma e*) - e*^2/2] + 1/2 log(1 + sigma^2 H_r),
e* the route's conditional mode (30 step-clipped Newton steps from 0, unrolled so the gradient is exact as they converge),
H_r = -sum_i w_i l_i''. l_i is the model's own NB2 row log-likelihood, including conc_density_power (phi depends on mu).
The route effects then leave the optimizer's coordinates entirely and sigma becomes ONE free coordinate (log sigma,
HalfNormal(sigma_prior_scale) prior from the overlay, no Jacobian, as the rest of the objective).
Everything else is Codex's posterior objective (newton_fit.py / local_joint_fit.py --objective posterior).
best_params.npz stores observer_route_eps = e* * sigma_hat / sigma_cfg so that the model's sigma_cfg * eps equals the fitted
conditional effect (readouts reproduce it), plus laplace_route_sigma = sigma_hat (marginal_heldout.py reads it).
Run through the takeover runner like newton_fit.py.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np


def run_hmc(a, parts, coords, x0, jax, jnp, info, sig_cfg, t0):
    """NUTS on the embedded-Laplace posterior (HMC-READY, 2026-11-05).

    Target in the optimizer's unconstrained coordinates u: log p(u) = -objective(u) + log|d constrain/du| + log sigma
    (the MAP objective omits Jacobians; sigma = exp(log sigma)). The 19,127 route effects stay integrated out by the per-route
    Laplace approximation (exact in structure; Stan's "embedded Laplace"). Dense mass matrix initialised from the
    eigen-clipped inverse Hessian at the mode (HESS: one stiff direction at 2e6, the shared habitat field + survival
    intercepts, which a diagonal mass cannot fix), then adapted in warmup. Samples saved every --hmc_segment draws.
    """
    import time as _t
    from numpyro.infer import MCMC, NUTS
    if a.route_mean or a.fix_sigma:
        raise SystemExit('--hmc supports the free-sigma, no-route-mean candidate only')

    nc = int(np.asarray(coords.x0).size)   # x = [model coords (nc), log sigma, then unconstrained extra blocks (CBC counts, BBL...)]

    def potential(x):
        xs, lsg = x[:nc], x[nc]
        return parts(x)[0] - coords.log_jacobian(xs) - lsg
    pot = jax.jit(potential)
    gfun = jax.grad(potential)
    hvp1 = jax.jit(lambda x, v: jax.jvp(gfun, (x,), (v,))[1])
    xj = jnp.asarray(x0); I = np.eye(x0.size)
    H = np.stack([np.asarray(hvp1(xj, jnp.asarray(I[i])), np.float64) for i in range(x0.size)], 1); H = .5 * (H + H.T)
    ev, V = np.linalg.eigh(H)
    evc = np.maximum(ev, 1e-2)
    Minv = (V / evc) @ V.T
    meta = dict(n=int(x0.size), potential_at_init=float(pot(xj)), eig_min=float(ev[0]), eig_max=float(ev[-1]),
                n_clipped=int((ev < 1e-2).sum()), hessian_sec=round(_t.time() - t0))
    print(json.dumps(dict(stage='hmc_mass', **meta)), flush=True)
    (a.out / 'hmc_meta.json').write_text(json.dumps(dict(meta, args=vars(a) | {'init': str(a.init), 'out': str(a.out)},
                                                         config=os.environ.get('AGE_MODEL_CONFIG')), indent=1, default=str) + '\n')
    np.savez_compressed(a.out / 'hmc_init_mass.npz', H=H, Minv=Minv, x0=x0, names=np.asarray(list(coords.names) + ['log_sigma']))
    # --hmc_mass (2026-11-06): dense_adapt degrades the Hessian mass. numpyro's dense adaptation regularises the window covariance
    # with 1e-3 * 5/(n+5) I, which dwarfs this posterior's stiff variances (sd ~7e-4) and, from <=100 warmup draws, leaves
    # kappa ~1e5: on a Gaussian surrogate with this Sigma every draw hits depth 8 and min ESS is 6/300 vs 220/300 unadapted.
    # whitened_diag: sample u with x = x0 + L u, L = chol(Minv) (the Hessian geometry), and let warmup adapt a DIAGONAL mass in u
    # -- per-direction rescaling around the Hessian, needing only ~d variances, not d^2/2 covariances. Surrogate: min ESS 461/300.
    if a.hmc_mass == 'whitened_diag':
        Lw = jnp.asarray(np.linalg.cholesky(Minv)); xc = jnp.asarray(x0)
        pot_run = jax.jit(lambda u: potential(xc + Lw @ u)); init = jnp.zeros_like(xc)
        back = lambda U: np.asarray(xc)[None] + np.asarray(U) @ np.asarray(Lw).T
        kernel = NUTS(potential_fn=pot_run, dense_mass=False, adapt_mass_matrix=True, max_tree_depth=a.hmc_depth, target_accept_prob=0.85)
    else:
        init, back = jnp.asarray(x0), (lambda U: np.asarray(U))
        kernel = NUTS(potential_fn=pot, dense_mass=True, inverse_mass_matrix=jnp.asarray(Minv),
                      adapt_mass_matrix=(a.hmc_mass == 'dense_adapt'), max_tree_depth=a.hmc_depth, target_accept_prob=0.85)
    C = int(a.hmc_chains)
    mcmc = MCMC(kernel, num_warmup=a.hmc_warmup, num_samples=a.hmc_segment, num_chains=C, chain_method='vectorized',
                progress_bar=False)
    key = jax.random.PRNGKey(a.seed)
    if C > 1:   # dispersed starts: +-0.5 posterior sd per whitened direction around the mode (Hessian scale for the dense paths)
        _z = .5 * np.random.default_rng(a.seed).standard_normal((C, x0.size)); _z[0] = 0.
        init = jnp.asarray(_z) if a.hmc_mass == 'whitened_diag' else jnp.asarray(np.asarray(x0)[None] + _z @ np.linalg.cholesky(Minv).T)
    _ss = lambda: np.round(np.ravel(np.asarray(mcmc.last_state.adapt_state.step_size)), 5).tolist()
    done, seg = 0, 0
    while done < a.hmc_samples and (_t.time() - t0) / 3600. < a.hmc_hours:
        ts = _t.time()
        if seg == 0:
            mcmc.warmup(key, init_params=init, extra_fields=('diverging', 'num_steps', 'potential_energy', 'accept_prob'),
                        collect_warmup=True)
            wx = {k: np.asarray(v).reshape(C, -1) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
            wrow = dict(stage='hmc_warmup', sec=round(_t.time() - ts), mass=a.hmc_mass, chains=C,
                        mean_steps_by_quarter=[float(q.mean()) for q in np.array_split(wx['num_steps'], 4, axis=1)],
                        divergences=int(wx['diverging'].sum()), accept_last100=float(wx['accept_prob'][:, -100:].mean()),
                        step_size=_ss())
            with open(a.out / 'hmc_trace.jsonl', 'a') as fh:
                fh.write(json.dumps(wrow) + '\n')
            print(json.dumps(wrow), flush=True)
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=('diverging', 'num_steps', 'potential_energy', 'accept_prob'))
        else:
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=('diverging', 'num_steps', 'potential_energy', 'accept_prob'))
        Xc = back(np.asarray(mcmc.get_samples(group_by_chain=True)).reshape(C, a.hmc_segment, -1))   # (chain, draw, dim)
        ex = {k: np.asarray(v).reshape(C, -1) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
        X = Xc.reshape(-1, Xc.shape[-1]); sig = np.exp(X[:, nc])
        cons = [{k: np.asarray(v) for k, v in coords.constrain(jnp.asarray(x[:nc])).items() if np.asarray(v).size <= 8} for x in X[::max(1, len(X) // 10)]]
        np.savez_compressed(a.out / f'hmc_seg{seg:03d}.npz', x=Xc, sigma=np.exp(Xc[..., nc]), n_coords=nc, **{f'extra_{k}': v for k, v in ex.items()},
                            step_size=np.asarray(mcmc.last_state.adapt_state.step_size))
        done += Xc.shape[1]
        row = dict(segment=seg, draws_per_chain=done, chains=C, sec=round(_t.time() - ts), total_h=round((_t.time() - t0) / 3600., 2),
                   divergences=int(ex['diverging'].sum()), mean_steps=float(ex['num_steps'].mean()), max_steps=int(ex['num_steps'].max()),
                   accept=float(ex['accept_prob'].mean()), step_size=_ss(),
                   sigma_mean=float(sig.mean()), potential_mean=float(ex['potential_energy'].mean()),
                   scalars_last={k: np.round(np.ravel(v), 4).tolist() for k, v in cons[-1].items() if k.startswith(('disease_rec', 'disease_delay', 'alpha_', 'concentration'))})
        with open(a.out / 'hmc_trace.jsonl', 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        seg += 1
    print(json.dumps(dict(stage='hmc_done', draws=done, hours=round((_t.time() - t0) / 3600., 2))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--maxiter', type=int, default=3000)
    p.add_argument('--drop_pseudozeros', action='store_true', help='DIAGNOSTIC (2026-11-07 audit): give the lineage 1902-39 '
                   'pseudo-zero BBS rows (no route id) zero training weight, as rangev2 intended -- they are otherwise counted ON TOP '
                   'of --range_absence (the "removed" comment in the absence block is a no-op)')
    p.add_argument('--perturb', type=float, default=0.0, help='DIAGNOSTIC: add N(0, sd) to the MODEL coordinates of x0 (unconstrained '
                   'scale; seed --seed) before fitting -- probes whether the optimum is unique (basin check)')
    p.add_argument('--laplace_warm', action='store_true', help='warm-start the inner route-effect Newton from the route effects at '
                   'the last accepted iterate (default: from 0). With conc_density_power a route conditional can be bimodal; a cold '
                   'start can switch modes under a tiny parameter change (a ~.16-nat jump that stalls L-BFGS, J110 2026-11-07). The '
                   'objective is fixed within an iteration (line search) and moves with the accepted point between iterations.')
    p.add_argument('--maxcor', type=int, default=200)
    p.add_argument('--sigma0', type=float, default=None, help='start sigma (default: init laplace_route_sigma or overlay sigma_u)')
    p.add_argument('--newton', type=int, default=30)
    p.add_argument('--fix', default='')
    p.add_argument('--fix_sigma', action='store_true', help='hold sigma at sigma0 (Laplace marginal at a pinned scale)')
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--agh', type=int, default=0, help='Laplace accuracy check at --init: per-route adaptive Gauss-Hermite with K and 2K nodes; no fit')
    p.add_argument('--hessian_only', action='store_true', help='write the full Hessian eigen-spectrum at --init (no fit): hessian.json + hessian.npz')
    p.add_argument('--hmc', action='store_true', help='NUTS from --init (a converged Laplace MAP) instead of a fit')
    p.add_argument('--grad_sites', action='store_true', help='kink diagnostic: per-site gradient at --init with finite-difference checks')
    p.add_argument('--hmc_warmup', type=int, default=300)
    p.add_argument('--hmc_samples', type=int, default=1000)
    p.add_argument('--hmc_segment', type=int, default=50, help='samples per saved segment')
    p.add_argument('--hmc_depth', type=int, default=8)
    p.add_argument('--hmc_chains', type=int, default=1, help='NUTS chains, vectorized on one device (dispersed starts)')
    p.add_argument('--hmc_mass', default='dense_adapt', choices=['dense_adapt', 'hessian_fixed', 'whitened_diag'],
                   help='dense_adapt = numpyro dense adaptation from the Hessian start (old default; degrades: see run_hmc); '
                   'hessian_fixed = keep the inverse-Hessian mass; whitened_diag = Hessian-whitened coords + diagonal adaptation (recommended)')
    p.add_argument('--hmc_hours', type=float, default=46., help='stop starting new segments after this many hours')
    p.add_argument('--cbc_counts', default=None, help='cbc_early.npz: CBC COUNT likelihood 1902-65 (user 2026-11-07); see --cbc_count_mode')
    p.add_argument('--cbc_count_mode', default='era', choices=['era', 'latent'],
                   help='era: presence-only before 1940 and for 1940s rows without effort; counts (incl. zeros) where effort is '
                   'recorded from 1940; 1950-65 rows without effort IMPUTED (circle median of recorded log effort, else the year '
                   'trend). latent: counts and zeros for all 1902-65 rows; missing log effort ~ N(m(t), tau^2), m / tau fitted '
                   'on the RECORDED efforts only (fixed, so counts cannot move them), integrated by moment-matching into the NB')
    p.add_argument('--cbc_winter_drift', action='store_true', help='CBC scale rises with years since the CIRCLE first recorded the '
                   'species (data-based): s + gamma (1 - exp(-dt / 10)), gamma ~ N(0, 1) -- winter concentration in newly colonised areas '
                   '(Weiss 1997: CBC grew 2.5-5x faster than breeding counts), an observation effect, not growth')
    p.add_argument('--cbc_presence', default=None, help='cbc_early.npz: add a PRESENCE-ONLY term for positive early-CBC circle-years (1902-65)')
    p.add_argument('--presence_k', type=float, default=0.43724203275856455)
    p.add_argument('--establishment_dates', default=None, help='npz from establishment_build.py: two-sided establishment '
                   'windows per state [first occurrence, first breeding] on a smooth mated-fraction indicator')
    p.add_argument('--estab_theta', type=float, default=0.5)
    p.add_argument('--estab_tau', type=float, default=0.05)
    p.add_argument('--cbc_rate', default=None, help='npz from cbc_rate_build.py: invasion-era CBC counts with effort; NB2 with free '
                   'source scale s, effort power p, log phi (3 trailing coordinates) and circle effects N(0,1) integrated by Laplace')
    p.add_argument('--disease_depression', type=float, default=None, help='SD (log scale) of a likelihood on the model '
                   'disease-on / disease-off abundance ratio by years since documented local arrival (1-5 yr), East cells dense before '
                   'the epizootic, matched to Hochachka & Dhondt 2000 Fig. 2 medians (% of a saturating Gompertz expectation)')
    p.add_argument('--depression_drift', type=float, default=None, help='SD (log scale) of N(0, SD) on log(ratio 2010-25 / ratio at 5 yr): '
                   'the depression does not keep deepening (literature: stable after ~3 yr; Hochachka et al. 2021)')
    p.add_argument('--synthetic_y', default=None, help='known-truth bias test: replace the BBS counts with synth_bbs.py output (same rows)')
    p.add_argument('--bbs_year_balance', type=float, default=0.0, help='p: route rows of year t weighted min(1, (n_ref / n_t)^p), '
                   'n_ref = mean training route rows per year 1966-75; p 1 equalises the total weight of every year (user 2026-11-06)')
    p.add_argument('--bbs_weight', type=float, default=1.0, help='weight on the BBS marginal likelihood (route-marginal NLL); '
                   'set from the within-route residual autocorrelation (AR1 ESS factor ~.88, resid_autocorr.py), not a free nudge')
    p.add_argument('--nest_weight', type=float, default=1.0, help='DIAGNOSTIC: total weight on the NestWatch log-likelihood (the model '
                   'already includes it once; adds (w - 1) x its value). User: re-weighting is a last nudge')
    p.add_argument('--bbl', default=None, help='R83 port: BBL adult dead-recovery m-array (bbl_prep.py); Seber likelihood on the '
                   'model adult survival at the banding cells + 12 reporting-rate logits (3 regions x 4 eras; 12 trailing coordinates)')
    p.add_argument('--bbl_until', type=int, default=1993, help='censor BBL intervals after this year (pre-disease)')
    p.add_argument('--year_wiggle', type=float, default=0., help='HalfNormal scale of a region (W/Plains/E) x year (1966-2025) '
                   'observation effect on BBS means, HIGH-PASSED (anything smooth at --year_wiggle_knot removed), so it absorbs short '
                   'transients / decadal wiggles but cannot carry a trend. 0 = off.')
    p.add_argument('--year_wiggle_knot', type=float, default=8.)
    p.add_argument('--bbl_disease', action='store_true', help='apply the model epizootic added mortality (disease target = survival) to '
                   'the BBL survival path: S = Sa (1 - amp * hill(N)), N = census density of the year (the forward uses the post-dispersal '
                   'pre-survival density; close). Needed when --bbl_until extends past the 1994 arrival.')
    p.add_argument('--bbl_weight', type=float, default=1.0, help='DIAGNOSTIC weight on the BBL log-likelihood (user: re-weighting is a last nudge)')
    p.add_argument('--sa_anchor', default=None, help='R81b port: "mean,sd" Normal anchor on mean adult survival over occupied EAST cells 1966-93')
    p.add_argument('--late_prior_cell', type=float, default=None, help='R103b port (DIAGNOSTIC comparator only, user ruling): '
                   'independent N(0, SD) on every occupied cell 1993 -> 2010 change of logit Sa, logit Sj, log Fmax')
    p.add_argument('--cbc_weight', type=float, default=1.0, help='DIAGNOSTIC tempering of the CBC rate NLL (user: re-weighting is a last nudge)')
    p.add_argument('--range_absence', default=None, help='npz from range_absence_build.py: (t, land cell) absence constraints '
                   '(full pre-1940 footprint + 1940-65 outside the native / introduced envelopes; rangev2 restoration)')
    p.add_argument('--cbc_source', default='union', choices=['union', 'bbl'], help='CBC circle source model: union (as before) '
                   'or bbl = BBL-anchored winter movement of breeders (cbc_obs.py); needs --cbc_bbl_pairs')
    p.add_argument('--cbc_bbl_pairs', default=None, help='bbl_summer_winter_pairs.csv (p14_bbl_sw.py)')
    p.add_argument('--cbc_drift', default='none', choices=['none', 'lin'], help='global CBC detection drift per decade (gam ~ N(0,1)); '
                   'alternative to --cbc_winter_drift (same slot)')
    p.add_argument('--route_skew', action='store_true', help='route effects ~ CENTRED skew-normal (mean 0, var 1, shape alpha ~ '
                   'N(0,5^2), one global parameter) instead of N(0,1): the fitted route effects are right-skewed (2026-11-09)')
    p.add_argument('--native_viable', default=None, help='DIAGNOSTIC (2026-11-08): npz of land-cell indices of NATIVE specimen '
                   'cells by region (keys mex, us): one-sided penalty when the regional mean log potential lambda in the first model '
                   'year falls below 0 -- the native range must be viable under 1902 habitat, as the seed (at capacity) assumes')
    p.add_argument('--native_viable_scale', type=float, default=.05, help='SD on the regional mean log lambda below 0')
    p.add_argument('--specimen_counts', action='store_true', help='specimens as COUNTS per cell-decade, NB2 with a fitted size '
                   '(specimen_lik.specimen_nll_counts) instead of presence / absence; adds one nuisance (log phi ~ N(0, 2))')
    p.add_argument('--specimens', default=None, help='effort-aware museum-specimen occupancy likelihood 1880-1940 (specimen_lik.py, '
                   'lever subagent round 5): precomputed cache npz from specimen_cache_build.py (replaces RP2 / presence-only specimens)')
    p.add_argument('--natal', default=None, help='BBL banding-movement likelihood (natal juveniles + adult breeding '
                   'dispersal; natal_lik.py): path to natal_records.npz (juv, adult arrays from natal_lik.map_records)')
    p.add_argument('--natal_form', default='class', choices=['class', 'cell'])
    p.add_argument('--natal_t_offset', type=int, default=1)
    p.add_argument('--natal_weight', type=float, default=1.0, help='DIAGNOSTIC only; leave 1')
    p.add_argument('--trough', default=None, help='DIAGNOSTIC "W,M": add 0.5 W max(D + M, 0)^2, D = 1902-1950 mean of '
                   '(mean potential lambda over Great Plains cells - over Eastern Temperate Forests cells); needs --trough_cells')
    p.add_argument('--trough_cells', default=None, help='npz with land-cell indices gp, etf (CEC Level I 9 / 8)')
    p.add_argument('--range_presence', default=None, help='npz from range_presence_build.py: (t, land cell) native-range PRESENCE '
                   'constraints, cost w * -log(1 - (1 + N/k)^-k) (the censored NB2 event paired with the absences)')
    p.add_argument('--range_presence_persist', type=float, default=None, help='persistence pi = P(cell occupied in the '
                   'presence year | occupied 1966-69): cost -log[pi P(det|N) + 1 - pi] (lever subagent round 4; DIAGNOSTIC, '
                   'pi should come from historical resurvey persistence, not tuning)')
    p.add_argument('--range_presence_k', type=float, default=None, help='k for --range_presence only (default: --presence_k); '
                   'large k -> Poisson detection 1 - e^-N, i.e. an occupancy-like constraint that stops rewarding density sooner')
    p.add_argument('--route_mean', action='store_true', help='free route-effect mean m ~ N(0,1): u = m + sigma e (an observation intercept on route rows)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    root = Path(os.environ['TAKEOVER_CODE_ROOT'])
    sys.path.insert(0, str(root / 'research' / 'exp'))
    import jax, jax.numpy as jnp
    from jax.scipy.special import gammaln
    from numpyro import handlers
    from numpyro.infer.util import log_density
    from scipy.optimize import minimize
    from harness import Sandbox, ap
    from src.model import hmc_common as hc
    from src.config_utils import load_age_model_config
    from local_map_coordinates import MapCoordinates

    t0 = time.time()
    ofx = load_age_model_config()['population_model']['observation']['observer_effect']
    sig_cfg, sig_scale = float(ofx['sigma_u']), float(ofx.get('sigma_prior_scale', 1.0))
    sb = Sandbox(os.environ['SBX_REFINED'])
    if a.synthetic_y:   # known-truth bias test: synthetic counts replace the real BBS rows everywhere (model obs term + Laplace)
        _ys = np.load(a.synthetic_y)['y']
        _yo = np.asarray(sb.data['observed_results'])
        if _ys.shape != _yo.shape:
            raise ValueError(f'synthetic_y shape {_ys.shape} != observed_results {_yo.shape}')
        sb.data = dict(sb.data); sb.data['observed_results'] = jnp.asarray(np.where(np.isfinite(_yo), _ys, _yo), dtype=jnp.asarray(_yo).dtype)
        print(json.dumps(dict(stage='synthetic_y', file=a.synthetic_y)), flush=True)
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.init) as saved:
        lat.update({k: jnp.asarray(saved[k]) for k in saved.files if not k.startswith('laplace_route_')})
        s_init = float(saved['laplace_route_sigma']) if 'laplace_route_sigma' in saved.files else sig_cfg
        _init_has_lineage = 'lineage_delta' in saved.files
        _eps_saved = np.asarray(saved['observer_route_eps'], np.float64) if 'observer_route_eps' in saved.files else None
        _m_saved = float(saved['laplace_route_mean']) if 'laplace_route_mean' in saved.files else 0.0
    sigma0 = a.sigma0 if a.sigma0 is not None else s_init
    if getattr(ap, 'LINEAGE', False) and not _init_has_lineage:   # (Sandbox.lat carries a RANDOM draw of it)   # start a new lineage offset at its prior centre
        lat['lineage_delta'] = jnp.asarray(float(ap._LIN.get('loc', 0.0)))   # (not a random prior draw: e^delta explodes)
        print(json.dumps(dict(stage='lineage_init', delta=float(lat['lineage_delta']))), flush=True)
    n_route = int(np.asarray(lat['observer_route_eps']).size)
    lat['observer_route_eps'] = jnp.zeros(n_route)          # the model sees no route effect; the Laplace term supplies it
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), a.seed)).get_trace(data=sb.data, prior_scale=1.)
    s = tr['obs']
    print(json.dumps(dict(stage='obs_site', keys=sorted(s.keys()), has_mask=s.get('mask') is not None,
                          has_scale=s.get('scale') is not None)), flush=True)
    excluded = ['dispersal_logit_intercept', 'dispersal_logit_slope'] if ap.VEIT_DEPARTURE else []
    fixed = [x for x in a.fix.split(',') if x]
    coords = MapCoordinates(tr, exclude=excluded + fixed + ['observer_route_eps'])
    del tr
    arrays, static = hc.split_data(sb.data)
    array_model = hc.array_model(ap.build_model_2d, static)

    y_np = np.asarray(sb.data['observed_results'], np.float64)
    w_np = np.asarray(sb.data['obs_weight'], np.float64) if sb.data.get('obs_weight') is not None else np.ones(y_np.shape)
    train = np.isfinite(w_np) & (w_np > 0) & np.isfinite(y_np)
    if a.drop_pseudozeros:
        _pz = np.asarray(sb.data['obs_obsroute_idx']) < 0
        print(json.dumps(dict(stage='drop_pseudozeros', rows=int((train & _pz).sum()))), flush=True)
        train = train & ~_pz
    if a.bbs_year_balance:   # time balance (user 2026-11-06): down-weight the well-observed modern years' route rows
        _yr = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])].astype(int)
        _rt = train & (np.asarray(sb.data['obs_obsroute_idx']) >= 0)
        _n = {int(v): int((_rt & (_yr == v)).sum()) for v in np.unique(_yr[_rt])}
        _ref = np.mean([_n[v] for v in range(1966, 1976) if v in _n])
        _mult = np.array([min(1.0, (_ref / max(_n.get(int(v), 1), 1)) ** a.bbs_year_balance) for v in _yr])
        w_np = np.where(_rt, w_np * _mult, w_np)
        print(json.dumps(dict(stage='bbs_year_balance', p=a.bbs_year_balance, n_ref=round(float(_ref), 1),
                              mult_by_decade={d: round(float(np.mean(_mult[_rt & (_yr >= d) & (_yr < d + 10)])), 3) for d in range(1966, 2026, 10)})), flush=True)
    w = jnp.asarray(np.where(train, w_np, 0.))              # training rows only (held-out rows weight 0, as the mask)
    y = jnp.asarray(np.where(train, y_np, 0.))              # weight-0 rows are made finite so 0 * l stays 0 in value AND gradient
    trn = jnp.asarray(train)
    print(json.dumps(dict(stage='rows', n=int(y_np.size), train=int(train.sum()), train_with_route=int((train & (np.asarray(sb.data['obs_obsroute_idx']) >= 0)).sum()))), flush=True)
    idx_np = np.asarray(sb.data['obs_obsroute_idx'])
    on = jnp.asarray(idx_np >= 0); idc = jnp.asarray(np.maximum(idx_np, 0))
    ywg = None
    if a.year_wiggle:   # region x year HIGH-PASS observation effect (overnight 2026-11-06; review hypothesis "eta 1 buys responsiveness")
        _yr = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])].astype(int)
        _lon = sb.lon[sb.lookup[np.asarray(sb.data['obs_rows']), np.asarray(sb.data['obs_cols'])]]
        _reg = np.where(_lon < -100, 0, np.where(_lon < -95, 1, 2))
        _Y = np.arange(1966, 2026); _k = np.arange(1966, 2026 + a.year_wiggle_knot, a.year_wiggle_knot)
        _B = np.maximum(0., 1. - np.abs(_Y[:, None] - _k[None, :]) / a.year_wiggle_knot)   # hats (sum to 1: constant included)
        _Q = np.eye(_Y.size) - _B @ np.linalg.pinv(_B)                                        # removes anything smooth at the knot scale
        _yi = np.clip(_yr - 1966, 0, _Y.size - 1); _act = train & (idx_np >= 0) & (_yr >= 1966)
        ywg = dict(Q=jnp.asarray(_Q), reg=jnp.asarray(_reg), yi=jnp.asarray(_yi), act=jnp.asarray(_act), nY=_Y.size, n=3 * _Y.size + 1)
        print(json.dumps(dict(stage='year_wiggle', hn_scale=a.year_wiggle, knot=a.year_wiggle_knot, rows=int(_act.sum()),
                              highpass_rank=int(round(np.trace(_Q))))), flush=True)

    def wiggle(xw):   # xw = [z (3 x nY), log s]; E = s * z Q^T (rows: regions); cost = .5|z|^2 + HalfNormal(s) with the log-s Jacobian
        z = jnp.reshape(xw[:-1], (3, ywg['nY'])); s = jnp.exp(xw[-1])
        E = s * z @ ywg['Q'].T
        return E, .5 * jnp.sum(z ** 2) + .5 * (s / a.year_wiggle) ** 2 - xw[-1]
    kappa_on = bool(ap._OBS_SPEC.get('conc_density_power'))
    pres = None
    if a.cbc_presence:   # PRESENCE-ONLY (user 2026-11-05): absences ignored (effort unknown); a positive circle-year proves presence
        c = np.load(a.cbc_presence)
        posm = (c['cbc_count'] > 0) & (np.asarray(sb.years)[c['cbc_t']] <= 1965)
        ci = c['cbc_circle'][posm]; src = c['cbc_source_idx'][ci]
        wu = np.where(src >= 0, np.maximum(c['cbc_w_local'][ci], c['cbc_w_seasonal'][ci]), 0.)   # local U 300 km SW seasonal allowance
        pres = dict(t=jnp.asarray(c['cbc_t'][posm]), src=jnp.asarray(np.maximum(src, 0)), w=jnp.asarray(wu),
                    rows=jnp.asarray(np.asarray(sb.data['land_rows'])), cols=jnp.asarray(np.asarray(sb.data['land_cols'])),
                    scale=float(sb.data['pop_scalar']), n=int(posm.sum()))
        print(json.dumps(dict(stage='cbc_presence', n_positive_circle_years=pres['n'], k=a.presence_k)), flush=True)
    estab = None
    if a.establishment_dates:   # ESTABLISHMENT WINDOWS (user 2026-11-06): dimensionless, not mu-linear; two-sided
        ze = np.load(a.establishment_dates); yl = list(map(int, np.asarray(sb.years)))
        ti = lambda y: yl.index(int(y)) if int(y) in yl else -1
        estab = dict(rows=jnp.asarray(ze['rows']), cols=jnp.asarray(ze['cols']), mask=jnp.asarray(ze['mask']),
                     t_lo=jnp.asarray([ti(o - 1) if o > 0 else -1 for o in ze['occ']]), t_hi=jnp.asarray([ti(b) for b in ze['breed']]),
                     has_lo=jnp.asarray([1. if (o > 0 and ti(o - 1) >= 0) else 0. for o in ze['occ']]), names=[str(x) for x in ze['names']])
        print(json.dumps(dict(stage='establishment', states=estab['names'], theta=a.estab_theta, tau=a.estab_tau)), flush=True)
    crate = None
    if a.cbc_rate:   # CBC RATE term (FINAL-MODEL ROUND 2026-11-06): within-circle rate of increase; level absorbed by s
        zc = np.load(a.cbc_rate)
        crate = dict(t=jnp.asarray(zc['t']), c=jnp.asarray(zc['circle']), y=jnp.asarray(zc['count'], float), le=jnp.asarray(zc['log_effort']),
                     src=jnp.asarray(zc['src']), w=jnp.asarray(zc['w']), n=int(zc['circle'].max()) + 1,
                     rows=jnp.asarray(np.asarray(sb.data['land_rows'])), cols=jnp.asarray(np.asarray(sb.data['land_cols'])),
                     scale=float(sb.data['pop_scalar']))
        print(json.dumps(dict(stage='cbc_rate', rows=int(zc['t'].size), circles=crate['n'], positives=int((zc['count'] > 0).sum()))), flush=True)
    ccnt = None
    if a.cbc_counts:   # CBC COUNT likelihood (user 2026-11-07): effort-aware, circle effects integrated out (Laplace), two modes
        zc = np.load(a.cbc_counts); _yrs = np.asarray(sb.years).astype(int)[zc['cbc_t']]
        keep = (_yrs <= 1965) & (np.asarray(zc['cbc_train']) > 0)
        rec = np.asarray(zc['cbc_effort_missing']) == 0; le_raw = np.asarray(zc['cbc_log_effort'], float); yc = np.asarray(zc['cbc_count'], float)
        rk = keep & rec; _kn = np.array([1920., 1930., 1940., 1950., 1960., 1966.])

        def _basis(yy):
            yy = np.clip(np.asarray(yy, float), _kn[0], _kn[-1])
            return np.maximum(0., 1. - np.abs(yy[:, None] - _kn[None, :]) / 10.)
        _coef, *_ = np.linalg.lstsq(_basis(_yrs[rk]), le_raw[rk], rcond=None)
        m_t = _basis(_yrs) @ _coef; tau = float(np.std(le_raw[rk] - m_t[rk]))
        circ = np.asarray(zc['cbc_circle'])
        if a.cbc_count_mode == 'era':
            cnt = keep & ((rec & (_yrs >= 1940)) | (~rec & (_yrs >= 1950)))
            le = le_raw.copy(); v = np.zeros_like(le)
            for i in np.flatnonzero(cnt & ~rec):   # circle median of recorded log effort (1940-65), else the year trend
                o = rk & (circ == circ[i]) & (_yrs >= 1940)
                le[i] = np.median(le_raw[o]) if o.any() else m_t[i]
            prs = keep & ~cnt & (yc > 0)          # presence-only rows: pre-1940 positives and 1940s positives without effort
        else:
            cnt = keep; le = np.where(rec, le_raw, m_t); v = np.where(rec, 0., tau ** 2); prs = np.zeros_like(keep)
        _fp = {}
        for _c, _y, _n in zip(circ, _yrs, yc):   # first year each circle recorded the species (<= 1965, any row)
            if _n > 0 and _y <= 1965:
                _fp[_c] = min(_fp.get(_c, 9999), _y)
        _dt = np.array([max(y_ - _fp.get(c_, 9999), 0) for c_, y_ in zip(circ, _yrs)], float)
        gdrift = 1. - np.exp(-_dt / 10.)
        ci = circ[cnt]; uc, cidx = np.unique(ci, return_inverse=True)
        src = zc['cbc_source_idx'][ci]; wu = np.where(src >= 0, np.maximum(zc['cbc_w_local'][ci], zc['cbc_w_seasonal'][ci]), 0.)
        ccnt = dict(g=jnp.asarray(gdrift[cnt]), npar=4 if a.cbc_winter_drift else 3,
                    t=jnp.asarray(zc['cbc_t'][cnt]), c=jnp.asarray(cidx), y=jnp.asarray(yc[cnt]), le=jnp.asarray(le[cnt]),
                    v=jnp.asarray(v[cnt]), src=jnp.asarray(np.maximum(src, 0)), w=jnp.asarray(wu), n=int(uc.size),
                    rows=jnp.asarray(np.asarray(sb.data['land_rows'])), cols=jnp.asarray(np.asarray(sb.data['land_cols'])),
                    scale=float(sb.data['pop_scalar']))
        if prs.any() and pres is None:   # era mode supplies its own presence-only rows
            ci2 = circ[prs]; src2 = zc['cbc_source_idx'][ci2]
            pres = dict(t=jnp.asarray(zc['cbc_t'][prs]), src=jnp.asarray(np.maximum(src2, 0)),
                        w=jnp.asarray(np.where(src2 >= 0, np.maximum(zc['cbc_w_local'][ci2], zc['cbc_w_seasonal'][ci2]), 0.)),
                        rows=jnp.asarray(np.asarray(sb.data['land_rows'])), cols=jnp.asarray(np.asarray(sb.data['land_cols'])),
                        scale=float(sb.data['pop_scalar']), n=int(prs.sum()))
        print(json.dumps(dict(stage='cbc_counts', mode=a.cbc_count_mode, count_rows=int(cnt.sum()), zeros=int((cnt & (yc == 0)).sum()),
                              imputed_or_latent=int((cnt & ~rec).sum()), presence_rows=int(prs.sum()), circles=ccnt['n'],
                              effort_trend=[round(float(x), 3) for x in _coef], tau=round(tau, 3))), flush=True)
        if a.cbc_source == 'bbl':   # BBL-anchored winter source model (cbc_obs.py, lever-probe subagent; no free parameters)
            import cbc_obs, pandas as pd
            _pp = pd.read_csv(a.cbc_bbl_pairs)
            _par = {k: dict(v) for k, v in cbc_obs.BBL.items()}
            _par['east']['kappa'] = float(cbc_obs.kappa_from_bearings(_pp[(_pp.lon_dd_s > -95) & (_pp.km > 80)].bear.values))
            _W, _ = cbc_obs.build_source_matrix(dict(cbc_source_idx=zc['cbc_source_idx'], cbc_w_local=zc['cbc_w_local']), sb.latd, sb.lon, params=_par)
            ccnt['W'] = jnp.asarray(np.asarray(_W, np.float64)); ccnt['circ_row'] = jnp.asarray(circ[cnt])
            print(json.dumps(dict(stage='cbc_source_bbl', kappa_east=_par['east']['kappa'], W_shape=list(_W.shape))), flush=True)
        if a.cbc_drift == 'lin':     # global detection drift per decade (nuisance gam ~ N(0,1)) in the existing gam slot
            ccnt['g'] = jnp.asarray((_yrs[cnt] - 1940.) / 10.); ccnt['npar'] = 4

    def cbc_count_cost(trace, cc):   # NB with circle effects (Laplace); missing-effort rows: lognormal noise moment-matched into phi
        s_, pw, lphi = cc[0], cc[1], cc[2]; ph0 = jnp.exp(lphi)
        Nl = (trace['Na_grid']['value'] + trace['Nj_grid']['value'])[:, ccnt['rows'], ccnt['cols']] * ccnt['scale']
        if 'W' in ccnt:   # --cbc_source bbl: A_row = sum_j N[t_row, j] W[circle_row, j]
            A = jnp.sum(Nl[ccnt['t']] * ccnt['W'][ccnt['circ_row']], axis=1)
        else:
            A = jnp.sum(Nl[ccnt['t'][:, None], ccnt['src']] * ccnt['w'], axis=1)
        vv = pw ** 2 * ccnt['v']
        gam = cc[3] if ccnt['npar'] == 4 else 0.
        base = s_ + gam * ccnt['g'] + pw * ccnt['le'] + .5 * vv + jnp.log(jnp.maximum(A, 1e-12))
        ph = 1. / ((1. + 1. / ph0) * jnp.exp(vv) - 1.)          # Var = mu + mu^2 [(1 + 1/phi) e^v - 1]
        yy = ccnt['y']; ci_ = ccnt['c']; nc = ccnt['n']

        def ll(u):
            mu = jnp.exp(base + u)
            return (gammaln(yy + ph) - gammaln(ph) - gammaln(yy + 1.) + ph * (jnp.log(ph) - jnp.log(ph + mu))
                    + yy * (jnp.log(mu) - jnp.log(ph + mu)))
        f = lambda u: jnp.sum(ll(u)); d1 = jax.grad(f); d2 = lambda u: jax.jvp(d1, (u,), (jnp.ones_like(u),))[1]
        seg = lambda q: jax.ops.segment_sum(q, ci_, num_segments=nc)

        def step(c, _):
            u = c[ci_]; g = seg(d1(u)) - c; H = jnp.maximum(-seg(d2(u)), 0.) + 1.
            return c + jnp.clip(g / H, -1., 1.), None
        c, _ = jax.lax.scan(step, jnp.zeros(nc), None, length=40)
        H = jnp.maximum(-seg(d2(c[ci_])), 0.)
        nll = -f(c[ci_]) + .5 * jnp.sum(c ** 2) + .5 * jnp.sum(jnp.log1p(H))
        prior = .5 * ((pw - .6) / .3) ** 2 + .5 * lphi ** 2 + .5 * (s_ / 5.) ** 2 + .5 * gam ** 2
        return nll + prior, nll

    def cbc_rate_cost(trace, cb):
        s, pw, lphi = cb[0], cb[1], cb[2]; ph = jnp.exp(lphi)
        Nl = (trace['Na_grid']['value'] + trace['Nj_grid']['value'])[:, crate['rows'], crate['cols']] * crate['scale']
        A = jnp.sum(Nl[crate['t'][:, None], crate['src'][crate['c']]] * crate['w'][crate['c']], axis=1)
        base = s + pw * crate['le'] + jnp.log(jnp.maximum(A, 1e-12)); yy = crate['y']; ci_ = crate['c']; nc = crate['n']

        def ll(u):
            mu = jnp.exp(base + u)
            return gammaln(yy + ph) - gammaln(ph) - gammaln(yy + 1.) + ph * (lphi - jnp.log(ph + mu)) + yy * (jnp.log(mu) - jnp.log(ph + mu))
        f = lambda u: jnp.sum(ll(u)); d1 = jax.grad(f); d2 = lambda u: jax.jvp(d1, (u,), (jnp.ones_like(u),))[1]
        seg = lambda v: jax.ops.segment_sum(v, ci_, num_segments=nc)

        def step(c, _):
            u = c[ci_]; g = seg(d1(u)) - c; H = jnp.maximum(-seg(d2(u)), 0.) + 1.
            return c + jnp.clip(g / H, -1., 1.), None
        c, _ = jax.lax.scan(step, jnp.zeros(nc), None, length=40)
        H = jnp.maximum(-seg(d2(c[ci_])), 0.)
        nll = -f(c[ci_]) + .5 * jnp.sum(c ** 2) + .5 * jnp.sum(jnp.log1p(H))
        prior = .5 * ((pw - .6) / .3) ** 2 + .5 * lphi ** 2 + .5 * (s / 5.) ** 2
        return nll + prior, nll
    dd = None
    if a.disease_depression or a.depression_drift:   # FINAL ROUND: Hochachka & Dhondt 2000 (PNAS 97:5303) Fig. 2
        _yl = np.asarray(sb.years).astype(int); _T = len(_yl)
        _on = np.asarray(sb.data['disease_onset'], float).ravel()                 # documented arrival, timestep units, per land cell
        _y = np.asarray(sb.data['observed_results'], float); _real = np.asarray(sb.data['obs_obsroute_idx']) >= 0
        _cell = sb.lookup[np.asarray(sb.data['obs_rows']), np.asarray(sb.data['obs_cols'])]; _ty = _yl[np.asarray(sb.data['obs_time_indices'])]
        _m = _real & (_cell >= 0) & (_ty >= 1990) & (_ty <= 1993) & np.isfinite(_y)
        _dens = np.bincount(_cell[_m], weights=_y[_m], minlength=len(sb.rows)) / np.maximum(np.bincount(_cell[_m], minlength=len(sb.rows)), 1)
        _east = sb.region['east'] & (np.bincount(_cell[_m], minlength=len(sb.rows)) > 0) & (_on > 0) & (_on < _T)
        _dense = _east & (_dens >= np.median(_dens[_east]))                        # H&D used blocks with sustained high abundance
        _ci = np.flatnonzero(_dense); _o = np.round(_on[_ci]).astype(int)
        _K = [1, 2, 3, 4, 5]; _tgt = np.log(np.array([.83, .55, .43, .49, .37]))   # Fig. 2 medians at whole years (read off the box plot)
        _late = (_yl >= 2010) & (_yl <= 2025)
        dd = dict(rows=jnp.asarray(np.asarray(sb.rows)[_ci]), cols=jnp.asarray(np.asarray(sb.cols)[_ci]),
                  tk=jnp.asarray(np.clip(_o[None, :] + np.array(_K)[:, None], 0, _T - 1)), tgt=jnp.asarray(_tgt),
                  late=jnp.asarray(np.flatnonzero(_late)), n=int(_ci.size))
        print(json.dumps(dict(stage='disease_depression', cells=dd['n'], onset_years=[int(_yl[min(max(v, 0), _T - 1)]) for v in np.unique(_o)][:8],
                              sd=a.disease_depression, drift_sd=a.depression_drift)), flush=True)
        DD_STASH = {}
        _orig_fwd = ap.forward_sim_age_structured

        def _fwd_cf(*aa, **kw):   # run the forward twice: as fitted, and with the epizootic's added mortality removed
            import inspect as _insp
            b = _insp.signature(_orig_fwd).bind(*aa, **kw); b.apply_defaults()
            out = _orig_fwd(*b.args, **b.kwargs)
            if b.arguments.get('disease_m_amp') is not None:
                k2 = dict(b.arguments); k2['disease_m_amp'] = jnp.zeros_like(k2['disease_m_amp'])
                DD_STASH['off'] = _orig_fwd(**k2)[0]; DD_STASH['on'] = out[0]
            return out
        ap.forward_sim_age_structured = _fwd_cf

    def depression_cost():
        on, off = DD_STASH['on'], DD_STASH['off']
        Non = on[dd['tk'], dd['rows'][None, :], dd['cols'][None, :]].sum(1); Noff = off[dd['tk'], dd['rows'][None, :], dd['cols'][None, :]].sum(1)
        lr = jnp.log(jnp.maximum(Non, 1e-30)) - jnp.log(jnp.maximum(Noff, 1e-30))       # (5,) log ratio by years since arrival
        c = 0.
        if a.disease_depression:
            c = c + .5 * jnp.sum(((lr - dd['tgt']) / a.disease_depression) ** 2)
        if a.depression_drift:
            Lon = on[dd['late']][:, dd['rows'], dd['cols']].sum(); Loff = off[dd['late']][:, dd['rows'], dd['cols']].sum()
            c = c + .5 * ((jnp.log(Lon / Loff) - lr[-1]) / a.depression_drift) ** 2
        return c, lr

    bbl = None
    if a.bbl:   # R83 port (Plan L C1)
        _bz = np.load(a.bbl); _yrs = np.asarray(sb.years); _K = int(_bz['K']); _Y = _bz['by'][:, None] + np.arange(1, _K + 1)[None, :]
        bbl = dict(cell=jnp.asarray(_bz['cell']), yidx=jnp.asarray(np.clip(_Y - _yrs[0], 0, len(_yrs) - 1)),
                   valid=jnp.asarray(_Y <= min(_yrs[-1], a.bbl_until)), n=jnp.asarray(_bz['n'], float),
                   m=jnp.asarray(np.where(_Y <= a.bbl_until, _bz['m'], 0.), float),
                   ridx=jnp.asarray(_bz['grp'][:, None] * 4 + np.digitize(_Y, [1980, 1995, 2006])),
                   sel=jnp.asarray(np.asarray(sb.rows) * sb.Nx + np.asarray(sb.cols)), n_land=len(sb.rows))
        print(json.dumps(dict(stage='bbl', cohorts=int(_bz['n'].size), released=float(_bz['n'].sum()), recoveries=float(np.where(_Y <= a.bbl_until, _bz['m'], 0).sum()), until=a.bbl_until)), flush=True)
    saa = None
    if a.sa_anchor:
        _m, _s = map(float, a.sa_anchor.split(',')); _yl = np.asarray(sb.years)
        saa = dict(mu=_m, sd=_s, t=jnp.asarray(np.flatnonzero((_yl >= 1966) & (_yl <= 1993))),
                   c=jnp.asarray(np.flatnonzero(np.isfinite(sb.first_obs) & sb.region['east'])),
                   sel=jnp.asarray(np.asarray(sb.rows) * sb.Nx + np.asarray(sb.cols)), n_land=len(sb.rows))

    def _land_sa(trace, sel, n_land):
        X = jnp.reshape(trace['Sa_flat']['value'], (len(sb.years), -1))
        return X if X.shape[1] == n_land else X[:, sel]

    def bbl_cost(trace, rlog):   # Brownie / Seber, as refit_fields.bbl_ll (R83)
        Sa = _land_sa(trace, bbl['sel'], bbl['n_land'])
        if 'growth_s_real' in trace:   # population_model.growth_kernel: banded adults survive at the LEARNED realised rate
            Sa = trace['growth_s_real']['value']          # (time, N_land), pre-disease, at that year's simulated density
        if a.bbl_disease and 'disease_mortality_amp' in trace:
            amp = trace['disease_mortality_amp']['value']
            amp = amp if amp.shape[-1] == bbl['n_land'] else jnp.reshape(amp, (len(sb.years), -1))[:, bbl['sel']]
            Nt = (trace['Na_grid']['value'] + trace['Nj_grid']['value'])[:, np.asarray(sb.rows), np.asarray(sb.cols)]
            hn_ = trace['disease_hill_n']['value']; kh = trace['disease_k_half']['value']
            hn = jnp.power(jnp.maximum(Nt, 0.) + 1e-12, hn_)
            Sa = Sa * (1. - amp * hn / (hn + kh ** hn_))
        if 'survival_dd_c' in trace:   # adult_survival_dd: banded adults survive at the density-braked rate
            from src.model.density_regulation import crowding_argument
            _cs = jnp.reshape(trace['survival_dd_c']['value'], (len(sb.years), -1))
            _Kl = jnp.reshape(trace['K_flat']['value'], (len(sb.years), -1))
            if _cs.shape[1] != bbl['n_land']:
                _cs, _Kl = _cs[:, bbl['sel']], _Kl[:, bbl['sel']]
            _Nt = (trace['Na_grid']['value'] + trace['Nj_grid']['value'])[:, np.asarray(sb.rows), np.asarray(sb.cols)]
            Sa = Sa / (1. + _cs * crowding_argument(_Nt / jnp.maximum(_Kl, 1e-12)))
        S = jnp.clip(Sa[bbl['yidx'], bbl['cell'][:, None]], 1e-6, 1 - 1e-6); S = jnp.where(bbl['valid'], S, 1.)
        alive = jnp.exp(jnp.cumsum(jnp.log(S), 1) - jnp.log(S)); r = jax.nn.sigmoid(rlog[bbl['ridx']])
        P = jnp.where(bbl['valid'], alive * (1. - S) * r, 0.); never = jnp.clip(1. - P.sum(1), 1e-12, 1.)
        ll = jnp.sum(bbl['m'] * jnp.log(jnp.maximum(P, 1e-300))) + jnp.sum((bbl['n'] - bbl['m'].sum(1)) * jnp.log(never))
        return -a.bbl_weight * ll + .5 * jnp.sum(((rlog - np.log(.005 / .995)) / 1.5) ** 2), -ll

    lpc = None
    if a.late_prior_cell:   # R103b port: occupied = any positive real BBS row in the cell
        _yl = list(map(int, np.asarray(sb.years)))
        lpc = dict(t0=_yl.index(1993), t1=_yl.index(2010), occ=jnp.asarray(np.flatnonzero(np.isfinite(sb.first_obs))),
                   sel=jnp.asarray(np.asarray(sb.rows) * sb.Nx + np.asarray(sb.cols)), n_land=len(sb.rows))
        print(json.dumps(dict(stage='late_prior_cell', sd=a.late_prior_cell, occupied=int(lpc['occ'].size))), flush=True)

    def late_cell_cost(trace):
        out = 0.
        for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'):
            X = jnp.reshape(trace[k]['value'], (len(sb.years), -1))
            X = X if X.shape[1] == lpc['n_land'] else X[:, lpc['sel']]
            X = jnp.log(jnp.maximum(X, 1e-6)) if k == 'Fmax_flat' else (lambda c: jnp.log(c) - jnp.log1p(-c))(jnp.clip(X, 1e-6, 1 - 1e-6))
            d = X[lpc['t1']][lpc['occ']] - X[lpc['t0']][lpc['occ']]
            out = out + .5 * jnp.sum((d / a.late_prior_cell) ** 2)
        return out
    absn = None
    if a.range_absence:   # rangev2-style ABSENCE constraints on cell totals (zero cost k log(1 + N/k), N on the route-count scale)
        za = np.load(a.range_absence)
        absn = dict(t=jnp.asarray(za['t']), r=jnp.asarray(za['row']), c=jnp.asarray(za['col']), w=jnp.asarray(za['w']),
                    scale=float(sb.data['pop_scalar']), n=int(za['t'].size))
        print(json.dumps(dict(stage='range_absence', n_cell_years=absn['n'])), flush=True)
    rprs = None
    if a.range_presence:  # native-range PRESENCE constraints on cell totals (same NB2 detection model as the absences)
        zp = np.load(a.range_presence)
        rprs = dict(t=jnp.asarray(zp['t']), r=jnp.asarray(zp['row']), c=jnp.asarray(zp['col']), w=jnp.asarray(zp['w']),
                    scale=float(sb.data['pop_scalar']), n=int(zp['t'].size))
        print(json.dumps(dict(stage='range_presence', n_cell_years=rprs['n'], total_weight=float(np.sum(zp['w'])))), flush=True)
    nat = None
    spc = None
    if a.specimens:   # P(y_cd > 0) = 1 - exp(-s_era E^beta g(N_cd)); nuisances [log s pre-1910, log s 1910-40, logit beta]
        import specimen_lik as SL
        _z = np.load(a.specimens)
        spc = {k: jnp.asarray(_z[k]) for k in ('w', 'cell', 'E', 'pos', 'tmask', 'era', 'y')}
        spc.update(rows=jnp.asarray(np.asarray(sb.data['land_rows'])), cols=jnp.asarray(np.asarray(sb.data['land_cols'])),
                   scale=float(sb.data['pop_scalar']), SL=SL)
        print(json.dumps(dict(stage='specimens', rows=int(_z['w'].size), positive=int(_z['pos'].sum()),
                              weighted_absences=float(((1 - _z['pos']) * _z['w']).sum()), source=str(a.specimens))), flush=True)
    if a.natal:   # banding-movement likelihood (2026-11-07 night; module natal_lik.py by the lever-probe subagent, validated)
        import natal_lik as NLK
        _nr = np.load(a.natal)
        _keys = ('land_rows', 'land_cols', 'juvenile_fft_kernel_stack', 'juvenile_edge_correction_stack',
                 'adult_fft_kernel', 'adult_edge_correction', 'initpop_latent')
        _dnp = {k: np.asarray(sb.data[k]) for k in _keys}
        nat = dict(cache=NLK.build_cache(_dnp, np.asarray(sb.years), sb.latd, sb.lon, juv=_nr['juv'], adult=_nr['adult'], dtype=np.float64),
                   T=len(sb.years), sel=jnp.asarray(np.asarray(sb.rows) * sb.Nx + np.asarray(sb.cols)), NLK=NLK)
        print(json.dumps(dict(stage='natal', juv=int(_nr['juv'].shape[0]), adult=int(_nr['adult'].shape[0]),
                              form=a.natal_form, t_offset=a.natal_t_offset, weight=a.natal_weight)), flush=True)
    nviab = None
    if a.native_viable:
        _nv = np.load(a.native_viable)
        nviab = dict(regions={k: jnp.asarray(np.asarray(_nv[k]).astype(int)) for k in _nv.files},
                     sel=jnp.asarray(np.asarray(sb.rows) * sb.Nx + np.asarray(sb.cols)), n_land=len(sb.rows))
        print(json.dumps(dict(stage='native_viable', scale=a.native_viable_scale, cells={k: int(v.size) for k, v in nviab['regions'].items()})), flush=True)

    def native_viable_cost(trace):
        def L(k):
            X = jnp.reshape(trace[k]['value'], (len(sb.years), -1))
            return X if X.shape[1] == nviab['n_land'] else X[:, nviab['sel']]
        lam0 = L('Sa_flat')[0] + L('Sj_flat')[0] * L('Fmax_flat')[0]
        c = 0.
        for cells in nviab['regions'].values():
            mlog = jnp.mean(jnp.log(jnp.maximum(lam0[cells], 1e-6)))
            c = c + .5 * (jnp.minimum(mlog, 0.) / a.native_viable_scale) ** 2
        return c

    trough = None
    if a.trough:   # DIAGNOSTIC (user-approved 2026-11-07): Great Plains as the early trough -- soft hinge on the 1902-1950 mean
        _tz = np.load(a.trough_cells); _w, _mg = map(float, a.trough.split(','))   # potential lambda, GP below E Temp. Forests
        _yl = np.asarray(sb.years).astype(int)
        trough = dict(t=jnp.asarray(np.flatnonzero((_yl >= 1902) & (_yl <= 1950))), gp=jnp.asarray(_tz['gp']), etf=jnp.asarray(_tz['etf']),
                      w=_w, margin=_mg, sel=jnp.asarray(np.asarray(sb.rows) * sb.Nx + np.asarray(sb.cols)), n_land=len(sb.rows))
        print(json.dumps(dict(stage='trough', weight=_w, margin=_mg, gp_cells=int(_tz['gp'].size), etf_cells=int(_tz['etf'].size))), flush=True)

    def trough_cost(trace):
        def L(k):
            X = jnp.reshape(trace[k]['value'], (len(sb.years), -1))
            return X if X.shape[1] == trough['n_land'] else X[:, trough['sel']]
        lam = (L('Sa_flat') + L('Sj_flat') * L('Fmax_flat'))[trough['t']]
        D = jnp.mean(lam[:, trough['gp']].mean(1) - lam[:, trough['etf']].mean(1))
        return 0.5 * trough['w'] * jnp.maximum(D + trough['margin'], 0.) ** 2, D

    lsig_prior = lambda sg: -0.5 * (sg / sig_scale) ** 2       # HalfNormal log density up to a constant

    def row_ll(u, mu0, phib, kap):    # NB2 log pmf at mean mu0 e^u, concentration phib * max(mu,1e-3)^kappa (the model's own)
        mu = mu0 * jnp.exp(u)
        ph = phib * (jnp.power(jnp.maximum(mu, 1e-3), kap) if kappa_on else 1.)
        return (gammaln(y + ph) - gammaln(ph) - gammaln(y + 1.) + ph * (jnp.log(ph) - jnp.log(ph + mu))
                + y * (jnp.log(jnp.maximum(mu, 1e-300)) - jnp.log(ph + mu)))

    def route_prior(alpha):
        """-log p(e) and its first two derivatives for the route-effect prior. alpha None: standard normal (as before).
        alpha given (--route_skew): CENTRED skew-normal with mean 0, variance 1 and shape alpha (alpha = 0 -> normal), so
        sigma keeps its meaning as the route-effect SD and the route mean stays 0. Includes the normalising constant, so the
        Laplace NLL is -f + sum P(e*) + .5 sum log(H_data + P''(e*)) - .5 n log(2 pi) for either prior."""
        if alpha is None:
            P = lambda e: 0.5 * e ** 2 + 0.5 * jnp.log(2. * jnp.pi)
        else:
            dl = alpha / jnp.sqrt(1. + alpha ** 2); om = 1. / jnp.sqrt(1. - 2. * dl ** 2 / jnp.pi); xi = -om * dl * jnp.sqrt(2. / jnp.pi)
            P = lambda e: (-jnp.log(2.) + jnp.log(om) + 0.5 * jnp.log(2. * jnp.pi) + 0.5 * ((e - xi) / om) ** 2
                           - jax.scipy.special.log_ndtr(alpha * (e - xi) / om))
        P1 = jax.vmap(jax.grad(P)); P2 = jax.vmap(jax.grad(jax.grad(P)))
        return P, P1, P2

    def laplace(mu0, phib, kap, sg, m=0., e_init=None, alpha=None):
        """Laplace-marginal observation NLL over training rows; returns (nll, e*, H)."""
        f = lambda u: jnp.sum(w * row_ll(u, mu0, phib, kap))
        d1 = jax.grad(f)
        d2 = lambda u: jax.jvp(d1, (u,), (jnp.ones_like(u),))[1]        # rows are independent: Hessian is diagonal
        seg = lambda v: jax.ops.segment_sum(jnp.where(on, v, 0.), idc, num_segments=n_route)
        P, P1, P2 = route_prior(alpha)

        def step(e, _):
            u = jnp.where(on, m + sg * e[idc], 0.)
            g = sg * seg(d1(u)) - P1(e)
            H = sg ** 2 * jnp.maximum(-seg(d2(u)), 0.) + P2(e)
            return e + jnp.clip(g / H, -1., 1.), None      # damped: a cold start can see |g| ~ 1e3
        e, _ = jax.lax.scan(step, jnp.zeros(n_route) if e_init is None else e_init, None, length=a.newton)
        u = jnp.where(on, m + sg * e[idc], 0.)
        H = sg ** 2 * jnp.maximum(-seg(d2(u)), 0.)
        if alpha is None:   # unchanged expression (bitwise-identical to earlier fits)
            nll = -f(u) + 0.5 * jnp.sum(e ** 2) + 0.5 * jnp.sum(jnp.log1p(H))
        else:   # routes without training rows integrate to exactly 0 (prior only): sum over routes WITH data
            has = seg(jnp.ones_like(u)) > 0
            nll = -f(u) + jnp.sum(jnp.where(has, P(e) + 0.5 * jnp.log(H + P2(e)) - 0.5 * jnp.log(2. * jnp.pi), 0.))
        return nll, e, H, -f(u), jnp.sum(~jnp.isfinite(row_ll(u, mu0, phib, kap)) & trn), jnp.max(jnp.abs(e))

    def parts(x, e_init=None):
        x, rsk = (x[:-1], x[-1]) if a.route_skew else (x, None)
        _nsp = 4 if a.specimen_counts else 3
        x, xsp = (x[:-_nsp], x[-_nsp:]) if spc is not None else (x, None)
        x, xn = (x[:-2], x[-2:]) if nat is not None else (x, None)
        x, xw = (x[:-ywg['n']], x[-ywg['n']:]) if ywg is not None else (x, None)
        x, cc3 = (x[:-ccnt['npar']], x[-ccnt['npar']:]) if ccnt is not None else (x, None)
        x, rl = (x[:-12], x[-12:]) if bbl is not None else (x, None)
        x, cb = (x[:-3], x[-3:]) if crate is not None else (x, None)
        x, m = (x[:-1], x[-1]) if a.route_mean else (x, 0.)
        xs, lsg = (x, jnp.log(sigma0)) if a.fix_sigma else (x[:-1], x[-1])
        sg = jnp.exp(lsg)
        lp, trace = log_density(array_model, (), {'arrays': arrays}, coords.constrain(xs))
        so = trace['obs']; v = so['fn'].log_prob(so['value'])
        v = v * (so['scale'] if so.get('scale') is not None else 1.)
        if so.get('mask') is not None:
            v = jnp.where(so['mask'], v, 0.)
        obs_model = jnp.sum(v)                                    # the model's obs term at route effect 0 (removed)
        mu0 = jnp.where(trn, trace['expected_obs']['value'], 1.)  # includes the first-year offset; route effect 0
        wcost = 0.
        if ywg is not None:
            _E, wcost = wiggle(xw)
            mu0 = mu0 * jnp.exp(jnp.where(ywg['act'], _E[ywg['reg'], ywg['yi']], 0.))
        kap = trace['obs_conc_kappa']['value'] if kappa_on else 0.
        phib = jnp.where(trn, trace['obs_concentration']['value'], 1.) / (jnp.power(jnp.maximum(mu0, 1e-3), kap) if kappa_on else 1.)
        nll_r, e, H, cond, nbad, emax = laplace(mu0, phib, kap, sg, m, e_init, alpha=rsk)
        if rsk is not None:   # weak prior on the skew-normal shape: alpha ~ N(0, 5^2)
            nll_r = nll_r + 0.5 * (rsk / 5.) ** 2
        sn = trace['nest_loglik']; nest = jnp.sum(sn['fn'].log_prob(sn['value']) * (sn['scale'] if sn.get('scale') is not None else 1.))
        prior_sig = 0. if a.fix_sigma else lsig_prior(sg)
        pcost = 0.
        if pres is not None:
            Nl = (trace['Na_grid']['value'] + trace['Nj_grid']['value'])[:, pres['rows'], pres['cols']] * pres['scale']
            A = jnp.sum(Nl[pres['t'][:, None], pres['src']] * pres['w'], axis=1)
            kk = a.presence_k
            pcost = jnp.sum(-jnp.log(jnp.maximum(-jnp.expm1(-kk * jnp.log1p(A / kk)), 1e-300)))
        ecost = 0.
        if estab is not None:
            Ng = trace['Na_grid']['value'] + trace['Nj_grid']['value']
            g = trace['allee_gamma']['value']; kk_ = ap.ALLEE_AGG_K

            def mated(x):
                return -jnp.expm1(-x) if kk_ is None else -jnp.expm1(-kk_ * jnp.log1p(x / kk_))

            def E(tidx):   # P(state established at year index tidx): OR over the state's cells of sigmoid((mated - theta)/tau)
                mf = mated(g * Ng[tidx[:, None], estab['rows'], estab['cols']])
                e = jax.nn.sigmoid((mf - a.estab_theta) / a.estab_tau) * estab['mask']
                return 1. - jnp.exp(jnp.sum(jnp.log1p(-jnp.clip(e, 0., 1. - 1e-12)), axis=1))
            e_lo = E(jnp.maximum(estab['t_lo'], 0)); e_hi = E(estab['t_hi'])
            ecost = -jnp.sum(estab['has_lo'] * jnp.log(jnp.clip(1. - e_lo, 1e-9, 1.))) - jnp.sum(jnp.log(jnp.clip(e_hi, 1e-9, 1.)))
        acost = 0.
        if absn is not None:
            Na = (trace['Na_grid']['value'] + trace['Nj_grid']['value'])[absn['t'], absn['r'], absn['c']] * absn['scale']
            acost = jnp.sum(absn['w'] * a.presence_k * jnp.log1p(jnp.maximum(Na, 0.) / a.presence_k))
            if 'obs' in trace:   # the old pseudo-zeros are replaced: remove their contribution (rows with no route id)
                pass
        rpcost = 0.
        if rprs is not None:
            Np = (trace['Na_grid']['value'] + trace['Nj_grid']['value'])[rprs['t'], rprs['r'], rprs['c']] * rprs['scale']
            kk = a.range_presence_k or a.presence_k
            _pdet = -jnp.expm1(-kk * jnp.log1p(jnp.maximum(Np, 0.) / kk))
            if a.range_presence_persist is not None:   # back-cast honesty: occupied in year t only with probability pi
                _pdet = a.range_presence_persist * _pdet + (1. - a.range_presence_persist)
            rpcost = jnp.sum(rprs['w'] * -jnp.log(jnp.maximum(_pdet, 1e-300)))
        ccost, cnll = (0., 0.) if cb is None else cbc_rate_cost(trace, cb)
        ccost = a.cbc_weight * ccost
        cc_nll = 0.
        if ccnt is not None:
            _cc, cc_nll = cbc_count_cost(trace, cc3); ccost = ccost + _cc
        if lpc is not None:
            ccost = ccost + late_cell_cost(trace)
        dcost, dlr = 0., jnp.zeros(5)
        if dd is not None:
            dcost, dlr = depression_cost(); ccost = ccost + dcost
        if a.nest_weight != 1.0:
            ccost = ccost - (a.nest_weight - 1.0) * (trace['nest_loglik_exposure']['value'] + trace['nest_loglik_count']['value'])
        bnll = 0.
        if bbl is not None:
            _bc, bnll = bbl_cost(trace, rl); ccost = ccost + _bc
        if saa is not None:
            _sa = _land_sa(trace, saa['sel'], saa['n_land'])[saa['t']][:, saa['c']].mean()
            ccost = ccost + .5 * ((_sa - saa['mu']) / saa['sd']) ** 2
        tcost, tD = (0., 0.) if trough is None else trough_cost(trace)
        ncost, njuv, nad = 0., 0., 0.
        if nat is not None:
            gm = jax.nn.softplus(xn[0]); ga = xn[1]
            _nn, _np = nat['NLK'].natal_nll(trace, nat['cache'], nat['T'], nat['sel'], gamma=gm, adult_stay_logit=ga,
                                             form=a.natal_form, t_offset=a.natal_t_offset, share=None, return_parts=True,
                                             adult_fixed_p=nat.get('adult_pa'))
            ncost = a.natal_weight * _nn + .5 * gm ** 2 + .5 * (ga / 2.) ** 2
            njuv, nad = _np['juv_ll'], _np['adult_ll']
        scost, snll = 0., 0.
        if spc is not None:
            _Ns = (trace['Na_grid']['value'] + trace['Nj_grid']['value'])[:, spc['rows'], spc['cols']] * spc['scale']
            _bt = jax.nn.sigmoid(xsp[2])
            if a.specimen_counts:
                snll = spc['SL'].specimen_nll_counts(_Ns, spc, xsp[:2], _bt, xsp[3])
                scost = snll + .5 * (xsp[3] / 2.) ** 2
            else:
                snll = spc['SL'].specimen_nll(_Ns, spc, xsp[:2], _bt)
                scost = snll
            scost = scost + .5 * jnp.sum((xsp[:2] / 3.) ** 2) - (jnp.log(_bt) + jnp.log1p(-_bt))   # log s ~ N(0,3); beta ~ Beta(2,2) (MAP: no Jacobian)
            if a.hmc:   # HMC samples the logit; run_hmc adds Jacobians for the model coordinates only, so add this block's
                scost = scost - (jnp.log(_bt) + jnp.log1p(-_bt))   # logit Jacobian beta(1-beta) here (verifier 2026-11-08: target was Uniform(0,1) on beta)
        nvcost = 0. if nviab is None else native_viable_cost(trace)
        obj = -(lp - obs_model) + a.bbs_weight * nll_r - prior_sig + 0.5 * m ** 2 + pcost + acost + rpcost + ecost + ccost + wcost + tcost + ncost + scost + nvcost
        return obj, dict(trough_cost=jnp.asarray(tcost, float), trough_D=jnp.asarray(tD, float), specimen_nll=jnp.asarray(snll, float),
                         native_viable_cost=jnp.asarray(nvcost, float),
                         route_skew=jnp.asarray(0. if rsk is None else rsk, float),
                         natal_cost=jnp.asarray(ncost, float), natal_juv_ll=jnp.asarray(njuv, float), natal_adult_ll=jnp.asarray(nad, float),cbc_rate_nll=jnp.asarray(cnll, float), cbc_count_nll=jnp.asarray(cc_nll, float), bbl_nll=jnp.asarray(bnll, float), depression_cost=jnp.asarray(dcost, float),
                         depression_y1=jnp.exp(dlr[0]), depression_y3=jnp.exp(dlr[2]), depression_y5=jnp.exp(dlr[4]),
                         cbc_s=jnp.asarray(0. if cb is None else cb[0], float), cbc_p=jnp.asarray(0. if cb is None else cb[1], float),
                         cbc_logphi=jnp.asarray(0. if cb is None else cb[2], float),
marginal_obs=nll_r, conditional_obs=cond, nest=-nest, nonfinite_rows=nbad, e_absmax=emax, obs_model_eps0=-obs_model, sigma=sg, route_mean=jnp.asarray(m, float), presence_cost=jnp.asarray(pcost, float), absence_cost=jnp.asarray(acost, float), range_presence_cost=jnp.asarray(rpcost, float), establishment_cost=jnp.asarray(ecost, float)), e

    _vg = jax.jit(jax.value_and_grad(lambda x, e0: parts(x, e0)[0]))
    _info = jax.jit(lambda x, e0: parts(x, e0)[1:])
    _E = {'e': jnp.zeros(n_route)}          # inner-solve start; stays 0 unless --laplace_warm
    if a.laplace_warm and _eps_saved is not None and _eps_saved.size == n_route:   # start from the init's own route effects
        _E['e'] = jnp.asarray((_eps_saved * sig_cfg - _m_saved) / sigma0)
        print(json.dumps(dict(stage='laplace_warm_init', e_absmax=float(np.max(np.abs(np.asarray(_E['e'])))))), flush=True)
    vg = lambda x: _vg(x, _E['e'])
    info = lambda x: _info(x, _E['e'])

    def _resume(key, n):   # a saved nuisance block from --init, if present with the right size
        try:
            with np.load(a.init) as _z:
                return np.asarray(_z[key], np.float64).ravel() if (key in _z.files and _z[key].size == n) else None
        except Exception:
            return None
    x0 = np.asarray(coords.x0, np.float64)
    if not a.fix_sigma:
        x0 = np.append(x0, np.log(sigma0))
    if a.route_mean:
        x0 = np.append(x0, 0.)
    if crate is not None:   # s initialised at log(sum y / sum A) at the init state; p .6; log phi 0
        _xx = x0[:-1] if a.route_mean else x0
        _xs = _xx if a.fix_sigma else _xx[:-1]
        _, _tr = log_density(array_model, (), {'arrays': arrays}, coords.constrain(jnp.asarray(_xs)))
        _Nl = (np.asarray(_tr['Na_grid']['value']) + np.asarray(_tr['Nj_grid']['value']))[:, np.asarray(crate['rows']), np.asarray(crate['cols'])] * crate['scale']
        _A = (_Nl[np.asarray(crate['t'])[:, None], np.asarray(crate['src'])[np.asarray(crate['c'])]] * np.asarray(crate['w'])[np.asarray(crate['c'])]).sum(1)
        s0 = float(np.log(max(float(np.asarray(crate['y']).sum()), 1.) / max(float(_A.sum()), 1e-12)))
        x0 = np.append(x0, [s0, .6, 0.])
        print(json.dumps(dict(stage='cbc_rate_init', s0=s0, A_sum=float(_A.sum()))), flush=True)
    if bbl is not None:   # resumes the saved BBL reporting parameters when the init carries them (2026-11-07 fix)
        _rb = _resume('laplace_bbl', 12)
        x0 = np.append(x0, _rb if _rb is not None else np.full(12, np.log(.005 / .995)))
        print(json.dumps(dict(stage='bbl_init', resumed=_rb is not None)), flush=True)
    if ccnt is not None:   # s at log(sum y / sum A e^{.6 le}) at the init state; p .6; log phi 0
        _xx = x0[:-12] if bbl is not None else x0
        _xx = _xx[:-3] if crate is not None else _xx
        _xx = _xx[:-1] if a.route_mean else _xx
        _xs = _xx if a.fix_sigma else _xx[:-1]
        _, _tr = log_density(array_model, (), {'arrays': arrays}, coords.constrain(jnp.asarray(_xs)))
        _Nl = (np.asarray(_tr['Na_grid']['value']) + np.asarray(_tr['Nj_grid']['value']))[:, np.asarray(ccnt['rows']), np.asarray(ccnt['cols'])] * ccnt['scale']
        _A = (_Nl[np.asarray(ccnt['t'])[:, None], np.asarray(ccnt['src'])] * np.asarray(ccnt['w'])).sum(1) * np.exp(.6 * np.asarray(ccnt['le']))
        s0 = float(np.log(max(float(np.asarray(ccnt['y']).sum()), 1.) / max(float(_A.sum()), 1e-12)))
        _rc = _resume('laplace_cbc_counts', ccnt['npar'])   # resume the saved CBC count parameters (2026-11-07 fix)
        if _rc is None and ccnt['npar'] == 4:               # 3-parameter init + new drift slot: resume and start gam at 0
            _rc = _resume('laplace_cbc_counts', 3); _rc = None if _rc is None else np.append(_rc, 0.)
        x0 = np.append(x0, _rc if _rc is not None else [s0, .6, 0.] + ([0.] if ccnt['npar'] == 4 else []))
        print(json.dumps(dict(stage='cbc_counts_init', s0=s0, resumed=_rc is not None)), flush=True)
    if ywg is not None:   # z = 0, s = the half-normal scale; resumes from a saved wiggle if the init carries one
        _xw = np.append(np.zeros(3 * ywg['nY']), np.log(a.year_wiggle))
        try:
            with np.load(a.init) as _zi:
                if 'laplace_year_wiggle_x' in _zi.files and _zi['laplace_year_wiggle_x'].size == ywg['n']:
                    _xw = np.asarray(_zi['laplace_year_wiggle_x'], np.float64)
        except Exception:
            pass
        x0 = np.append(x0, _xw)
    if nat is not None:   # [gamma_raw, g_adult]; gamma = softplus(gamma_raw) init .3; resumes a saved block
        _rn = _resume('laplace_natal', 2)
        x0 = np.append(x0, _rn if _rn is not None else [float(np.log(np.expm1(.3))), 0.])
        _nc = int(np.asarray(coords.x0).size)
        _, _trn = log_density(array_model, (), {'arrays': arrays}, coords.constrain(jnp.asarray(x0[:_nc])))
        nat['adult_pa'] = jnp.asarray(nat['NLK'].adult_departure_constants(_trn, nat['cache'], nat['T'], nat['sel'], t_offset=a.natal_t_offset))
        print(json.dumps(dict(stage='natal_init', resumed=_rn is not None, adult_p_fixed_mean=float(np.mean(np.asarray(nat['adult_pa'])))), ), flush=True)
    if spc is not None:   # [log s0, log s1, logit beta] at the agent's profiled values (s ~ .02, beta ~ .75); resumes a saved block
        if a.specimen_counts:
            _rs = _resume('laplace_specimens', 4)
            if _rs is None:
                _r3 = _resume('laplace_specimens', 3); _rs = np.append(_r3 if _r3 is not None else [-3.9, -3.8, 1.1], 0.)
        else:
            _rs = _resume('laplace_specimens', 3)
            if _rs is None:
                _r4 = _resume('laplace_specimens', 4); _rs = _r4[:3] if _r4 is not None else None
        x0 = np.append(x0, _rs if _rs is not None else [-3.9, -3.8, 1.1])
        print(json.dumps(dict(stage='specimens_init', resumed=_rs is not None)), flush=True)
    if a.route_skew:   # skew-normal shape alpha (0 = normal); resumes a saved value
        _rk = _resume('laplace_route_skew', 1)
        x0 = np.append(x0, _rk if _rk is not None else [0.])
        print(json.dumps(dict(stage='route_skew_init', resumed=_rk is not None)), flush=True)
    if a.grad_sites:   # which coordinates carry the gradient at --init (kink diagnostic), plus one-sided step checks
        g = np.asarray(jax.grad(lambda x: parts(x)[0])(jnp.asarray(x0)))
        names = list(coords.names) + ([] if a.fix_sigma else ['log_sigma']) + (['route_mean'] if a.route_mean else [])
        sizes = [int(np.asarray(coords.initial[k]).size) for k in coords.names] + [1] * (len(names) - len(coords.names))
        bnd = np.cumsum([0] + sizes); f0v = float(parts(jnp.asarray(x0))[0])
        _nf = [(names[i], int((~np.isfinite(g[bnd[i]:bnd[i + 1]])).sum())) for i in range(len(names)) if not np.isfinite(g[bnd[i]:bnd[i + 1]]).all()]
        print(json.dumps(dict(stage='nonfinite_grad_sites', sites=_nf, n_total=int((~np.isfinite(g)).sum()), n_x=int(g.size), n_named=int(bnd[-1]))), flush=True)
        order = np.argsort([-np.abs(g[bnd[i]:bnd[i + 1]]).max() for i in range(len(names))])
        _k0 = bnd[order[0]] + int(np.argmax(np.abs(g[bnd[order[0]]:bnd[order[0] + 1]])))   # objective parts at x and x - 1e-6 e_k
        _e0 = np.zeros_like(x0); _e0[_k0] = 1e-6
        _i0 = info(jnp.asarray(x0)); _i1 = info(jnp.asarray(x0 - _e0))
        _d0 = {k: float(v) for k, v in _i0[0].items()}; _d1 = {k: float(v) for k, v in _i1[0].items()}
        _de = np.abs(np.asarray(_i1[1]) - np.asarray(_i0[1])); _top = np.argsort(-_de)[:5]
        print(json.dumps(dict(stage='route_jump', routes=[int(r) for r in _top], de=[round(float(_de[r]), 4) for r in _top],
                              e0=[round(float(np.asarray(_i0[1])[r]), 3) for r in _top], n_rows=[int(np.sum(np.asarray(idc)[np.asarray(on)] == r)) for r in _top])), flush=True)
        print(json.dumps(dict(stage='parts_jump', coord=int(_k0), f0=f0v, f_back=float(parts(jnp.asarray(x0 - _e0))[0]),
                              diff={k: _d1[k] - _d0[k] for k in _d0 if abs(_d1[k] - _d0[k]) > 1e-3})), flush=True)
        rows = []
        for i in order[:10]:
            gi = g[bnd[i]:bnd[i + 1]]; j = int(np.argmax(np.abs(gi))); k = bnd[i] + j
            fd = {}
            for h in (1e-6, 1e-4):
                e = np.zeros_like(x0); e[k] = h
                fd[str(h)] = [round((float(parts(jnp.asarray(x0 + e))[0]) - f0v) / h, 2), round((f0v - float(parts(jnp.asarray(x0 - e))[0])) / h, 2)]
            rows.append(dict(site=names[i], size=sizes[i], grad=float(gi[j]), at=j, fd_fwd_back=fd))
            print(json.dumps(rows[-1]), flush=True)
        (a.out / 'grad_sites.json').write_text(json.dumps(dict(objective=f0v, rows=rows), indent=1) + '\n')
        return
    if a.agh:   # LAPLACE ACCURACY (review item 5): per-route adaptive Gauss-Hermite vs Laplace at --init
        xj = jnp.asarray(x0)
        xx, m = (xj[:-1], xj[-1]) if a.route_mean else (xj, 0.)
        xs, lsg = (xx, jnp.log(sigma0)) if a.fix_sigma else (xx[:-1], xx[-1])
        sg = jnp.exp(lsg)
        lp, trace = log_density(array_model, (), {'arrays': arrays}, coords.constrain(xs))
        mu0 = jnp.where(trn, trace['expected_obs']['value'], 1.)
        kap = trace['obs_conc_kappa']['value'] if kappa_on else 0.
        phib = jnp.where(trn, trace['obs_concentration']['value'], 1.) / (jnp.power(jnp.maximum(mu0, 1e-3), kap) if kappa_on else 1.)
        _, e, H, _, _, _ = laplace(mu0, phib, kap, sg, m)
        seg = lambda v: jax.ops.segment_sum(jnp.where(on, v, 0.), idc, num_segments=n_route)
        fr = lambda ee: seg(w * row_ll(jnp.where(on, m + sg * ee[idc], 0.), mu0, phib, kap))   # per-route log-lik
        nrow = np.asarray(seg(w * 1.))
        Hp = H + 1.; s = 1. / jnp.sqrt(Hp)
        lap = fr(e) - .5 * e ** 2 - .5 * jnp.log(Hp)                         # log marginal per route (Laplace)
        out = dict(init=str(a.init), sigma=float(sg), n_route_with_rows=int((nrow > 0).sum()))
        for K in (a.agh, 2 * a.agh):
            xk, wk = np.polynomial.hermite_e.hermegauss(K)                    # weight exp(-x^2/2); sum wk = sqrt(2 pi)
            terms = jnp.stack([fr(e + s * float(xi)) - .5 * (e + s * float(xi)) ** 2 + .5 * float(xi) ** 2 + np.log(wi)
                               for xi, wi in zip(xk, wk)])
            agh = jnp.log(s) - .5 * np.log(2 * np.pi) + jax.scipy.special.logsumexp(terms, axis=0)
            d = np.asarray(agh - lap)[nrow > 0]
            out[f'K{K}'] = dict(total_agh_minus_laplace=float(d.sum()), abs_q50_q90_q99_max=[float(v) for v in np.quantile(np.abs(d), [.5, .9, .99, 1.])],
                                n_abs_gt_0p1=int((np.abs(d) > .1).sum()))
            dd = np.asarray(agh - lap)
        # where are the biggest errors: by rows per route and by share of zeros
        yz = np.asarray(seg(w * (y == 0)))
        big = np.argsort(-np.abs(dd))[:10]
        out['worst'] = [dict(route=int(i), diff=float(dd[i]), n_rows=float(nrow[i]), zero_share=float(yz[i] / max(nrow[i], 1)),
                             e_star=float(e[i]), H=float(H[i])) for i in big]
        for lo, hi in ((1, 3), (3, 10), (10, 30), (30, 1e9)):
            mk = (nrow >= lo) & (nrow < hi)
            out[f'rows_{lo}_{int(min(hi, 999))}'] = dict(n=int(mk.sum()), sum_diff=float(dd[mk].sum()), mean_abs=float(np.abs(dd[mk]).mean()) if mk.any() else None)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'agh_check.json').write_text(json.dumps(out, indent=1) + '\n')
        print(json.dumps(out, indent=1), flush=True)
        return
    if a.hmc:   # NUTS on the embedded-Laplace marginal posterior (route effects integrated out per route)
        run_hmc(a, parts, coords, x0, jax, jnp, info, sig_cfg, t0)
        return
    if a.hessian_only:   # the whole posterior curvature at this point: 146 coordinates, so the dense Hessian is cheap
        names = list(coords.names) + ([] if a.fix_sigma else ['log_sigma']) + (['route_mean'] if a.route_mean else [])
        sizes = [int(np.asarray(coords.initial[k]).size) for k in coords.names] + ([] if a.fix_sigma else [1]) + ([1] if a.route_mean else [])
        bounds = np.cumsum([0] + sizes)
        obj = lambda x: parts(x)[0]
        gfun = jax.grad(obj)                                    # column-by-column HVPs: a vmapped hessian needs ~8 GB
        hvp1 = jax.jit(lambda x, v: jax.jvp(gfun, (x,), (v,))[1])
        xj = jnp.asarray(x0); I = np.eye(x0.size)
        Hm = np.stack([np.asarray(hvp1(xj, jnp.asarray(I[i])), np.float64) for i in range(x0.size)], 1); Hm = .5 * (Hm + Hm.T)
        g = np.asarray(jax.grad(obj)(jnp.asarray(x0)))
        ev, V = np.linalg.eigh(Hm)
        def shares(v):
            sh = [(names[i], float((v[bounds[i]:bounds[i + 1]] ** 2).sum())) for i in range(len(names))]
            return [(k, round(x, 3)) for k, x in sorted(sh, key=lambda t: -t[1])[:5] if x > .02]
        diag = {names[i]: float(np.diag(Hm)[bounds[i]:bounds[i + 1]].max()) for i in range(len(names))}
        out = dict(init=str(a.init), n=int(x0.size), grad_max=float(np.abs(g).max()), n_negative=int((ev < 0).sum()),
                   eig_max=float(ev[-1]), eig_min=float(ev[0]), condition=float(ev[-1] / max(ev[0], 1e-300)) if ev[0] > 0 else None,
                   eig_all=[float(x) for x in ev],
                   top=[dict(eig=float(ev[-1 - i]), sites=shares(V[:, -1 - i])) for i in range(6)],
                   bottom=[dict(eig=float(ev[i]), sites=shares(V[:, i])) for i in range(8)],
                   site_max_diag=dict(sorted(diag.items(), key=lambda t: -t[1])))
        (a.out / 'hessian.json').write_text(json.dumps(out, indent=1) + chr(10))
        np.savez_compressed(a.out / 'hessian.npz', H=Hm, eig=ev, V=V, names=np.asarray(names), bounds=bounds)
        print(json.dumps({k: out[k] for k in ('n', 'grad_max', 'n_negative', 'eig_max', 'eig_min', 'condition')}), flush=True)
        return
    trace_f = (a.out / 'trace.jsonl').open('x')
    best = {'objective': float('inf')}
    n = {'f': 0, 'it': 0}

    def save(x):
        d, e = info(jnp.asarray(x)); sg = float(d['sigma'])
        xrk = None
        if a.route_skew:
            x, xrk = x[:-1], np.asarray(x[-1:])
        xsp = None
        if spc is not None:
            _nsp = 4 if a.specimen_counts else 3
            x, xsp = x[:-_nsp], np.asarray(x[-_nsp:])
        xn = None
        if nat is not None:
            x, xn = x[:-2], np.asarray(x[-2:])
        xw = None
        if ywg is not None:
            x, xw = x[:-ywg['n']], np.asarray(x[-ywg['n']:])
        vals_cc = None
        if ccnt is not None:
            vals_cc = np.asarray(x[-ccnt['npar']:]); x = x[:-ccnt['npar']]
        vals_bbl = np.asarray(x[-12:]) if bbl is not None else None
        x = x[:-12] if bbl is not None else x
        x = x[:-3] if crate is not None else x
        xs = x[:-1] if a.route_mean else x
        xs = xs if a.fix_sigma else xs[:-1]
        vals = {k: np.asarray(v) for k, v in coords.constrain(jnp.asarray(xs)).items()}
        vals['observer_route_eps'] = (np.asarray(e) * sg + float(d['route_mean'])) / sig_cfg
        vals['laplace_route_mean'] = np.asarray(float(d['route_mean']))
        vals['laplace_route_sigma'] = np.asarray(sg)
        if vals_cc is not None:
            vals['laplace_cbc_counts'] = vals_cc
        if vals_bbl is not None:
            vals['laplace_bbl'] = vals_bbl
        if xn is not None:
            vals['laplace_natal'] = xn
        if xsp is not None:
            vals['laplace_specimens'] = xsp
        if xrk is not None:
            vals['laplace_route_skew'] = xrk
        if xw is not None:
            vals['laplace_year_wiggle_x'] = xw
            vals['laplace_year_wiggle_E'] = np.asarray(wiggle(jnp.asarray(xw))[0]); vals['laplace_year_wiggle_s'] = np.asarray(np.exp(xw[-1]))
        if crate is not None:
            vals['laplace_route_cbc_rate'] = np.asarray([float(d['cbc_s']), float(d['cbc_p']), float(d['cbc_logphi'])])
        tmp = a.out / 'best_params.tmp.npz'; np.savez_compressed(tmp, **vals); tmp.replace(a.out / 'best_params.npz')
        return d, e

    def fun(x):
        v, g = vg(jnp.asarray(x)); v, g = float(v), np.asarray(g, np.float64)
        if not (np.isfinite(v) and np.isfinite(g).all()):
            return np.inf, np.zeros_like(x)
        n['f'] += 1
        if v < best['objective']:
            best.update(objective=v, x=np.array(x), gnorm=float(np.linalg.norm(g)), gmax=float(np.abs(g).max()))
        return v, g

    def callback(xk, *args):
        n['it'] += 1
        if a.laplace_warm:      # next iteration's inner solve starts from this accepted point's route effects
            _E['e'] = jnp.asarray(info(jnp.asarray(best['x']))[1])
        if n['it'] % 10 and n['it'] > 3:
            return
        d, e = save(best['x'])
        row = dict(iteration=n['it'], objective=best['objective'], **{k: float(v) for k, v in d.items()},
                   route_sd=float(np.std(np.asarray(e)) * float(d['sigma'])), grad_l2=best.get('gnorm'), grad_max=best.get('gmax'),
                   f_evals=n['f'], sec=round(time.time() - t0, 1))
        trace_f.write(json.dumps(row) + '\n'); trace_f.flush(); print(json.dumps(row), flush=True)

    if a.perturb > 0:
        _nc = int(np.asarray(coords.x0).size)
        x0 = np.array(x0, np.float64); x0[:_nc] += a.perturb * np.random.default_rng(a.seed).standard_normal(_nc)
        print(json.dumps(dict(stage='perturb', sd=a.perturb, n=_nc, seed=a.seed)), flush=True)
    d0, _ = info(jnp.asarray(x0))
    print(json.dumps(dict(stage='info0', **{k: float(v) for k, v in d0.items()})), flush=True)
    f0, _ = fun(x0)
    print(json.dumps(dict(stage='start', objective=f0, n=int(x0.size), sigma0=sigma0, **{k: float(v) for k, v in d0.items()},
                          sec=round(time.time() - t0, 1))), flush=True)
    res = minimize(fun, x0, jac=True, method='L-BFGS-B', callback=callback,
                   options=dict(maxiter=a.maxiter, maxcor=a.maxcor, maxls=40, ftol=1e-12, gtol=1e-4))
    d, e = save(best['x'])
    out = dict(objective=best['objective'], **{k: float(v) for k, v in d.items()}, route_sd=float(np.std(np.asarray(e)) * float(d['sigma'])),
               gradient_l2=best['gnorm'], gradient_max=best['gmax'], iterations=n['it'], f_evals=n['f'], message=str(res.message),
               success=bool(res.success), start_objective=f0, init=str(a.init), config=os.environ.get('AGE_MODEL_CONFIG'),
               fixed_sites=fixed, fix_sigma=a.fix_sigma, sigma0=sigma0, seconds=round(time.time() - t0))
    (a.out / 'result.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
