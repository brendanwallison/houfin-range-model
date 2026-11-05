"""C2c: partial refit of the HABITAT-READOUT sites only -- single shared field vs exchangeable (separate correlated)
fields -- on the full model log posterior (priors included), everything else held at the fitted (refined) point.

Run under the base overlay (--form single: fixed_coupling 1, as fitted) or under SBX_OVERLAY =
specs/overlay_step43_exchangeable.json (--form exchangeable: rho ~ logit-normal(.85, .15), private parts eps (M, 4)).
For exchangeable the new sites start at an EXACT reproduction of the single-field fit: eps = 0, coupling logit =
logit(.85), manifold_factor / sqrt(.85) (so w_env is unchanged); checked by the log density.
Refitted sites: manifold_factor, [manifold_idio, manifold_coupling_logit], w_scale_raw, alpha_a, alpha_j, alpha_f,
alpha_k. Objective: minus numpyro log_density (full model, priors + all observations). L-BFGS-B with exact JAX
gradients (GPU float64). Readouts at the refitted point: log density, BBS log-lik (25-49N rows), offsets, potential
lambda East 1940 / 1966 / 2010 and its Shapley split over Sa / Sj / Fmax, rho.
DIAGNOSTIC ONLY (sandbox): the user chose the single field at ladder step 2; nothing here implies a production change.
Usage (inside sbx_env.sh): python refit_fields.py --form single|exchangeable --out DIR [--maxiter 150]
"""
import argparse, itertools, json, math, os, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density
from scipy.optimize import minimize
from harness import Sandbox, capture, to_land, potential_lambda, ap

OPT_SCALE = 1e4
RHO0 = .85


def lambda_split(sb, lat, data=None, real_data=None, real_rates=(), G=None):
    """Potential lambda Sa + Sj Fmax medians / Shapley split, East 1940 & 1966 -> 2010, plus occupied-cell stats.
    R74 (review): real_rates are taken from the REAL-Z pass (as swap_real does in the forecast), and when G is given
    (the decoupled low-density multiplier, scalar or (T,) East-median series) an extra G-inclusive LOW-DENSITY lambda
    Sa + Sj Fmax G is reported for occupied cells."""
    x, _ = capture(sb.data if data is None else data, lat)
    T, Nx = len(sb.years), int(np.asarray(sb.data['land_mask']).shape[1])
    R = {k: to_land(x[f'{k}_flat'], sb.rows, sb.cols, T, Nx) for k in ('Sa', 'Sj', 'Fmax')}
    if real_rates:
        xr, _ = capture(real_data, lat)
        for k in real_rates:
            R[k] = to_land(xr[f'{k}_flat'], sb.rows, sb.cols, T, Nx)
    yb = bool(x.get('yearling_breeding')); lam = lambda r: potential_lambda(r['Sa'], r['Sj'], r['Fmax'], yb)
    out = {}
    for y0 in (1940, 1966):
        i0, i1 = int(np.flatnonzero(sb.years == y0)[0]), int(np.flatnonzero(sb.years == 2010)[0])
        r0 = {k: v[i0] for k, v in R.items()}; r1 = {k: v[i1] for k, v in R.items()}
        tot = lam(r1) - lam(r0); shap = {k: np.zeros_like(tot) for k in R}
        for order in itertools.permutations(R):
            cur = dict(r0)
            for k in order:
                b = lam(cur); cur[k] = r1[k]; shap[k] += (lam(cur) - b) / 6
        m = sb.region['east']
        out[f'{y0}-2010'] = dict(lambda_y0=float(np.median(lam(r0)[m])), lambda_2010=float(np.median(lam(r1)[m])),
                                 shares={k: float(v[m].mean() / tot[m].mean()) for k, v in shap.items()},
                                 rates={k: [float(np.median(r0[k][m])), float(np.median(r1[k][m]))] for k in R})
        # R65 (reviews 3/5, J22): the same on OCCUPIED cells only (a BBS detection), East and West, with quantiles and
        # the share of occupied cells that are deep sinks (< .5) at y0 and sources (> 1) in 2010
        occ = np.isfinite(sb.first_obs)
        for reg in ('east', 'west'):
            mo = sb.region[reg] & occ; l0, l1 = lam(r0)[mo], lam(r1)[mo]
            out[f'{y0}-2010'][f'occupied_{reg}'] = dict(
                n=int(mo.sum()), q10_50_90_y0=[float(q) for q in np.quantile(l0, [.1, .5, .9])],
                q10_50_90_2010=[float(q) for q in np.quantile(l1, [.1, .5, .9])],
                deep_sink_share_y0=float((l0 < .5).mean()), source_share_2010=float((l1 > 1).mean()))
            if G is not None:
                g0 = float(np.asarray(G).ravel()[i0]) if np.ndim(G) else float(G)
                g1 = float(np.asarray(G).ravel()[i1]) if np.ndim(G) else float(G)
                e0 = r0['Sa'][mo] + r0['Sj'][mo] * r0['Fmax'][mo] * g0; e1 = r1['Sa'][mo] + r1['Sj'][mo] * r1['Fmax'][mo] * g1
                out[f'{y0}-2010'][f'occupied_{reg}'].update(
                    G=[g0, g1], lowdens_q10_50_90_y0=[float(q) for q in np.quantile(e0, [.1, .5, .9])],
                    lowdens_q10_50_90_2010=[float(q) for q in np.quantile(e1, [.1, .5, .9])])
    return out


def record_forward_arg(model_fn, kw, params, key):
    """Run the model's log density once with forward_sim_age_structured wrapped, returning the forward's `key` arg."""
    import inspect
    orig = ap.forward_sim_age_structured; rec = {}
    def wrapped(*aa, **kk):
        b = inspect.signature(orig).bind(*aa, **kk); b.apply_defaults()
        rec['v'] = b.arguments[key] if isinstance(key, str) else {k: b.arguments[k] for k in key}
        return orig(*aa, **kk)
    ap.forward_sim_age_structured = wrapped
    try:
        log_density(model_fn, (), kw, params)
    finally:
        ap.forward_sim_age_structured = orig
    return rec['v']


def edited_label(a):
    """J22: which habitat inputs this run edits, written into every summary."""
    if a.decouple:
        base = (f'Sa/Sj/Fmax read Z_gathered -> {a.field_z}; K real Z; Z_disp real' if a.field_z != 'real'
                else (f'Sa/Sj/Fmax readouts frozen: {a.decouple_keys}; K real Z' if a.vr_alpha == 0
                      else f'Sa/Sj/Fmax read {a.vr_alpha} of the time change in {a.decouple_keys}; K real Z'))
    else:
        base = f'field_alpha {a.field_alpha} on {a.decouple_keys}' if a.field_alpha != 1 else 'none'
        if a.field_z != 'real':
            base = f'Z_gathered -> {a.field_z} (all rates incl. K; Z_disp real)'
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--form', choices=['single', 'exchangeable'], required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--maxiter', type=int, default=150)
    p.add_argument('--estab', action='store_true',
                   help='A3: add a SEPARATE establishment / low-density readout of Z (G = exp(g0 + z . b), fading with N/K '
                        'at rho <= --rho_max; lowdens_forward fork), jointly refitted with the habitat field')
    p.add_argument('--rho_max', type=float, default=.3); p.add_argument('--ridge_sd', type=float, default=.5)
    p.add_argument('--b_scale', type=float, default=.1)
    p.add_argument('--field_alpha', type=float, default=1.,
                   help='HOMOTOPY: the habitat FIELD reads Z_ref + alpha (Z - Z_ref) (1966-2025 mean ref); the establishment '
                        'readout always reads the real Z')
    p.add_argument('--init_npz', type=Path, help='warm start: habitat sites (+ estab vector) from a previous step')
    p.add_argument('--init_r16', type=Path, help='warm start for the establishment vector from a zreadout summary')
    p.add_argument('--tag', default='')
    p.add_argument('--survanchor', action='store_true',
                   help='C2: low-density adult survival anchored at a fitted habitat-independent hi (survanchor_forward), '
                        'jointly refitted with the habitat sites; excludes --estab')
    p.add_argument('--sa_q', type=float, default=1.)
    p.add_argument('--floor', action='store_true',
                   help='B9c: a FITTED departure floor (dep_forward, all cells; logit start --floor0) jointly with the habitat sites')
    p.add_argument('--floor0', type=float, default=.1)
    p.add_argument('--decouple', action='store_true',
                   help='A9: capacity K reads the REAL Z (a second model pass under the same parameters records it), '
                        'Sa / Sj / Fmax read FROZEN Z (field_alpha 0 data); use with --estab for front timing')
    p.add_argument('--sa_hi_fixed', type=float, default=None,
                   help='C2 fixed anchor: hi held at this value (not refitted), so the fit cannot opt out')
    p.add_argument('--tight', action='store_true', help='no relative-reduction / gradient early stop (R30 continuations)')
    p.add_argument('--estab_z', choices=['real', 'static', 'detrended'], default='real',
                   help='R47: the establishment readout reads real Z, its time mean, or Z minus the continental yearly mean')
    p.add_argument('--field_z', choices=['real', 'detrended', 'detrended_west', 'detrended_region', 'linclock', 'scrambled', 'linregion', 'linregion93', 'swap_ew'], default='real',
                   help='R52/R55: habitat field reads Z; Z minus the 25-49N yearly mean; minus the WEST yearly mean; minus each '
                        'region own yearly mean; detrended + a LINEAR band clock (placebo: a trend continues); or Z with years '
                        'permuted in 5-yr blocks (spatial structure kept; placebo)')
    p.add_argument('--real_rates', default='', help='R69: with --decouple, these vital rates (Sa,Sj,Fmax) also read the REAL-Z pass')
    p.add_argument('--vr_alpha', type=float, default=0., help='R65: with --decouple, Sa/Sj/Fmax read Zref + vr_alpha (Z - Zref) (K real)')
    p.add_argument('--fixed_npz', default=None, help='R66: npz whose --fixed_keys latents are held fixed (not refitted)')
    p.add_argument('--fixed_keys', default='observer_route_eps,observer_first_year_eta')
    p.add_argument('--vr_alpha_late', type=float, default=None, help='R82: with --decouple, scale the vital-rate Z change after --late_from')
    p.add_argument('--late_from', type=int, default=1993)
    p.add_argument('--dd_juv', action='store_true', help='Plan L A5 / S2: density-dependent juvenile survival 1 / (1 + cs (N/K)^p), c re-solved')
    p.add_argument('--dd_emergent', action='store_true', help='do NOT re-solve c: the equilibrium emerges below K (K = resource scale)')
    p.add_argument('--dd_only', action='store_true', help='no fecundity crowding (c = 0): equilibrium N* = K/cs (Sj F/(1-Sa) - 1) emerges from the rates')
    p.add_argument('--dd_cs0', type=float, default=1.); p.add_argument('--dd_p', type=float, default=1.)
    p.add_argument('--bbl_until', type=int, default=1993, help='BBL intervals / recoveries after this year are censored (pre-disease, leak-free with --train_until)')
    p.add_argument('--late_prior_cell', type=float, default=None, help='R103b: SD of an independent N(0, SD) prior on EACH occupied cell 1993 -> 2010 change of logit Sa, logit Sj, log Fmax (decoupled + estab only)')
    p.add_argument('--late_prior', type=float, default=None, help='review 9 C6: SD of a N(0, SD) prior on the occupied-mean 1993 -> 2010 change of logit Sa, logit Sj, log Fmax per region (decoupled + estab only)')
    p.add_argument('--bbl_disease', action='store_true', help='BBL likelihood on DISEASE-INCLUSIVE adult survival Sa (1 - m), m at local density ~ K (use with --bbl_until 2024)')
    p.add_argument('--bbl', default='', help='Plan L C1: BBL adult dead-recovery m-array npz (bbl_prep.py); decoupled + estab only')
    p.add_argument('--sa_anchor', default='', help='R81: MEAN,SD soft anchor on mean adult survival over occupied East cells 1966-93 (decoupled + estab only)')
    p.add_argument('--fix_g0', action='store_true', help='R77: pin the low-density multiplier level at G = 1 (with ridge b ~ 0: no low-density boost or penalty)')
    p.add_argument('--exclude_targets', default='', help='R73: sites NOT refitted (comma list), e.g. manifold_factor,w_scale_raw')
    p.add_argument('--extra_targets', default='', help='R59: extra latent sites refitted too (comma list, e.g. observer_route_eps)')
    p.add_argument('--disease_readout', action='store_true', help='R84-style disease toll / effective lambda / disease-off at the end point (summary key disease)')
    p.add_argument('--rescore', action='store_true', help='R49: skip optimisation; read out the --init_npz parameters')
    p.add_argument('--train_until', type=int, default=None, help='R46: BBS rows after this year are held out of the objective and scored at the end')
    p.add_argument('--decouple_keys', default='Z_gathered,Z_disp_gathered',
                   help='R42: habitat inputs FROZEN / scaled for the readouts (Z_gathered alone keeps Z_disp real)')
    p.add_argument('--estab_gate', choices=['ratio', 'allee'], default='ratio',
                   help='R37 (J19 fix): the establishment readout fades with N/K (ratio, as A9) or with the absolute Allee deficit')
    p.add_argument('--free_n50', type=float, default=None,
                   help='R28: refit the Allee n50 (n50_raw site) jointly, its prior SD widened from the overlay pin to this '
                        'value (same location); the pinned prior term is swapped exactly inside the log posterior')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat0 = {k: np.asarray(v, float) for k, v in sb.lat.items()}
    if a.fixed_npz:   # R66: hold these latents at values from another run (e.g. training-only observer effects)
        _fz = np.load(a.fixed_npz)
        for k in a.fixed_keys.split(','):
            lat0[k] = np.asarray(_fz[k], float)
        sb.lat = {k: np.asarray(v, float) for k, v in lat0.items()}
    if a.free_n50:   # [cur] PINS n50_raw (numpyro condition, age_priors.PINNED); unpin it and start from the pinned value
        from src.model import age_priors as _apri
        n50_pinned = float(_apri.PINNED.pop("n50_raw")); lat0["n50_raw"] = np.asarray(n50_pinned)
    if a.form == 'exchangeable':
        M = lat0['manifold_factor'].shape[0]
        lat0['manifold_factor'] = lat0['manifold_factor'] / math.sqrt(RHO0)
        lat0['manifold_idio'] = np.zeros((M, 4)); lat0['manifold_coupling_logit'] = np.asarray(math.log(RHO0 / (1 - RHO0)))
    targets = ['manifold_factor', 'w_scale_raw', 'alpha_a', 'alpha_j', 'alpha_f', 'alpha_k'] + \
              (['manifold_idio', 'manifold_coupling_logit'] if a.form == 'exchangeable' else []) + \
              (['n50_raw'] if a.free_n50 else []) + [k for k in (a.extra_targets.split(',') if a.extra_targets else []) if k]
    if a.exclude_targets:   # R73: hold these sites at their (fit) values -- e.g. the habitat weights, so vr_alpha cannot be compensated
        targets = [k for k in targets if k not in a.exclude_targets.split(',')]
    shapes = {k: np.shape(lat0[k]) for k in targets}
    sizes = {k: int(np.prod(s)) if s else 1 for k, s in shapes.items()}
    fixed = {k: jnp.asarray(v) for k, v in lat0.items() if k not in targets}
    from harness import edit_habitat
    d_field = sb.data if (a.field_alpha == 1 and not a.decouple) else edit_habitat(
        sb.data, sb.years, a.vr_alpha if a.decouple else a.field_alpha, [1966, 2025], keys=tuple(a.decouple_keys.split(',')))
    real_data = sb.data
    if a.vr_alpha_late is not None and a.decouple:   # R82: the vital rates read the change AFTER --late_from scaled by vr_alpha_late
        Zr = np.asarray(d_field['Z_gathered'], float); yy = np.asarray(sb.years)
        i0 = int(np.flatnonzero(yy == a.late_from)[0]); late = yy > a.late_from
        Zl = Zr.copy(); Zl[late] = Zr[i0][None] + a.vr_alpha_late * (Zr[late] - Zr[i0][None])
        d_field = dict(d_field); d_field['Z_gathered'] = jnp.asarray(Zl, dtype=jnp.asarray(sb.data['Z_gathered']).dtype)
        sb.data = d_field   # readouts use the same late-scaled field (K still from real_data via kw_real)
    if a.field_z != 'real':   # R52 / R55: what time signal in Z the habitat field may read
        band_ = sb.region['east'] | sb.region['plains'] | sb.region['west']
        if a.decouple:   # R53: Sa/Sj/Fmax read detrended REAL Z (spatial change kept, clock removed); K real via kw_real
            d_field = real_data
        Zf = np.asarray(d_field['Z_gathered'], float)
        if a.field_z in ('detrended', 'linclock'):
            mt = Zf[:, band_].mean(1, keepdims=True); Zn = Zf - mt + mt.mean(0, keepdims=True)
            if a.field_z == 'linclock':   # add back a straight-line fit of the band clock (per dim) over the years
                yy = np.asarray(sb.years, float); X = np.column_stack([np.ones_like(yy), yy - yy.mean()])
                coef, *_ = np.linalg.lstsq(X, mt[:, 0, :], rcond=None)
                Zn = Zn + (X @ coef - mt.mean(0)[0])[:, None, :]
        elif a.field_z == 'detrended_west':
            mt = Zf[:, sb.region['west']].mean(1, keepdims=True); Zn = Zf - mt + mt.mean(0, keepdims=True)
        elif a.field_z == 'detrended_region':
            Zn = Zf.copy()
            for r_ in ('east', 'plains', 'west'):
                m_ = sb.region[r_]; mt = Zf[:, m_].mean(1, keepdims=True)
                Zn[:, m_] = Zf[:, m_] - mt + mt.mean(0, keepdims=True)
        elif a.field_z in ('linregion', 'linregion93'):   # R61: each region's yearly mean replaced by its own straight line
            yy = np.asarray(sb.years, float); Zn = Zf.copy()
            fitm = (yy <= 1993) if a.field_z == 'linregion93' else np.ones_like(yy, bool)   # 93: fit on <= 1993, extrapolate
            X = np.column_stack([np.ones_like(yy), yy - yy.mean()])
            for r_ in ('east', 'plains', 'west'):
                m_ = sb.region[r_]; mt = Zf[:, m_].mean(1)
                coef, *_ = np.linalg.lstsq(X[fitm], mt[fitm], rcond=None)
                Zn[:, m_] = Zf[:, m_] - mt[:, None, :] + (X @ coef)[:, None, :]
        elif a.field_z == 'swap_ew':   # R64: East cells get the WEST's yearly-mean trajectory and vice versa (wrong-region placebo)
            Zn = Zf.copy(); mE = sb.region['east']; mW = sb.region['west']
            tE = Zf[:, mE].mean(1, keepdims=True); tW = Zf[:, mW].mean(1, keepdims=True)
            Zn[:, mE] = Zf[:, mE] - tE + tE.mean(0, keepdims=True) + (tW - tW.mean(0, keepdims=True))
            Zn[:, mW] = Zf[:, mW] - tW + tW.mean(0, keepdims=True) + (tE - tE.mean(0, keepdims=True))
        elif a.field_z == 'scrambled':   # permute 5-yr blocks of years (fixed seed); spatial pattern within a year kept
            T_ = Zf.shape[0]; blocks = [np.arange(i, min(i + 5, T_)) for i in range(0, T_, 5)]
            order = np.random.default_rng(20261104).permutation(len(blocks))
            Zn = Zf[np.concatenate([blocks[i] for i in order])]
        d_field = dict(d_field); d_field['Z_gathered'] = jnp.asarray(Zn, dtype=jnp.asarray(sb.data['Z_gathered']).dtype)
        sb.data = d_field   # readouts (sb.run) use the same detrended field
    kw_real = dict(data=real_data, prior_scale=1.)
    kw = dict(data=d_field, prior_scale=1.)

    def unpack(v):
        out, i = {}, 0
        for k in targets:
            out[k] = v[i:i + sizes[k]].reshape(shapes[k]); i += sizes[k]
        return out

    if a.init_npz:   # warm start of the habitat sites from a previous homotopy step
        prev = np.load(a.init_npz)
        for k in targets:
            if k in prev.files:
                lat0[k] = np.asarray(prev[k], float)
    v0 = np.concatenate([np.ravel(lat0[k]) for k in targets])

    if a.free_n50:
        from src.model import age_priors as _apri
        n50_loc = float(_apri._ALLEE_PRIOR["n50_raw_loc"]); n50_sd0 = float(_apri._ALLEE_PRIOR["n50_raw_scale"])

    held = None
    if a.train_until:   # R46 held-out: observation rows after this year leave the objective (NestWatch stays in)
        held = jnp.asarray(np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])] > a.train_until)

    def neg_logp(v):
        params = dict(fixed); params.update(unpack(v))
        lp, tr_ld = log_density(ap.build_model_2d, (), kw, params)
        if held is not None:
            s = tr_ld['obs']
            lpo = s['fn'].log_prob(s['value']) * (s['scale'] if s.get('scale') is not None else 1.)
            lp = lp - jnp.sum(jnp.where(held, lpo, 0.))
        if a.free_n50:   # replace the model prior N(loc, sd0) on the unpinned site by N(pinned value, free_n50)
            x50 = params["n50_raw"]
            lp = lp + 0.5 * ((x50 - n50_loc) / n50_sd0) ** 2 - 0.5 * ((x50 - n50_pinned) / a.free_n50) ** 2
        return -lp

    real_keys = ['K_flat'] + [f'{r}_flat' for r in (a.real_rates.split(',') if a.real_rates else [])]
    SA_STASH = {}
    sa_t = np.flatnonzero((np.asarray(sb.years) >= 1966) & (np.asarray(sb.years) <= 1993))
    sa_c = np.flatnonzero(sb.region['east'] & np.isfinite(sb.first_obs))
    lp_t0 = int(np.flatnonzero(np.asarray(sb.years) == 1993)[0]); lp_t1 = int(np.flatnonzero(np.asarray(sb.years) == 2010)[0])
    lp_c = {r: jnp.asarray(np.flatnonzero(sb.region[r] & np.isfinite(sb.first_obs))) for r in ('east', 'plains', 'west')}
    def swap_real(b, rec):   # R69: K (and any --real_rates) from the real-Z pass; crowding c re-derived from the mixed rates
        for k in real_keys:
            b.arguments[k] = rec[k]
        if a.real_rates:
            from harness import c_of
            b.arguments['c_flat'] = c_of(b.arguments['Fmax_flat'], b.arguments['Sa_flat'], b.arguments['Sj_flat'],
                                         {'yearling_breeding': b.arguments.get('yearling_breeding')})
        if a.bbl:   # Plan L C1: the (T, N_land) adult survival the forecast uses
            Sa__ = jnp.reshape(b.arguments['Sa_flat'], (len(sb.years), -1))
            SA_STASH['sa'] = Sa__ if Sa__.shape[1] == len(sb.rows) else Sa__[:, jnp.asarray(sb.rows * sb.Nx + sb.cols)]
            if a.bbl_disease and b.arguments.get('disease_m_amp') is not None:   # disease-inclusive Sa, local density ~ K
                _land = lambda v: (lambda r: r if r.shape[1] == len(sb.rows) else r[:, jnp.asarray(sb.rows * sb.Nx + sb.cols)])(
                    jnp.reshape(v, (len(sb.years), -1)))
                _K = _land(b.arguments['K_flat']); _hn = b.arguments['disease_hill_n']
                _h = jnp.power(jnp.maximum(_K, 0.) + 1e-12, _hn)
                SA_STASH['sa'] = SA_STASH['sa'] * (1. - _land(b.arguments['disease_m_amp']) * _h / (_h + b.arguments['disease_k_half'] ** _hn))
        if a.sa_anchor:   # R81: stash mean adult survival over occupied East cells 1966-93 for the BBL anchor penalty
            Sa_ = jnp.reshape(b.arguments['Sa_flat'], (len(sb.years), -1))
            Sa_ = Sa_ if Sa_.shape[1] == len(sb.rows) else Sa_[:, jnp.asarray(sb.rows * sb.Nx + sb.cols)]
            SA_STASH['v'] = jnp.mean(Sa_[jnp.asarray(sa_t)][:, jnp.asarray(sa_c)])
        if a.late_prior:   # review 9 C6: soft B1 prior on the post-invasion rate change (occupied means, per region, 1993 -> 2010)
            _land = lambda v: (lambda r: r if r.shape[1] == len(sb.rows) else r[:, jnp.asarray(sb.rows * sb.Nx + sb.cols)])(
                jnp.reshape(v, (len(sb.years), -1)))
            _tr = {'Sa_flat': lambda x: jnp.log(x) - jnp.log1p(-x), 'Sj_flat': lambda x: jnp.log(x) - jnp.log1p(-x), 'Fmax_flat': jnp.log}
            SA_STASH['late'] = jnp.stack([jnp.mean(_tr[k](jnp.clip(_land(b.arguments[k]), 1e-6, 1 - 1e-6) if k != 'Fmax_flat' else jnp.maximum(_land(b.arguments[k]), 1e-6))[lp_t1][lp_c[r]])
                                          - jnp.mean(_tr[k](jnp.clip(_land(b.arguments[k]), 1e-6, 1 - 1e-6) if k != 'Fmax_flat' else jnp.maximum(_land(b.arguments[k]), 1e-6))[lp_t0][lp_c[r]])
                                          for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat') for r in ('east', 'plains', 'west')])
        if a.late_prior_cell:   # R103b: the same change PER occupied cell (a regional-mean prior escapes by redistribution)
            _land = lambda v: (lambda r: r if r.shape[1] == len(sb.rows) else r[:, jnp.asarray(sb.rows * sb.Nx + sb.cols)])(
                jnp.reshape(v, (len(sb.years), -1)))
            _occ = jnp.asarray(np.flatnonzero(np.isfinite(sb.first_obs)))
            _d = []
            for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'):
                X = _land(b.arguments[k])
                X = jnp.log(jnp.maximum(X, 1e-6)) if k == 'Fmax_flat' else (lambda c: jnp.log(c) - jnp.log1p(-c))(jnp.clip(X, 1e-6, 1 - 1e-6))
                _d.append(X[lp_t1][_occ] - X[lp_t0][_occ])
            SA_STASH['late_cell'] = jnp.concatenate(_d)

    if a.dd_juv:   # Plan L A5 / S2: density-dependent juvenile survival, one extra parameter log(cs)
        import ddsurv_forward as ddf
        fwd_dd = ddf.build()
        n_hab = v0.size
        v0 = np.concatenate([v0, [math.log(a.dd_cs0)]])
        def neg_logp_dd(v):
            ddf.DS.update(active=True, cs=jnp.exp(v[n_hab]), p=a.dd_p, resolve=not (a.dd_emergent or a.dd_only), dd_only=a.dd_only)
            orig = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd_dd
            try:
                val = neg_logp(v[:n_hab])
            finally:
                ap.forward_sim_age_structured = orig; ddf.DS['active'] = False
            return val
        vg = jax.jit(jax.value_and_grad(neg_logp_dd))
    elif a.floor:   # B9c: one extra parameter, logit(floor)
        import dep_forward as dfk
        fwd_fl = dfk.build()
        x_probe, _ = capture(sb.data, sb.lat)
        ones_grid = jnp.ones((len(sb.years),) + tuple(np.asarray(x_probe['land_mask']).shape))
        n_hab = v0.size
        v0 = np.concatenate([v0, [math.log(a.floor0 / (1 - a.floor0))]])
        def neg_logp_fl(v):
            dfk.DEP.update(active=True, grid=ones_grid, floor=jax.nn.sigmoid(v[n_hab]))
            orig = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd_fl
            try:
                val = neg_logp(v[:n_hab])
            finally:
                ap.forward_sim_age_structured = orig; dfk.DEP['active'] = False
            return val
        vg = jax.jit(jax.value_and_grad(neg_logp_fl))
    elif a.survanchor:   # C2: one extra parameter, logit(hi)
        import survanchor_forward as saf
        fwd_sa = saf.build()
        n_hab = v0.size
        v0 = np.concatenate([v0, [math.log(.6 / .4)]])
        def neg_logp_sa(v):
            saf.SA.update(active=True, hi=(a.sa_hi_fixed if a.sa_hi_fixed is not None else jax.nn.sigmoid(v[n_hab])),
                          q=a.sa_q)
            orig = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd_sa
            try:
                val = neg_logp(v[:n_hab])
            finally:
                ap.forward_sim_age_structured = orig; saf.SA['active'] = False
            return val
        vg = jax.jit(jax.value_and_grad(neg_logp_sa))
    elif a.estab:   # establishment readout: extra parameters appended after the habitat sites
        import lowdens_forward as ldf
        fwd_ld = ldf.build()
        Z = np.asarray(sb.data['Z_gathered'], float)
        zmu, zsd = Z.reshape(-1, Z.shape[-1]).mean(0), Z.reshape(-1, Z.shape[-1]).std(0) + 1e-9
        z_j = jnp.asarray((Z - zmu) / zsd); Mz = Z.shape[-1]
        if a.estab_z != 'real':   # R47 (review 2 item 4b): what in Z may G read
            Zs_ = (Z - zmu) / zsd
            if a.estab_z == 'static':        # time-mean Z: no time change at all (dimension-matched control)
                Zs_ = np.broadcast_to(Zs_[(sb.years >= 1966) & (sb.years <= 2025)].mean(0), Zs_.shape).copy()
            elif a.estab_z == 'detrended':   # remove the continental (25-49N band) yearly mean: no uniform clock
                band_ = sb.region['east'] | sb.region['plains'] | sb.region['west']
                Zs_ = Zs_ - Zs_[:, band_].mean(1, keepdims=True)
            z_j = jnp.asarray(Zs_)
        x_probe, _ = capture(sb.data, sb.lat)
        gshape = (len(sb.years),) + tuple(np.asarray(x_probe['land_mask']).shape)
        def est_grid(e):   # e = [g0, u (Mz), log rho]
            lin = (0. * e[0] if a.fix_g0 else e[0]) + jnp.tensordot(z_j, a.b_scale * e[1:1 + Mz], axes=([2], [0]))
            return jnp.ones(gshape).at[:, sb.rows, sb.cols].set(jnp.exp(lin))
        n_hab = v0.size
        e0 = np.r_[0., np.zeros(Mz), math.log(.27)]
        if a.init_npz and 'estab' in np.load(a.init_npz).files:
            e0 = np.asarray(np.load(a.init_npz)['estab'], float)
        elif a.init_r16:
            r16 = json.loads(a.init_r16.read_text())
            e0 = np.r_[r16['intercept'], np.asarray(r16['b']) / a.b_scale, math.log(min(r16['rho'], a.rho_max) * .999)]
        v0 = np.concatenate([v0, e0]); n_e = e0.size
        if a.bbl:   # Plan L C1: BBL adult dead-recovery m-array (bbl_prep.py) + 12 reporting-rate logits (3 regions x 4 eras)
            _bz = np.load(a.bbl); _yrs = np.asarray(sb.years)
            _K = int(_bz['K']); _Y = _bz['by'][:, None] + np.arange(1, _K + 1)[None, :]
            BBL = dict(cell=jnp.asarray(_bz['cell']), yidx=jnp.asarray(np.clip(_Y - _yrs[0], 0, len(_yrs) - 1)),
                       valid=jnp.asarray(_Y <= min(_yrs[-1], a.bbl_until)), n=jnp.asarray(_bz['n']),
                       m=jnp.asarray(np.where(_Y <= a.bbl_until, _bz['m'], 0.)),
                       ridx=jnp.asarray(_bz['grp'][:, None] * 4 + np.digitize(_Y, [1980, 1995, 2006])))
            v0 = np.concatenate([v0, np.full(12, math.log(.005 / .995))])
        def bbl_ll(Sa_land, rlog):   # Brownie / Seber: survive intervals by+1 .. by+k-1, die in by+k, found and reported (r)
            S = jnp.clip(Sa_land[BBL['yidx'], BBL['cell'][:, None]], 1e-6, 1 - 1e-6)
            S = jnp.where(BBL['valid'], S, 1.)
            alive = jnp.exp(jnp.cumsum(jnp.log(S), 1) - jnp.log(S))          # prod of S over earlier intervals
            r = jax.nn.sigmoid(rlog[BBL['ridx']])
            P = jnp.where(BBL['valid'], alive * (1. - S) * r, 0.)
            never = jnp.clip(1. - P.sum(1), 1e-12, 1.)
            return jnp.sum(BBL['m'] * jnp.log(jnp.maximum(P, 1e-300))) + jnp.sum((BBL['n'] - BBL['m'].sum(1)) * jnp.log(never))
        def neg_logp_e(v):
            e = v[n_hab:n_hab + n_e]
            fwd_use = fwd_ld
            if a.decouple:   # K from the REAL-Z pass under the same habitat parameters
                import inspect as _insp
                params_r = dict(fixed); params_r.update(unpack(v[:n_hab]))
                K_real = record_forward_arg(ap.build_model_2d, kw_real, params_r, real_keys)
                @__import__("functools").wraps(fwd_ld)
                def fwd_use(*aa, **kk):
                    b = _insp.signature(fwd_ld).bind(*aa, **kk); b.apply_defaults(); swap_real(b, K_real)
                    return fwd_ld(*b.args, **b.kwargs)
            ldf.LD.update(active=True, grid=est_grid(e), rho=jnp.minimum(jnp.exp(e[-1]), a.rho_max), gate=a.estab_gate)
            orig = ap.forward_sim_age_structured; ap.forward_sim_age_structured = fwd_use
            try:
                val = neg_logp(v[:n_hab])
            finally:
                ap.forward_sim_age_structured = orig; ldf.LD['active'] = False
            pen = 0.
            if a.sa_anchor:   # BBL dead-recovery anchor (SURVIVAL_BBL_SCOPING s.2.1): East 1960-93 adult survival ~.56
                m_, s_ = (float(x) for x in a.sa_anchor.split(','))
                pen = 0.5 * ((SA_STASH['v'] - m_) / s_) ** 2
            if a.late_prior:   # N(0, late_prior) on each region x rate change 1993 -> 2010 (logit Sa, logit Sj, log Fmax)
                pen = pen + 0.5 * jnp.sum((SA_STASH['late'] / a.late_prior) ** 2)
            if a.late_prior_cell:   # independent N(0, SD) on every occupied cell's 1993 -> 2010 rate change
                pen = pen + 0.5 * jnp.sum((SA_STASH['late_cell'] / a.late_prior_cell) ** 2)
            if a.bbl:
                rlog = v[n_hab + n_e:]
                pen = pen - bbl_ll(SA_STASH['sa'], rlog) + 0.5 * jnp.sum(((rlog - math.log(.005 / .995)) / 1.5) ** 2)
            return val + pen + 0.5 * jnp.sum((a.b_scale * e[1:1 + Mz]) ** 2) / a.ridge_sd ** 2
        vg = jax.jit(jax.value_and_grad(neg_logp_e))
    else:
        vg = jax.jit(jax.value_and_grad(neg_logp))
    # identity: the reparameterised start reproduces the base fit's log density (single: trivially)
    base_lp = None
    if a.form == 'exchangeable':
        pass   # the base log density under the single-field overlay is recorded by the 'single' arm (its start)
    t0 = time.time(); f0, g0 = vg(jnp.asarray(v0)); f0 = float(f0)
    log = (a.out / f'refit_{a.form}{"_estab" if a.estab else ("_survanchor" if a.survanchor else ("_floor" if a.floor else ""))}{"_decouple" if a.decouple else ""}{a.tag}.jsonl').open('w')
    rec = dict(form=a.form, start_neg_logp=f0, n_params=int(v0.size), grad_norm=float(np.linalg.norm(np.asarray(g0))),
               first_eval_sec=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
    n = [0]
    def f(v):
        val, g = vg(jnp.asarray(v)); val, g = float(val), np.asarray(g, float); n[0] += 1
        if not np.isfinite(val) or not np.all(np.isfinite(g)):
            val, g = 1e12, np.zeros_like(v)
        log.write(json.dumps(dict(eval=n[0], neg_logp=val, grad_norm=float(np.linalg.norm(g)), sec=round(time.time() - t0, 1))) + '\n')
        log.flush()
        return val / OPT_SCALE, g / OPT_SCALE
    if a.rescore:   # R49: no optimisation -- read out the warm-start parameters (held-out rows scored by the harness)
        from types import SimpleNamespace
        res = SimpleNamespace(x=np.asarray(v0, float), fun=f0 / OPT_SCALE, success=False, message='RESCORE (no optimisation)')
    else:
        res = minimize(f, v0, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=a.maxiter, **(dict(ftol=1e-15, gtol=1e-12) if a.tight else {})))
    bbl_info = None
    if a.estab and a.bbl:   # eager evaluation at the end point -> concrete stash
        neg_logp_e(jnp.asarray(res.x))
        _rl = jnp.asarray(res.x[n_hab + n_e:]); _sa = np.asarray(SA_STASH['sa'])
        _t = (np.asarray(sb.years) >= 1966) & (np.asarray(sb.years) <= 1993)
        _occ = np.isfinite(sb.first_obs)
        bbl_info = dict(bbl_ll=float(bbl_ll(jnp.asarray(_sa), _rl)), report_rates=np.asarray(jax.nn.sigmoid(_rl)).round(5).tolist(),
                        sa_mean_occupied={r_: {'1966-93': float(_sa[_t][:, sb.region[r_] & _occ].mean()),
                                               '1994-2020': float(_sa[(np.asarray(sb.years) > 1993) & (np.asarray(sb.years) <= 2020)][:, sb.region[r_] & _occ].mean())}
                                          for r_ in ('east', 'plains', 'west')})
    v_hab = res.x[:n_hab] if (a.estab or a.survanchor or a.floor or a.dd_juv) else res.x
    dd_info = None
    if a.dd_juv:   # install the fitted density-dependent juvenile survival for every readout below
        _cs = float(math.exp(float(res.x[n_hab]))); ddf.DS.update(active=True, cs=_cs, p=a.dd_p, resolve=not (a.dd_emergent or a.dd_only), dd_only=a.dd_only)
        ap.forward_sim_age_structured = fwd_dd; dd_info = dict(cs=_cs, p=a.dd_p, emergent=a.dd_emergent, dd_only=a.dd_only, juv_survival_factor_at_K=1. / (1. + _cs))
    lat1 = dict(lat0); lat1.update({k: np.asarray(v) for k, v in unpack(v_hab).items()})
    est_info = None
    if a.floor:   # install the fitted floor for every readout below
        fl_fit = float(1. / (1. + math.exp(-float(res.x[n_hab]))))
        dfk.DEP.update(active=True, grid=ones_grid, floor=fl_fit); ap.forward_sim_age_structured = fwd_fl
        est_info = dict(departure_floor=fl_fit)
    if a.survanchor:   # install the fitted anchor for every readout below
        hi_fit = (a.sa_hi_fixed if a.sa_hi_fixed is not None else float(1. / (1. + math.exp(-float(res.x[n_hab])))))
        saf.SA.update(active=True, hi=hi_fit, q=a.sa_q); ap.forward_sim_age_structured = fwd_sa
        est_info = dict(sa_hi=hi_fit, sa_q=a.sa_q)
    if a.estab:   # install the fitted establishment readout for every readout below
        e = jnp.asarray(res.x[n_hab:n_hab + n_e]); G = est_grid(e); rho_e = float(min(math.exp(float(e[-1])), a.rho_max))
        ldf.LD.update(active=True, grid=G, rho=rho_e, gate=a.estab_gate); ap.forward_sim_age_structured = fwd_ld
        if a.decouple:   # readouts: K from the real-Z pass at the fitted habitat parameters
            import inspect as _insp2
            params_r1 = dict(fixed); params_r1.update({k: jnp.asarray(v) for k, v in unpack(res.x[:n_hab]).items()})
            ap.forward_sim_age_structured = _af_orig_for_record = __import__('src.model.age_forward', fromlist=['x']).forward_sim_age_structured
            K_real1 = record_forward_arg(ap.build_model_2d, kw_real, params_r1, real_keys)
            @__import__("functools").wraps(fwd_ld)
            def fwd_dec(*aa, **kk):
                b = _insp2.signature(fwd_ld).bind(*aa, **kk); b.apply_defaults(); swap_real(b, K_real1)
                return fwd_ld(*b.args, **b.kwargs)
            ap.forward_sim_age_structured = fwd_dec
        Gl = np.asarray(G)[:, sb.rows, sb.cols]
        est_info = dict(rho=rho_e, g0=float(e[0]), b_norm=float(np.linalg.norm(a.b_scale * np.asarray(e[1:-1]))),
                        G_east={str(y): float(np.median(Gl[sb.years == y][0][sb.region['east']])) for y in (1940, 1960, 1980, 2000, 2020)},
                        G_west={str(y): float(np.median(Gl[sb.years == y][0][sb.region['west']])) for y in (1940, 1960, 1980, 2000, 2020)})
    # readouts at the refitted point
    sb.lat = lat1; sb.base_abund = None; sb.base_x = None; sb.base_med = None
    hab = dict(habitat_alpha=a.vr_alpha, habitat_keys=a.decouple_keys.split(',')) if a.decouple else ({} if a.field_alpha == 1 else dict(habitat_alpha=a.field_alpha, habitat_keys=a.decouple_keys.split(',')))
    if a.field_z != 'real' or a.vr_alpha_late is not None:
        hab = {}   # sb.data is already the detrended field
    row = sb.run(dict(name=f'refit_{a.form}', **hab))
    ll = -sum(v['loglik'] for v in row['abundance'].values())
    dis_info = None
    if a.disease_readout:   # under the installed end-point structure (decoupled K, DD fork, ...)
        from r84_disease import disease_stats
        _hab = dict(hab)
        if a.estab and a.decouple:   # the K the forward actually uses (real-Z pass), not the edited field's
            _hab['_k_flat'] = np.asarray(K_real1['K_flat'])
        dis_info = disease_stats(sb, _hab)
    if a.floor:   # the start readout is the base fit WITHOUT the floor
        import src.model.age_forward as _af1; ap.forward_sim_age_structured = _af1.forward_sim_age_structured; dfk.DEP['active'] = False
    if a.survanchor:   # the start readout is the base fit WITHOUT the anchor
        import src.model.age_forward as _af0; ap.forward_sim_age_structured = _af0.forward_sim_age_structured; saf.SA['active'] = False
    if a.estab:   # the start readout is the base fit WITHOUT the establishment readout
        import src.model.age_forward as _af; ap.forward_sim_age_structured = _af.forward_sim_age_structured; ldf.LD['active'] = False
    sb.lat = {k: np.asarray(v, float) for k, v in lat0.items()}; sb.base_abund = None; sb.base_x = None; sb.base_med = None
    row0 = sb.run(dict(name='start', **hab))
    ll0 = -sum(v['loglik'] for v in row0['abundance'].values())
    _rr = tuple(a.real_rates.split(',')) if a.real_rates else ()
    _G = None
    if a.estab and est_info is not None and 'G_east' in (est_info or {}):
        _ge = est_info['G_east']; _yk = sorted(int(k) for k in _ge)
        _G = np.interp(np.asarray(sb.years, float), _yk, [_ge[str(k)] for k in _yk])
    split1 = lambda_split(sb, lat1, d_field, real_data, _rr, _G); split0 = lambda_split(sb, lat0, d_field, real_data, _rr)
    rho = (1 / (1 + math.exp(-float(lat1['manifold_coupling_logit'])))) if a.form == 'exchangeable' else 1.0
    out = dict(form=a.form, field_alpha=a.field_alpha, decouple=a.decouple, estab=a.estab, survanchor=a.survanchor, floor=a.floor, establishment=est_info, estab_gate=a.estab_gate, estab_z=a.estab_z, field_z=a.field_z, edited_inputs=edited_label(a), real_rates=a.real_rates, dd_info=dd_info, bbl_info=bbl_info, vr_alpha_late=a.vr_alpha_late, late_from=a.late_from, train_until=a.train_until, heldout_bbs_ll=({k: v['loglik'] for k, v in row['abundance'].items() if a.train_until and int(k.split('_')[1].split('-')[0]) > a.train_until} if a.train_until else None), free_n50=a.free_n50, n50_raw_end=(float(np.ravel(unpack(v_hab)["n50_raw"])[0]) if a.free_n50 else None), start_neg_logp=f0, end_neg_logp=float(res.fun) * OPT_SCALE, converged=bool(res.success),
               message=str(res.message), n_eval=n[0], minutes=round((time.time() - t0) / 60, 1), rho=rho,
               idio_norm=(float(np.linalg.norm(lat1['manifold_idio'])) if a.form == 'exchangeable' else None),
               bbs_nll_start=ll0, bbs_nll_end=ll, offsets_start=row0['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               offsets_end=row['curves']['rel_10pct']['arrival_offset']['by_cohort'],
               never_reached_end=row['curves']['rel_10pct']['arrival_offset']['never'],
               lambda_split_start=split0, lambda_split_end=split1,
               loglik_by_end={k: v['loglik'] for k, v in row['abundance'].items()}, abundance_end=row['abundance'], disease=dis_info)
    np.savez_compressed(a.out / f'refit_{a.form}{"_estab" if a.estab else ("_survanchor" if a.survanchor else ("_floor" if a.floor else ""))}{"_decouple" if a.decouple else ""}{a.tag}_params.npz',
                        **{k: np.asarray(v) for k, v in lat1.items() if k in targets},
                        **({'estab': np.asarray(res.x[n_hab:n_hab + n_e])} if a.estab else {}),
                        **({'bbl_report_logit': np.asarray(res.x[n_hab + n_e:])} if (a.estab and a.bbl) else {}),
                        **({'survanchor_logit_hi': np.asarray(res.x[n_hab:])} if a.survanchor else {}))
    (a.out / f'refit_{a.form}{"_estab" if a.estab else ("_survanchor" if a.survanchor else ("_floor" if a.floor else ""))}{"_decouple" if a.decouple else ""}{a.tag}_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('form', 'start_neg_logp', 'end_neg_logp', 'converged', 'n_eval', 'rho', 'idio_norm',
                                          'bbs_nll_start', 'bbs_nll_end', 'offsets_end')}), flush=True)
    for k in ('lambda_split_start', 'lambda_split_end'):
        print(k, json.dumps(out[k]['1940-2010']), flush=True)


if __name__ == '__main__':
    main()
