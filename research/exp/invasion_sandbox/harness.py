"""2-D replay sandbox: a fitted GRID54 checkpoint re-run forward with edited habitat, rates or initial state.

Run through sbx_env.sh STEP (frozen code on PYTHONPATH, CPU float64). No fitting: the refined MAP point is plugged
into the model once to capture the exact arguments of forward_sim_age_structured (capture pattern of
research/exp/run_contribution_audit.py), then each arm edits

  data level (before rate projection):  habitat_alpha + alpha_ref -- Z_eff = Z_ref + alpha (Z_t - Z_ref) on Z and
                                        Z_disp (alpha 0 = frozen at Z_ref, 1 = fitted);
  argument level (after projection):    region ('east_only' / 'west_only' native seed), release_year, founders_scale,
                                        fmax_scale (c recomputed as in age_priors.equilibrium_age_quantities),
                                        allee_gamma_scale (n50 / scale), disease_scale.

and replays. Outputs per arm: curves (east occupied area, radius from the release cell, east total) at absolute and
relative thresholds, first-occupied-year offsets vs first BBS positive (as grid54_front_speed.py), potential-lambda
summaries (Sa + Sj Fmax, East / Plains / West, early / modern), and a SHA-256 of the final state (host canary).

Usage (inside sbx_env.sh): python harness.py --spec SPEC.json --out OUT_DIR
SPEC.json: {"arms": [{"name": "fitted"}, {"name": "frozen_modern", "habitat_alpha": 0, "alpha_ref": [1966, 2025]},
                     {"name": "a50_east", "habitat_alpha": 0.5, "alpha_ref": [1966, 2025], "region": "east_only"}]}
"""
import argparse, hashlib, inspect, json, os, time
from pathlib import Path
import jax
PRECISION = os.environ.get('SBX_PRECISION', 'float64')
jax.config.update('jax_enable_x64', PRECISION == 'float64')
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from src.config_utils import load_age_model_config, load_data_config
from src.model import age_priors as ap
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params, auto_delta_params_to_latents

BIRDS_PER_KM2_PER_UNIT = 20 / (50 * np.pi * .4 * .5)   # 20 route counts per model unit; 25.13 km^2 sampled, detection .5
REGIONS = {'east': (-95., 180.), 'plains': (-100., -95.), 'west': (-180., -100.)}  # exhaustive, 25-49N (G13)


def capture(data, lat):
    original = ap.forward_sim_age_structured; got = []
    def record(*aa, **kw):
        b = inspect.signature(original).bind(*aa, **kw); b.apply_defaults()
        got.append(dict(b.arguments)); return original(*aa, **kw)
    ap.forward_sim_age_structured = record
    try:
        handlers.seed(handlers.condition(ap.build_model_2d, data=lat), rng_seed=104)(data=data, prior_scale=1.)
    finally:
        ap.forward_sim_age_structured = original
    if len(got) != 1:
        raise ValueError(f'expected one forward call, got {len(got)}')
    return got[0], original


def edit_habitat(data, years, alpha, ref, keys=('Z_gathered', 'Z_disp_gathered')):
    """keys: which habitat arrays to edit -- Z_gathered (local rates) and / or Z_disp_gathered (journey survival)."""
    d = dict(data); win = (years >= ref[0]) & (years <= ref[1])
    for k in keys:
        z = jnp.asarray(d[k]); m = z[jnp.asarray(win)].mean(0, keepdims=True)
        d[k] = (m + alpha * (z - m)).astype(z.dtype)
    return d


def edit_args(x, arm, cell_lon, years, rows, cols):
    x = dict(x)
    if arm.get('region'):
        keep = (cell_lon >= -95.) if arm['region'] == 'east_only' else (cell_lon < -95.)
        ip = np.asarray(x['initpop_latent']).copy()
        if ip.ndim == 1 and ip.shape[0] == len(cell_lon):
            ip[~keep] = 0.
        elif ip.ndim == 2:                       # full (Ny, Nx) grid
            g = np.zeros(ip.shape, bool); g[rows, cols] = keep; ip[~g] = 0.
        else:
            raise ValueError(f'initpop_latent shape {ip.shape}; extend edit_args')
        x['initpop_latent'] = jnp.asarray(ip, dtype=jnp.asarray(x['initpop_latent']).dtype)
        if arm['region'] == 'west_only':
            x['inv_pop'] = jnp.zeros_like(jnp.asarray(x['inv_pop']))
    if 'release_year' in arm:
        x['inv_timestep'] = int(np.nonzero(years == arm['release_year'])[0][0])
    if 'founders_scale' in arm:
        x['inv_pop'] = jnp.asarray(x['inv_pop']) * arm['founders_scale']
    if 'fmax_scale' in arm:
        Sa, Sj = jnp.asarray(x['Sa_flat']), jnp.asarray(x['Sj_flat'])
        F = jnp.asarray(x['Fmax_flat']) * arm['fmax_scale']
        x['Fmax_flat'] = F
        x['c_flat'] = c_of(F, Sa, Sj, x)
    if 'allee_gamma_scale' in arm:
        x['allee_gamma'] = jnp.asarray(x['allee_gamma']) * arm['allee_gamma_scale']
    if 'disease_scale' in arm and x.get('disease_m_amp') is not None:
        x['disease_m_amp'] = jnp.asarray(x['disease_m_amp']) * arm['disease_scale']
    if 'n50_birds_km2' in arm:   # A1: absolute mate finding at a stated density (gamma acts on model units)
        x['allee_relative'] = False
        x['allee_gamma'] = jnp.asarray(np.log(2.) / (arm['n50_birds_km2'] / BIRDS_PER_KM2_PER_UNIT),
                                       dtype=jnp.asarray(x['Sa_flat']).dtype)
    if arm.get('rate_edits'):    # C1 / R5: multiplicative time x region modifiers on vital-rate arrays
        T = len(years); masks = region_masks(cell_lon)
        mult = {f: np.ones((T, len(cell_lon))) for f in ('Fmax', 'K', 'Sa', 'Sj')}
        for e in arm['rate_edits']:
            m = masks.get(e.get('region', 'all'), np.ones(len(cell_lon), bool))
            mult[e['field']][:, m] *= time_profile(e, years)[:, None]
        for f, mf in mult.items():
            if np.any(mf != 1):
                key = f + '_flat'; a = np.asarray(x[key]); x[key] = jnp.asarray(a * mf.reshape(a.shape), dtype=a.dtype)
        Sa, Sj, F = (jnp.asarray(x[k]) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'))
        x['Sa_flat'] = jnp.minimum(Sa, .995); x['Sj_flat'] = jnp.minimum(Sj, x['Sa_flat'])
        x['c_flat'] = c_of(F, x['Sa_flat'], x['Sj_flat'], x)
    if arm.get('rate_array'):    # R6: explicit (T, N_land) multipliers per field (in-process arms only, not JSON)
        for f, mf in arm['rate_array'].items():
            key = f + '_flat'; a = jnp.asarray(x[key]); x[key] = a * jnp.asarray(mf, a.dtype).reshape(a.shape)
        Sa, Sj, F = (jnp.asarray(x[k]) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'))
        x['Sa_flat'] = jnp.minimum(Sa, .995); x['Sj_flat'] = jnp.minimum(Sj, x['Sa_flat'])
        x['c_flat'] = c_of(F, x['Sa_flat'], x['Sj_flat'], x)
    for k, v in (arm.get('forward_args') or {}).items():   # direct overrides, e.g. attraction [a, eps, adults]
        if k not in x:
            raise ValueError(f'forward argument {k} not in this fit code')
        x[k] = tuple(v) if isinstance(v, list) else v
    if arm.get('band_ramp'):     # B1: East-only (or any region) juvenile radial-band reweighting through time
        w = band_ramp_weights(arm['band_ramp'], x, cell_lon, years)
        if 'directional_weights' in x:
            x['directional_weights'] = w
        elif 'juvenile_kernel_weights' in x and arm.get('region') == 'east_only':
            # older fit code: global per-year band weights -- East-only in effect when the West is removed
            east = np.asarray(cell_lon) >= -95.
            x['juvenile_kernel_weights'] = jnp.asarray(np.asarray(w)[:, east][:, 0, :], dtype=w.dtype)
        else:
            raise ValueError('band_ramp needs directional_weights, or juvenile_kernel_weights with region east_only')
    return x


def region_masks(cell_lon):
    return dict(all=np.ones(len(cell_lon), bool), east=cell_lon >= -95., west=cell_lon < -100.,
                not_west=cell_lon >= -100., plains=(cell_lon >= -100.) & (cell_lon < -95.))


def time_profile(e, years):
    """Multiplier through time: 'logistic' 1 + a / (1 + exp(-(t - t_mid) / tau)) (minus its 1902 value when
    'anchor_start'), 'step' 1 + a after t_mid, 'spline' piecewise-linear through (knots, values)."""
    form = e.get('form', 'logistic'); t = years.astype(float)
    if form == 'logistic':
        s = 1 / (1 + np.exp(-(t - e['t_mid']) / e['tau']))
        return 1 + e['a'] * s
    if form == 'step':
        return np.where(t >= e['t_mid'], 1 + e['a'], 1.)
    if form == 'spline':
        return np.interp(t, e['knots'], e['values'])
    raise ValueError(form)


def band_ramp_weights(spec, x, cell_lon, years):
    """(time, N_land, K) origin multipliers for the juvenile kernels: band log-weights (outer bands relative to the
    innermost) move linearly from spec['logw_start'] (before year_start) to spec.get('logw_end', 0) (after year_end) for
    origins in spec['region']; elsewhere 1. Normalised exactly as age_forward.juvenile_band_weights (mass-preserving)."""
    from src.model import age_forward as af
    stack = np.asarray(x['juvenile_fft_kernel_stack']); Kk = stack.shape[0]; nb = Kk // af.N_DIRECTIONS
    mass = np.real(stack[:, 0, 0]).reshape(af.N_DIRECTIONS, nb).sum(0)
    lw0 = np.asarray(spec['logw_start'], float); lw1 = np.asarray(spec.get('logw_end', np.zeros(nb - 1)), float)
    f = np.clip((years - spec['year_start']) / max(spec['year_end'] - spec['year_start'], 1), 0, 1)
    out = np.ones((len(years), len(cell_lon), Kk))
    m = region_masks(cell_lon)[spec.get('region', 'east')]
    for i, fi in enumerate(f):
        th = np.concatenate([[0.], (1 - fi) * lw0 + fi * lw1])
        w = np.exp(th) / np.sum(mass * np.exp(th))
        out[i][m] = np.tile(w, af.N_DIRECTIONS)[None]
    return jnp.asarray(out, dtype=jnp.asarray(x['Sa_flat']).dtype)


def install_floor(eps_birds_km2, exempt=None):
    """A2: density floor inside the scan -- cells whose post-breeding density is below eps (birds/km^2) are emptied.
    Patches age_forward.reproduction_age_structured (resolved by name at trace time); returns an undo function."""
    from src.model import age_forward as af
    orig = af.reproduction_age_structured; eps = eps_birds_km2 / BIRDS_PER_KM2_PER_UNIT
    def wrapped(*aa, **kw):
        out = orig(*aa, **kw); Na, Nj = out[0], out[1]
        keep = (Na + Nj) >= eps
        if exempt is not None:
            keep = keep | exempt
        return (jnp.where(keep, Na, 0.), jnp.where(keep, Nj, 0.)) + tuple(out[2:])
    af.reproduction_age_structured = wrapped
    return lambda: setattr(af, 'reproduction_age_structured', orig)


def c_of(F, Sa, Sj, x):
    """Crowding brake c (age_priors.equilibrium_age_quantities): yearling F Sj/(1-Sa) - 1, else F Sa Sj/(1-Sa) - 1."""
    xp = jnp if isinstance(F, jnp.ndarray) else np
    num = F * Sj if bool(x.get('yearling_breeding')) else F * Sa * Sj
    return xp.maximum(num / (1.0 - Sa + 1e-6) - 1.0, 0.0)


def potential_lambda(Sa, Sj, F, yearling):
    """Potential (density-, Allee-, movement-free) growth: yearling Sa + Sj F; delayed breeding the dominant
    eigenvalue of [[Sa, Sj], [F Sa, 0]] = (Sa + sqrt(Sa^2 + 4 F Sa Sj)) / 2."""
    return Sa + Sj * F if yearling else .5 * (Sa + np.sqrt(Sa * Sa + 4 * F * Sa * Sj))


def check_c_formula(x):
    """The captured c must equal the formula used by fmax_scale (else rate edits would be inconsistent)."""
    Sa, Sj, F = (np.asarray(x[k]) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'))
    c = c_of(F, Sa, Sj, x)
    return float(np.max(np.abs(c - np.asarray(x['c_flat']))))


def to_land(a, rows, cols, T, Nx):
    """(T, Ny, Nx), (T, Ny*Nx) or (T, N_land) -> (T, N_land)."""
    a = np.asarray(a)
    if a.ndim == 3:
        return a[:, rows, cols]
    a = a.reshape(T, -1)
    if a.shape[1] == len(rows):
        return a
    if a.shape[1] % Nx == 0:
        return a[:, rows * Nx + cols]
    raise ValueError(f'cannot map shape {a.shape} to land cells')


class Sandbox:
    """Loads one fit once; run(arm) replays an arm and returns its readout row. The FIRST arm run is the reference
    for relative thresholds and abundance ratios (as in grid54_front_speed.py)."""

    def __init__(self, refined):
        cfg = load_age_model_config(); dcfg = load_data_config()
        self.data = data = load_data(cfg['input_dir'], target_device=jax.devices()[0], precision=PRECISION,
                                     verbose=False)
        params, _ = load_map_params(refined); self.lat = auto_delta_params_to_latents(params)
        if PRECISION != 'float64':
            self.lat = {k: (np.asarray(v, np.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v)
                        for k, v in self.lat.items()}
        self.years = years = np.asarray(data['years'])
        self.rows, self.cols = rows, cols = np.asarray(data['land_rows']), np.asarray(data['land_cols'])
        self.T = len(years); self.Nx = int(np.asarray(data['land_mask']).shape[1])
        import rasterio
        from rasterio.warp import transform as crs_transform
        with rasterio.open(dcfg['grid']['ref_raster']) as src:
            xx, yy = rasterio.transform.xy(src.transform, rows, cols, offset='center')
            self.xx, self.yy = xx, yy = np.asarray(xx), np.asarray(yy)
            lon, latd = map(np.asarray, crs_transform(src.crs, 'EPSG:4326', list(xx), list(yy)))
            self.cell_km2 = abs(src.transform.a * src.transform.e) / 1e6
        self.lon, self.latd = lon, latd
        band = (latd >= 25) & (latd < 49)
        self.region = {k: band & (lon >= lo) & (lon < hi) for k, (lo, hi) in REGIONS.items()}
        self.lookup = lookup = -np.ones((int(rows.max()) + 1, int(cols.max()) + 1), int)
        lookup[rows, cols] = np.arange(len(rows))
        ocell = lookup[np.asarray(data['obs_rows']), np.asarray(data['obs_cols'])]
        oyr = years[np.asarray(data['obs_time_indices'])]; y = np.asarray(data['observed_results'])
        real = np.asarray(data['obs_obsroute_idx']) >= 0 if 'obs_obsroute_idx' in data else np.ones(len(y), bool)
        first_obs = np.full(len(rows), np.nan); surveyed = np.full(len(rows), np.nan)
        for i in np.flatnonzero(real & (ocell >= 0)):
            c = ocell[i]; surveyed[c] = np.fmin(surveyed[c], oyr[i])
            if y[i] > 0: first_obs[c] = np.fmin(first_obs[c], oyr[i])
        self.first_obs = first_obs
        self.ok = self.region['east'] & np.isfinite(first_obs) & (first_obs >= surveyed + 3)
        obs_ok = real & (ocell >= 0)
        self.obs_ok = obs_ok
        self.o_cell, self.o_t, self.o_y = ocell[obs_ok], np.asarray(data['obs_time_indices'])[obs_ok], y[obs_ok]
        self.eras = [(1966, 1979), (1980, 1993), (1994, 2009), (2010, 2025)]
        self.base_abund = None; self.base_x = None; self.base_med = None

    def observe(self, d, out):
        """The fit's OWN observation model on a replayed state: the model is traced again at the refined point with
        forward_sim_age_structured returning `out`, so route / observer / year effects, the breeding-season mix and
        the NB2 (or zero-inflated) likelihood apply exactly as in the fit. Returns expected count, log-likelihood and
        P(count = 0) for the real BBS rows (pseudo-zeros dropped), in BBS route-count units -- no unit conversion."""
        original = ap.forward_sim_age_structured
        ap.forward_sim_age_structured = lambda *aa, **kw: out
        try:
            tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=self.lat),
                                              rng_seed=104)).get_trace(data=d, prior_scale=1.)
        finally:
            ap.forward_sim_age_structured = original
        fn, y = tr['obs']['fn'], jnp.asarray(self.data['observed_results'])
        m = self.obs_ok
        # R49 (round-3 review): the base overlay's observation_mask (obs_weight 0: 2021-25 + 31 blocks) makes fn a
        # MaskedDistribution whose log_prob is 0 on held-out rows; score those rows with the raw distribution.
        raw = getattr(fn, 'base_dist', fn)
        ow = self.data.get('obs_weight')
        held = (np.asarray(ow) <= 0) if ow is not None else np.zeros(len(np.asarray(y)), bool)
        return dict(mu=np.asarray(tr['expected_obs']['value'])[m], ll=np.asarray(fn.log_prob(y))[m],
                    p0=np.exp(np.asarray(fn.log_prob(jnp.zeros_like(y))))[m],
                    ll_raw=np.asarray(raw.log_prob(y))[m], p0_raw=np.exp(np.asarray(raw.log_prob(jnp.zeros_like(y))))[m],
                    held=held[m])

    def run(self, arm, keep_density=False):
        data, years, rows, cols, T, Nx = self.data, self.years, self.rows, self.cols, self.T, self.Nx
        region, ok, first_obs = self.region, self.ok, self.first_obs
        t0 = time.time()
        d = edit_habitat(data, years, arm['habitat_alpha'], arm.get('alpha_ref', [1966, 2025]),
                         tuple(arm.get('habitat_keys', ('Z_gathered', 'Z_disp_gathered')))) \
            if 'habitat_alpha' in arm else data
        x, fwd = capture(d, self.lat)
        if self.base_x is None:
            self.base_x = x; c_err = check_c_formula(x)
            if c_err > (1e-8 if PRECISION == 'float64' else 1e-3):
                raise ValueError(f'captured c differs from the life-cycle formula by {c_err}')
            inv_loc = np.asarray(x['inv_locations']).reshape(-1, 2)[0]
            self.rel = int(self.lookup[inv_loc[0], inv_loc[1]])
            self.dist_km = np.hypot(self.xx - self.xx[self.rel], self.yy - self.yy[self.rel]) / 1e3
        if arm.get('k_real'):   # capacity from the REAL-Z capture (R21 / A9 decoupled structure) on edited readouts
            if getattr(self, '_K_real', None) is None:
                self._K_real = capture(data, self.lat)[0]['K_flat']
            x = dict(x); x['K_flat'] = self._K_real
        x = edit_args(x, arm, self.lon, years, rows, cols)
        undo = None
        if 'floor' in arm:   # release cell exempt: founders are far below any floor once spread over a 54 km cell
            ex = np.zeros(np.asarray(x['land_mask']).shape, bool); il = np.asarray(x['inv_locations']).reshape(-1, 2)
            ex[il[:, 0], il[:, 1]] = True
            undo = install_floor(arm['floor'], jnp.asarray(ex))
        try:
            out = fwd(**x); out = jax.tree.map(jax.block_until_ready, out)
        finally:
            if undo: undo()
        tot = to_land(out[0], rows, cols, T, Nx)
        h = hashlib.sha256(np.ascontiguousarray(np.asarray(out[0])[-1]).tobytes()).hexdigest()
        birds = tot * BIRDS_PER_KM2_PER_UNIT
        if self.base_med is None:
            self.base_med = float(np.nanmedian(tot[(years >= 2010)][:, ok])) if ok.any() else np.nan
        med_occ = self.base_med
        curves = {}
        for name, thr in (('abs_0.1_birds_km2', .1 / BIRDS_PER_KM2_PER_UNIT), ('abs_0.01_birds_km2', .01 / BIRDS_PER_KM2_PER_UNIT),
                          ('rel_10pct', .1 * med_occ),
                          ('rel_1pct', .01 * med_occ)):
            occ = tot >= thr
            e = occ & region['east'][None]
            curves[name] = dict(
                threshold_model_units=float(thr),
                east_occupied_km2=[float(v) for v in e.sum(1) * self.cell_km2],
                radius90_km=[float(np.percentile(self.dist_km[r], 90)) if r.any() else 0. for r in e],
                plains_occupied_km2=[float(v) for v in (occ & region['plains'][None]).sum(1) * self.cell_km2])
            first_mod = np.where(occ.any(0), years[np.argmax(occ, 0)], np.nan)
            lag = first_mod[ok] - first_obs[ok]; fo = first_obs[ok]
            def med(v):
                return float(np.nanmedian(v)) if np.isfinite(v).any() else None
            curves[name]['arrival_offset'] = dict(
                median=med(lag), never=float(np.mean(np.isnan(lag))),
                by_cohort={f'{a0}-{b0 - 1}': med(lag[(fo >= a0) & (fo < b0)])
                           for a0, b0 in zip([1966, 1975, 1985, 1995], [1975, 1985, 1995, 2010])})
        abund = {}
        for rname in ('east', 'plains', 'west'):
            for a0, b0 in self.eras:
                s = region[rname][self.o_cell] & (years[self.o_t] >= a0) & (years[self.o_t] <= b0)
                mstate = tot[self.o_t[s], self.o_cell[s]]
                good = np.isfinite(mstate)
                corr = (float(np.corrcoef(np.log1p(mstate[good] * 20), np.log1p(self.o_y[s][good]))[0, 1])
                        if good.sum() > 10 and mstate[good].std() > 0 else None)
                abund[f'{rname}_{a0}_{b0}'] = dict(model_sum=float(np.nansum(mstate)), obs_sum=float(self.o_y[s].sum()),
                                                   corr=corr, nonfinite=int((~good).sum()))
        ob = self.observe(d, out)
        for rname in ('east', 'plains', 'west'):
            for a0, b0 in self.eras:
                s = region[rname][self.o_cell] & (years[self.o_t] >= a0) & (years[self.o_t] <= b0)
                abund[f'{rname}_{a0}_{b0}'].update(
                    n_rows=int(s.sum()), expected_sum=float(ob['mu'][s].sum()),
                    obs_over_expected=float(self.o_y[s].sum() / ob['mu'][s].sum()) if ob['mu'][s].sum() > 0 else None,
                    loglik=float(ob['ll'][s].sum()), zero_share_obs=float((self.o_y[s] == 0).mean()) if s.any() else None,
                    zero_share_pred=float(ob['p0'][s].mean()) if s.any() else None)
                h, tr_ = s & ob['held'], s & ~ob['held']
                yr = years[self.o_t]
                abund[f'{rname}_{a0}_{b0}'].update(   # R49: held-out (never-fitted) rows scored with the raw distribution
                    n_heldout=int(h.sum()), loglik_heldout=float(ob['ll_raw'][h].sum()),
                    loglik_heldout_blocks=float(ob['ll_raw'][h & (yr <= 2020)].sum()),
                    loglik_heldout_2021=float(ob['ll_raw'][h & (yr > 2020)].sum()),
                    zero_share_obs_unmasked=float((self.o_y[tr_] == 0).mean()) if tr_.any() else None,
                    zero_share_pred_unmasked=float(ob['p0_raw'][tr_].mean()) if tr_.any() else None,
                    zero_share_obs_heldout=float((self.o_y[h] == 0).mean()) if h.any() else None,
                    zero_share_pred_heldout=float(ob['p0_raw'][h].mean()) if h.any() else None)
        if self.base_abund is None:
            self.base_abund = abund
        for k, v in abund.items():
            b = self.base_abund[k]['model_sum']
            v['model_over_reference'] = v['model_sum'] / b if b > 0 else None
        Sa, Sj, F = (to_land(x[k], rows, cols, T, Nx) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'))
        lam = potential_lambda(Sa, Sj, F, bool(x.get('yearling_breeding')))
        def era_mean(m, a0, b0):
            w = (years >= a0) & (years <= b0); return float(lam[w][:, m].mean())
        lam_summary = {r: dict(early=era_mean(m, 1902, 1915), bbs_early=era_mean(m, 1966, 1985),
                               modern=era_mean(m, 2010, 2025)) for r, m in region.items()}
        row = dict(step=os.environ.get('SBX_STEP'), arm=arm, seconds=round(time.time() - t0, 1), final_state_sha256=h,
                   nonfinite_state=int((~np.isfinite(tot)).sum()),
                   east_total_birds=[float(v) for v in (birds[:, region['east']] * self.cell_km2).sum(1)],
                   potential_lambda=lam_summary, abundance=abund, curves=curves, years=[int(v) for v in years],
                   release_cell=self.rel, release_year=int(years[int(x['inv_timestep'])]), n_front_cells=int(ok.sum()))
        if keep_density:
            row['_density'] = tot
        return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--refined', default=os.environ.get('SBX_REFINED'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(a.spec.read_text())
    sb = Sandbox(a.refined)
    for arm in spec['arms']:
        row = sb.run(arm, keep_density=True)
        tot = row.pop('_density')
        np.savez_compressed(a.out / f"{arm['name']}_density.npz", total=tot.astype(np.float32), years=sb.years)
        print(json.dumps(dict(arm=arm['name'], sec=row['seconds'], sha=row['final_state_sha256'][:12],
                              nonfinite=row['nonfinite_state'], offset10=row['curves']['rel_10pct']['arrival_offset'],
                              lam_east=row['potential_lambda']['east'])), flush=True)
        with (a.out / 'rows.jsonl').open('a') as fh:
            fh.write(json.dumps(row) + '\n')
    np.savez_compressed(a.out / 'cells.npz', lon=sb.lon, lat=sb.latd, dist_km=sb.dist_km, first_obs=sb.first_obs,
                        ok=sb.ok, rows=sb.rows, cols=sb.cols)


if __name__ == '__main__':
    main()
