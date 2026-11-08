"""Banding-movement likelihood (SURVIVAL_BBL_SCOPING s.3.3; proposal C, agent_levers 2026-11-07).

Differentiable (JAX) log-likelihood of BBL re-encounter DESTINATIONS given the model's own movement machinery:

  juvenile natal records (banded HY/local Apr-Sep in cell c, year y; first re-encounter Apr-Jul of a later year, cell d):
      surviving mass arriving at land cell e from origin c in the dispersal step of year index t = ty(y) + t_offset:
          arr_J(e) = (1 - p(c,t)) Sj(t,c) [e == c]  +  p(c,t) sum_k  W_k(e - c) / edge_k(c) * Q(t,e,k)
  adult breeding-dispersal records (banded AHY/SY/ASY Apr-Jul, re-encountered Apr-Jul of a later year; 319 on the BBL file):
          arr_A(e) = (1 - p(c,t)) [e == c]  +  p(c,t) Wa(e - c) / edge_a(c)        (x Sa(t,e) if adult_survival)
  p(c,t) = the model's departure probability (Veit, both ages) x native-departure scale (1 - r w_nat(c)):
          x = (Na + Nj)(t-1, c) / max(K(t,c), 1e-6);  p_v = clip(x - softplus(10 (x - 1)) / 10, 1e-6, 1)
  REPORTING: rho(e) = 1 at e == c, (1 + dist(c,e) / d0)^(-gamma) elsewhere (ONE parameter gamma >= 0, shared by both ages;
          d0 = 54 km). Long moves are less often reported / re-encountered by the original bander's effort. The adult records
          identify gamma because adults are near-sedentary in the model and in the data (s.3.3: "adult breeding-dispersal
          records ... calibrate how fast rho falls with distance from the banding site").
  Two likelihood forms:
      'class': P(class j | c) = sum_{e in j} arr(e) rho(e) / sum_e arr(e) rho(e), classes own / <=155 / 155-483 / >483 km
               (cell-centre haversine; natal_ll.py's form);
      'cell' : P(e | c) = arr(e) rho(e) / sum arr rho  (exact destination cell; sensitive to the model's fine-scale arrival field).
  KERNEL PARAMETER (one, optional): mixture share s of an ALTERNATE juvenile stack (same bands/labels; e.g. J110 or the
          2-D-misread J330), arrivals W = (1 - s) W_base / edge_base + s W_alt / edge_alt with Q shared (as kernel_ramp does
          in the forward). The SAME s must drive the forward (population_model.kernel_ramp with a constant share) -- the
          movement term must never see a different kernel than the population.

Compatibility mode (natal_ll.py, for validation): compat=True -> no edge correction, W normalised by its global sum, class
form, rho = [1, 1/k, 1/k, 1/k]. natal_ll.py reads p from the instrumented forward at the banding-year index; t_offset = 0.

Usage (laplace_fit.parts): see INTEGRATION.md. All heavy arrays are built once on the host by build_cache().
"""
import numpy as np
import jax
import jax.numpy as jnp

CLS_EDGES = (155.0, 483.0)
JUV_AGES = ('2', '4')                 # HY, local
ADULT_AGES = ('1', '5', '6', '8')     # AHY, SY, ASY, ATY
BBL_CSV = '/mnt/e/Datasets/houfin/bbl/raw_2025/NABBP_2025_grp_37.csv'


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def read_pairs(csv=BBL_CSV, ages=JUV_AGES, band_months=(4, 9), enc_months=(4, 7)):
    """BBL banding -> first later-year re-encounter pairs (natal_ll.py's filter; band-only finds 98 excluded)."""
    import pandas as pd
    cols = ['band', 'event_type', 'event_month', 'event_year', 'lat_dd', 'lon_dd', 'age_code', 'how_obtained_code']
    df = pd.read_csv(csv, usecols=cols, dtype=str)
    for c in ('event_month', 'event_year', 'lat_dd', 'lon_dd'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['event_month', 'event_year', 'lat_dd', 'lon_dd'])
    B = df[(df.event_type == 'B') & df.age_code.isin(list(ages)) & df.event_month.between(*band_months)]
    E = df[(df.event_type == 'E') & df.event_month.between(*enc_months) & (df.how_obtained_code.fillna('') != '98')]
    m = B.merge(E, on='band', suffixes=('_b', '_e'))
    m = m[m.event_year_e > m.event_year_b].sort_values('event_year_e').drop_duplicates('band', keep='first')
    return m


def map_records(pairs, years, lat, lon, snap_km=60.0):
    """Pairs -> (c, d, t_band, gap_years, region) on land cells (natal_ll.py: nearest land cell, both ends within 60 km)."""
    yrs = np.asarray(years).astype(int)
    out = []
    for _, r in pairs.iterrows():
        c = int(np.argmin(hav(r.lat_dd_b, r.lon_dd_b, lat, lon))); d = int(np.argmin(hav(r.lat_dd_e, r.lon_dd_e, lat, lon)))
        if hav(r.lat_dd_b, r.lon_dd_b, lat[c], lon[c]) > snap_km or hav(r.lat_dd_e, r.lon_dd_e, lat[d], lon[d]) > snap_km:
            continue
        y = int(r.event_year_b)
        if y < yrs[0] or y > yrs[-1]:
            continue
        reg = 0 if r.lon_dd_b < -100 else (1 if r.lon_dd_b < -80 else 2)         # West / central / East (reporting only)
        out.append((c, d, int(np.flatnonzero(yrs == y)[0]), int(r.event_year_e - r.event_year_b), reg))
    return np.asarray(out, dtype=np.int64).reshape(-1, 5)


def _real_kernels(fft_stack):
    W = np.real(np.fft.ifft2(np.asarray(fft_stack)))            # (K, Ly, Lx) or (Ly, Lx)
    return W


def _kernel_rows(W, rows, cols, origins):
    """W[..., (rows - rows[c]) % Ly, (cols - cols[c]) % Lx] for every origin c -> (n, NL[, K])."""
    Ly, Lx = W.shape[-2:]
    dy = (rows[None, :] - rows[origins][:, None]) % Ly
    dx = (cols[None, :] - cols[origins][:, None]) % Lx
    if W.ndim == 3:
        return np.moveaxis(W[:, dy, dx], 0, -1)                   # (n, NL, K)
    return W[dy, dx]                                               # (n, NL)


def build_cache(data, years, lat, lon, juv=None, adult=None, alt_stack=None, alt_edge=None, compat=False,
                dtype=np.float32, d0=54.0):
    """Host-side precomputation. data: the model data dict (numpy or jax arrays): land_rows, land_cols,
    juvenile_fft_kernel_stack (K, Ly, Lx), juvenile_edge_correction_stack (K, Ny, Nx), adult_fft_kernel (Ly, Lx),
    adult_edge_correction (Ny, Nx), initpop_latent (Ny, Nx). juv / adult: arrays from map_records (None -> read the BBL file)."""
    rows = np.asarray(data['land_rows']); cols = np.asarray(data['land_cols']); NL = rows.size
    if juv is None:
        juv = map_records(read_pairs(ages=JUV_AGES, band_months=(4, 9)), years, lat, lon)
    if adult is None:
        adult = map_records(read_pairs(ages=ADULT_AGES, band_months=(4, 7)), years, lat, lon)
    cache = dict(NL=NL, compat=bool(compat), d0=float(d0))

    def side(rec, juvenile):
        c, d, t = rec[:, 0], rec[:, 1], rec[:, 2]
        km = hav(lat[c][:, None], lon[c][:, None], lat[None, :], lon[None, :])                 # (n, NL)
        own = (np.arange(NL)[None, :] == c[:, None])
        cls = np.where(own, 0, np.where(km <= CLS_EDGES[0], 1, np.where(km <= CLS_EDGES[1], 2, 3))).astype(np.int8)
        S = dict(c=jnp.asarray(c), d=jnp.asarray(d), t=jnp.asarray(t), reg=jnp.asarray(rec[:, 4]),
                 cls=jnp.asarray(cls), own=jnp.asarray(own), cls_obs=jnp.asarray(cls[np.arange(c.size), d]),
                 logdist=jnp.asarray(np.log1p(km / d0).astype(dtype)), n=int(c.size))
        if juvenile:
            Wb = _real_kernels(data['juvenile_fft_kernel_stack'])
            if compat:
                Wb = Wb / Wb.sum()
                S['W'] = jnp.asarray(_kernel_rows(Wb, rows, cols, c).astype(dtype))
            else:
                e = np.asarray(data['juvenile_edge_correction_stack'])[:, rows[c], cols[c]].T      # (n, K)
                S['W'] = jnp.asarray((_kernel_rows(Wb, rows, cols, c) / (e[:, None, :] + 1e-6)).astype(dtype))
                if alt_stack is not None:
                    Wa_ = _real_kernels(alt_stack)
                    ea = np.asarray(alt_edge)[:, rows[c], cols[c]].T
                    S['W_alt'] = jnp.asarray((_kernel_rows(Wa_, rows, cols, c) / (ea[:, None, :] + 1e-6)).astype(dtype))
        else:
            Wa = _real_kernels(data['adult_fft_kernel'])
            ea = np.asarray(data['adult_edge_correction'])[rows[c], cols[c]]
            S['W'] = jnp.asarray((_kernel_rows(Wa, rows, cols, c) / (ea[:, None] + 1e-6)).astype(dtype))
        return S
    cache['juv'] = side(juv, True)
    cache['adult'] = side(adult, False) if adult is not None and len(adult) else None
    ip = np.asarray(data['initpop_latent'])[rows, cols]
    cache['w_nat'] = jnp.asarray(ip / max(ip.max(), 1e-12))
    cache['rows'] = jnp.asarray(rows); cache['cols'] = jnp.asarray(cols)
    return cache


def _p_veit(Ntot, K, scale):
    x = Ntot / jnp.maximum(K, 1e-6)
    p = jnp.clip(x - jax.nn.softplus(10.0 * (x - 1.0)) / 10.0, 1e-6, 1.0)
    return p * scale


def natal_nll(trace, cache, T, sel=None, gamma=0.0, k_compat=None, form='class', share=None, t_offset=1,
              use_adults=True, adult_survival=False, native_r=None, return_parts=False,
              adult_stay_logit=None, adult_gamma_only=True, adult_fixed_p=None):
    """Negative log-likelihood of the movement records. trace: numpyro trace (or dict name -> array/{'value'}).
    gamma: reporting decay (traced scalar, >= 0). k_compat: if given, class reporting rho = [1, 1/k, 1/k, 1/k] (natal_ll).
    share: kernel-mixture share of the alternate stack (scalar or (n_juv,)); None = base kernel only.
    native_r: native_departure_r (default: read from the trace when present, else no native scale).
    t_offset: 1 (default, model-correct) = birds banded in the season of year index i disperse in step i + 1 (p from the
              end-of-step-i state, K / Q / Sj of step i + 1); 0 = natal_ll.py's convention (validation only).
    adult_gamma_only (default True): the adult records inform ONLY the reporting decay gamma (and adult_stay_logit) --
              the population quantities in the adult arrival field are stop_gradient'ed, because the model's adult departure
              (Veit p_adult = p_total, ~45% of adults leave a 54-km cell per year at the banding cells) is a known
              simplification that the adult records contradict (77% own cell); letting them act on N/K would turn them
              into a hidden capacity lever.
    adult_stay_logit: optional traced scalar g; adult departure in the OBSERVATION term = p * sigmoid(g) (a nuisance that
              absorbs the adult-departure misfit so gamma is identified by the distance SHAPE of adult moves)."""
    NL = cache['NL']
    g = lambda nm: trace[nm]['value'] if isinstance(trace.get(nm), dict) else trace.get(nm)
    rows, cols = cache['rows'], cache['cols']
    Na = g('Na_grid'); Nj = g('Nj_grid')
    Ntot = (Na + Nj)[:, rows, cols]                                                        # (T, NL)
    def land(nm):
        v = g(nm)
        if v.ndim >= 2 and v.shape[1] != NL:
            v = jnp.reshape(v, (T, -1) + tuple(v.shape[2:]))
            v = v[:, sel] if v.shape[1] != NL else v
        return v
    K = land('K_flat'); Sj = land('Sj_flat'); Sa = land('Sa_flat'); Q = land('Q_flat')      # Q (T, NL, K)
    r = native_r if native_r is not None else (g('native_departure_r') if 'native_departure_r' in trace else None)
    scale = (1.0 - r * cache['w_nat']) if r is not None else jnp.ones(NL)

    def rho_of(S):
        if k_compat is not None:
            return jnp.where(S['own'], 1.0, 1.0 / k_compat)
        return jnp.where(S['own'], 1.0, jnp.exp(-gamma * S['logdist']))                     # (n, NL)

    def score(S, arr):
        rho = rho_of(S)
        a = jnp.maximum(arr, 0.0) * rho
        tot = jnp.sum(a, axis=1)
        if form == 'class':
            mass = jax.vmap(lambda ai, ci: jax.ops.segment_sum(ai, ci.astype(jnp.int32), num_segments=4))(a, S['cls'])
            ll = jnp.log(jnp.maximum(jnp.take_along_axis(mass, S['cls_obs'][:, None].astype(jnp.int32), 1)[:, 0], 1e-300)) - jnp.log(tot)
        else:
            ll = jnp.log(jnp.maximum(jnp.take_along_axis(a, S['d'][:, None], 1)[:, 0], 1e-300)) - jnp.log(tot)
        return ll, mass if form == 'class' else None

    # juveniles
    S = cache['juv']; t = jnp.clip(S['t'] + t_offset, 1, T - 1); c = S['c']
    p = _p_veit(Ntot[t - 1, c], K[t, c], scale[c])                                           # (n,)
    W = S['W'].astype(Q.dtype)
    if share is not None and 'W_alt' in S:
        s = jnp.broadcast_to(jnp.asarray(share, Q.dtype), (S['n'],))[:, None, None]
        W = (1.0 - s) * W + s * S['W_alt'].astype(Q.dtype)
    move = jnp.einsum('nek,nek->ne', W, Q[t])                                               # (n, NL)
    arr = p[:, None] * move + jnp.where(S['own'], ((1.0 - p) * Sj[t, c])[:, None], 0.0)
    ll_j, mass_j = score(S, arr)
    nll = -jnp.sum(ll_j)
    parts = dict(juv_ll=jnp.sum(ll_j), juv_ll_by_region=jax.ops.segment_sum(ll_j, S['reg'], num_segments=3),
                 juv_p_mean=jnp.mean(p))
    if return_parts and mass_j is not None:
        parts['juv_class_share_by_region'] = jax.ops.segment_sum(mass_j / jnp.sum(mass_j, 1, keepdims=True), S['reg'], num_segments=3)
        parts['juv_class_obs_by_region'] = jax.ops.segment_sum(jax.nn.one_hot(S['cls_obs'], 4), S['reg'], num_segments=3)
    A = cache.get('adult')
    if use_adults and A is not None and k_compat is None:
        ta = jnp.clip(A['t'] + t_offset, 1, T - 1); ca = A['c']
        if adult_fixed_p is not None:      # 2026-11-07 fix (main session): CONSTANT adult departure from the init state, so
            pa = jnp.asarray(adult_fixed_p, Q.dtype); sa_t = Sa[ta]   # the objective and its gradient agree (stop_gradient
        else:                                 # made them inconsistent -> L-BFGS line searches failed at 2 iterations)
            pa = _p_veit(Ntot[ta - 1, ca], K[ta, ca], scale[ca])
            sa_t = Sa[ta]
            if adult_gamma_only:
                pa, sa_t = jax.lax.stop_gradient(pa), jax.lax.stop_gradient(sa_t)
        if adult_stay_logit is not None:
            pa = pa * jax.nn.sigmoid(adult_stay_logit)
        arr_a = pa[:, None] * A['W'].astype(Q.dtype) + jnp.where(A['own'], (1.0 - pa)[:, None], 0.0)
        if adult_survival:
            arr_a = arr_a * sa_t
        ll_a, _ = score(A, arr_a)
        nll = nll - jnp.sum(ll_a)
        parts.update(adult_ll=jnp.sum(ll_a), adult_ll_by_region=jax.ops.segment_sum(ll_a, A['reg'], num_segments=3),
                     adult_p_mean=jnp.mean(pa))
    return (nll, parts) if return_parts else nll


def adult_departure_constants(trace, cache, T, sel=None, t_offset=1, native_r=None):
    """Adult Veit departure at each adult record's banding cell / year, evaluated ONCE (e.g. at the fit's init) and then passed
    as adult_fixed_p, so the adult records inform only gamma / adult_stay_logit through a CONSISTENT objective."""
    import numpy as _np
    NL = cache['NL']
    g = lambda nm: trace[nm]['value'] if isinstance(trace.get(nm), dict) else trace.get(nm)
    rows, cols = cache['rows'], cache['cols']
    Ntot = (g('Na_grid') + g('Nj_grid'))[:, rows, cols]
    K = g('K_flat'); K = jnp.reshape(K, (T, -1)); K = K[:, sel] if K.shape[1] != NL else K
    r = native_r if native_r is not None else (g('native_departure_r') if 'native_departure_r' in trace else None)
    scale = (1.0 - r * cache['w_nat']) if r is not None else jnp.ones(NL)
    A = cache['adult']; ta = jnp.clip(A['t'] + t_offset, 1, T - 1); ca = A['c']
    return _np.asarray(_p_veit(Ntot[ta - 1, ca], K[ta, ca], scale[ca]))
