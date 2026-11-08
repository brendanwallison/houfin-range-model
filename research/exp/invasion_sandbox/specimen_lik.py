"""Effort-aware museum-specimen likelihood (agent_levers round 5, 2026-11-07): where House Finches were AND were not, 1880-1940.

DATA (host side, build_cache):
  E_cd  target-group effort: preserved Aves specimens (GBIF classKey 212) collected in land cell c in decade d, from
        effort_aves_by_year_1880_1940.json (facet=year per 54-km cell polygon; gbif_effort_decade.py).
  y_cd  House Finch preserved specimens in cell c, decade d (records.csv, coordinate uncertainty <= 25 km or unknown).
  Decades: 1880-1901 (read against the 1902 model state; the model starts in 1902 after its spin-up), 1902-09, 1910-19,
  1920-29, 1930-40. Cells: every cell with E_cd > 0 in the effort file (west of -95 + all CEC L1 Great Plains).

MODEL (one row per cell-decade with effort):
  P(y_cd > 0) = 1 - exp(-s_e * E_cd^beta * g(N_cd)),   g(N) = 1 - (1 + N/k)^(-k)  (k = presence_k .437: the same NB2
  "a bird is there" detection logic as the range presences / absences), N_cd = the modelled cell total (route-count scale),
  mean over the decade's model years.
  - OCCUPANCY (detection) form, not a count: museum series are clumped (one collector takes 20 finches in an afternoon), so
    the count beyond the first specimen carries collector behaviour, not abundance. The binary form keeps the two things that
    matter: a positive anywhere is presence; an absence in a cell where many birds were collected is evidence of absence or
    rarity, in proportion to E^beta.
  - beta in (0, 1]: specimens are not independent collecting events (series, expeditions); beta < 1 discounts big totals.
    One nuisance, prior Beta(2, 2) on beta (profiled offline).
  - s_e: collecting share per ERA (pre-1910 / 1910-40): how readily a present House Finch enters a bird collection relative to
    all birds; absorbs common-species under-collection and era practice. log s_e ~ N(0, 3) (weak).
  - Targeting: collectors went where birds of interest were; conditioning on total bird effort E_cd removes WHERE collecting
    happened, not WHAT was taken. A per-collector or per-expedition effect needs specimen-level collector identity for both
    effort and finch records; the per-cell top-collector facet (recordedBy) supports a diagnostic (effective number of
    collectors per cell) and an optional cell-level robustness weight, not a fitted random effect (which would absorb the
    absences it is meant to read).
Returns NLL (jnp), differentiable in N (trace) and the nuisances.
"""
import json, pickle
from pathlib import Path
import numpy as np
import jax.numpy as jnp

G = Path('/mnt/e/Datasets/houfin/external/gbif_hofi_1800_1940')
DECADES = [(1880, 1901), (1902, 1909), (1910, 1919), (1920, 1929), (1930, 1940)]
PRESENCE_K = 0.43724203275856455


def cell_index_of_points(lat_pts, lon_pts):
    """lat/lon -> land index on the 54-km grid (as hull_specimens.py)."""
    import rasterio
    from rasterio.warp import transform
    m = pickle.load(open('/mnt/e/Datasets/houfin/grid54/inputs_shape/established1951_vl/metadata.pkl', 'rb'))
    lr, lc = np.asarray(m['land_rows']), np.asarray(m['land_cols'])
    lut = -np.ones((int(lr.max()) + 1, int(lc.max()) + 1), int); lut[lr, lc] = np.arange(len(lr))
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as s:
        tr, crs = s.transform, s.crs
    px, py = transform('EPSG:4326', crs, list(lon_pts), list(lat_pts))
    rr, cc = rasterio.transform.rowcol(tr, px, py); rr, cc = np.asarray(rr), np.asarray(cc)
    ok = (rr >= 0) & (rr < lut.shape[0]) & (cc >= 0) & (cc < lut.shape[1])
    out = np.full(len(rr), -1); out[ok] = lut[rr[ok], cc[ok]]
    return out


def build_cache(years, max_unc_m=25000, effort_file='effort_aves_by_year_1880_1940.json', min_neff_absence=0.0, presence_only=False):
    """min_neff_absence: absences count only where the cell's effective number of collectors (inverse Simpson over the
    recordedBy facet) >= this (robustness to single-collector targeting); presence_only: drop all absences (comparison)."""
    import pandas as pd
    eff = json.loads((G / effort_file).read_text())
    df = pd.read_csv(G / 'records.csv')
    df = df[(df.year >= 1880) & (df.year <= 1940) & (df.basisOfRecord == 'PRESERVED_SPECIMEN')]
    unc = pd.to_numeric(df.coordinateUncertaintyInMeters, errors='coerce'); df = df[~(unc > max_unc_m)]
    hc = cell_index_of_points(df.decimalLatitude.values, df.decimalLongitude.values); hy = df.year.values.astype(int)
    yrs = np.asarray(years).astype(int)
    rows = []
    neff_cell = {}
    for k, v in eff.items():
        rec = np.array(list(v.get('rec', {}).values()), float)
        neff_cell[int(k)] = (1. / np.sum((rec / rec.sum()) ** 2)) if rec.sum() > 0 else 0.
    for k, v in eff.items():
        c = int(k); yc = {int(a): b for a, b in v['year'].items()}
        for di, (a, b) in enumerate(DECADES):
            E = sum(n for yy, n in yc.items() if a <= yy <= b)
            if E <= 0:
                continue
            y = int(((hc == c) & (hy >= a) & (hy <= b)).sum())
            t0, t1 = max(a, yrs[0]), max(b, yrs[0])
            rows.append((c, di, E, y, int(np.flatnonzero(yrs == t0)[0]), int(np.flatnonzero(yrs == t1)[0])))
    R = np.asarray(rows)
    # finch records in cells / decades WITHOUT any effort entry (e.g. outside the queried domain) are reported, not used
    T = len(yrs)
    tmask = np.zeros((len(R), T));
    for i, (_, _, _, _, t0, t1) in enumerate(R):
        tmask[i, t0:t1 + 1] = 1. / (t1 - t0 + 1)
    wt = np.ones(len(R))
    absn = R[:, 3] == 0
    if presence_only:
        wt[absn] = 0.
    elif min_neff_absence > 0:
        wt[absn & (np.array([neff_cell[c] for c in R[:, 0]]) < min_neff_absence)] = 0.
    return dict(w=jnp.asarray(wt), neff=np.array([neff_cell[c] for c in R[:, 0]]), cell=jnp.asarray(R[:, 0]), dec=jnp.asarray(R[:, 1]), E=jnp.asarray(R[:, 2].astype(float)), y=jnp.asarray(R[:, 3]),
                pos=jnp.asarray((R[:, 3] > 0).astype(float)), tmask=jnp.asarray(tmask), era=jnp.asarray((R[:, 1] >= 2).astype(int)),
                n=len(R), raw=R)


def g_detect(N, k=PRESENCE_K):
    return -jnp.expm1(-k * jnp.log1p(jnp.maximum(N, 0.) / k))


def specimen_nll_counts(N_land, cache, log_s, beta, log_phi, k=PRESENCE_K, return_rows=False):
    """COUNT version (2026-11-08): the number of finch specimens per cell-decade ~ NB2(mean lam, size phi), with the SAME
    lam = s_era E^beta g(N) whose Poisson zero-probability the Bernoulli form uses (that form is this one dichotomised at
    phi -> inf). Collectors take birds in series, hence the overdispersion phi = exp(log_phi). Adds within-era RELATIVE
    abundance information the presence / absence form discards; absolute density is still confounded with s_era."""
    Ncd = jnp.sum(cache['tmask'] * N_land[:, cache['cell']].T, axis=1)
    ls = jnp.asarray(log_s)
    ls_row = ls[cache['era']] if ls.ndim else ls
    lam = jnp.maximum(jnp.exp(ls_row + beta * jnp.log(cache['E'])) * g_detect(Ncd, k), 1e-12)
    phi = jnp.exp(log_phi); y = jnp.asarray(cache['y'], float)
    from jax.scipy.special import gammaln
    ll = (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1.) + phi * jnp.log(phi / (phi + lam)) + y * jnp.log(lam / (phi + lam)))
    ll = cache['w'] * ll
    return (-jnp.sum(ll), -ll, Ncd) if return_rows else -jnp.sum(ll)


def specimen_nll(N_land, cache, log_s, beta, k=PRESENCE_K, return_rows=False):
    """N_land: (T, N_land) modelled cell totals on the route-count scale. log_s: (2,) per era (pre-1910, 1910-40) or scalar.
    beta: effort exponent. Returns the Bernoulli NLL over cell-decades."""
    Ncd = jnp.sum(cache['tmask'] * N_land[:, cache['cell']].T, axis=1)          # decade-mean cell total per row
    ls = jnp.asarray(log_s)
    ls_row = ls[cache['era']] if ls.ndim else ls
    lam = jnp.exp(ls_row + beta * jnp.log(cache['E'])) * g_detect(Ncd, k)
    lp_pos = jnp.log(jnp.maximum(-jnp.expm1(-lam), 1e-300)); lp_neg = -lam
    ll = cache['w'] * (cache['pos'] * lp_pos + (1. - cache['pos']) * lp_neg)
    return (-jnp.sum(ll), -ll, Ncd) if return_rows else -jnp.sum(ll)
