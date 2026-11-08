"""BBL-anchored winter (CBC) source model for the early-CBC count likelihood (agent_levers round 3, 2026-11-07).

Replaces laplace_fit's union source weights max(w_local, w_seasonal) (a fixed 300-km SW allowance) by an explicit
summer -> winter redistribution estimated from BBL summer -> winter recaptures (p14_bbl_sw.py, 290 pairs):
  winter density in cell k = sum_j N_j [ (1 - m_j) delta_jk + m_j P(k | j) ]
  circle expected count   ~ sum_k w_local(circle, k) * winter_k   =>   W(circle, j) = sum_k w_local(c, k) [(1-m_j) d_jk + m_j P(k|j)]
  P(k | j): movers' displacement density over LAND cells, distance ~ LogNormal(log 265 km, .83) and bearing ~ von Mises(218 deg,
            kappa) for EASTERN-origin birds (BBL: n 190, 57% > 80 km, movers' quartiles 159 / 265 / 486 km, bearings S/SW);
            isotropic LogNormal(log 337 km, .83) for western / Plains origins (BBL: few movers, n 6, no consistent bearing).
  m_j: migrant share by breeding region from BBL (share > 80 km): East (lon > -95) .57 (1960s .58), interior West .14,
       Pacific .06, Plains .27 (n 11 -- weak; use the interior-West value .14 unless asked otherwise).
  NO free parameters (an optional single scale on m is provided for a sensitivity / profile, not for fitting).
Also: a global CBC detection drift (feeder / observer culture) gam * (year - 1940) / 10, gam ~ N(0, 1) per decade -- a nuisance
(p13: gam-hat ~ +.28-.34 per decade on RP2 fits, ~0 on y11).
Usage: W = build_source_matrix(cbc_early.npz arrays, land lat/lon) once on the host (n_circle x N_land, float32, ~40 MB), then
A = cbc_source_abundance(N_land_t, W, circle_of_row, t_of_row) in the CBC cost instead of the union sum.
"""
import numpy as np
import jax.numpy as jnp

BBL = dict(east=dict(m=.57, mu=np.log(265.), sd=.83, bearing=218., kappa=None),
           west=dict(m=.14, mu=np.log(337.), sd=.83, bearing=None, kappa=0.),
           pacific=dict(m=.06, mu=np.log(337.), sd=.83, bearing=None, kappa=0.))


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def kappa_from_bearings(b_deg):
    """von Mises concentration from the mean resultant length (Best & Fisher approximation)."""
    b = np.radians(np.asarray(b_deg)); R = np.hypot(np.cos(b).mean(), np.sin(b).mean())
    if R < .53:
        return 2 * R + R ** 3 + 5 * R ** 5 / 6
    if R < .85:
        return -.4 + 1.39 * R + .43 / (1 - R)
    return 1 / (R ** 3 - 4 * R ** 2 + 3 * R)


def movement_matrix(lat, lon, region_of_cell, params=BBL, chunk=512):
    """P (N_land x N_land): row j = winter destination distribution of a MIGRANT from breeding cell j (rows sum to 1)."""
    NL = lat.size; P = np.zeros((NL, NL), np.float32)
    for a in range(0, NL, chunk):
        j = np.arange(a, min(a + chunk, NL))
        d = hav(lat[j][:, None], lon[j][:, None], lat[None, :], lon[None, :])
        dy = lat[None, :] - lat[j][:, None]; dx = (lon[None, :] - lon[j][:, None]) * np.cos(np.radians(lat[j]))[:, None]
        bear = np.degrees(np.arctan2(dx, dy)) % 360
        dd = np.maximum(d, 27.)
        for reg, p in params.items():
            sel = region_of_cell[j] == reg
            if not sel.any():
                continue
            logn = -((np.log(dd[sel]) - p['mu']) ** 2) / (2 * p['sd'] ** 2) - np.log(dd[sel])      # log density in distance
            dens = np.exp(logn) / dd[sel]                                                          # per unit AREA (/ r)
            if p.get('bearing') is not None and p.get('kappa'):
                dens = dens * np.exp(p['kappa'] * np.cos(np.radians(bear[sel] - p['bearing'])))
            dens[d[sel] < 1.] = 0.                                                                 # migrants leave the cell
            P[j[sel]] = (dens / np.maximum(dens.sum(1, keepdims=True), 1e-300)).astype(np.float32)
    return P


def build_source_matrix(cbc, lat, lon, m_scale=1.0, params=None, east_lon=-95., pacific_lon=-117.):
    """cbc: dict with cbc_source_idx (n_circle, S), cbc_w_local (n_circle, S). Returns W (n_circle, N_land) float32."""
    params = dict(BBL if params is None else params)
    if params['east'].get('kappa') is None:
        params['east'] = dict(params['east'], kappa=4.0)
    reg = np.where(lon > east_lon, 'east', np.where(lon < pacific_lon, 'pacific', 'west'))
    P = movement_matrix(lat, lon, reg, params)
    m = np.array([params[r]['m'] for r in reg]) * m_scale
    src = np.asarray(cbc['cbc_source_idx']); wl = np.where(src >= 0, np.asarray(cbc['cbc_w_local']), 0.)
    nC, NL = src.shape[0], lat.size
    WL = np.zeros((nC, NL), np.float64)
    np.add.at(WL, (np.repeat(np.arange(nC), src.shape[1]), np.maximum(src, 0).ravel()), wl.ravel())
    # W = WL [(1 - m) I + diag(m) P]  ->  W[c, j] = WL[c, j] (1 - m_j) + m_j sum_k WL[c, k] P[j, k]
    W = WL * (1. - m)[None, :] + (WL @ P.T.astype(np.float64)) * m[None, :]
    return W.astype(np.float32), dict(m=m, region=reg)


def cbc_source_abundance(N_land, W, circle_of_row, t_of_row):
    """A_row = sum_j N_land[t_row, j] W[circle_row, j]  (jnp; N_land (T, N_land) on the route-count scale)."""
    return jnp.sum(N_land[t_of_row] * W[circle_of_row], axis=1)
