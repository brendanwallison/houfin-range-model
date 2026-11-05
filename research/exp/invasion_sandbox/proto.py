"""Fast invasion prototypes on uniform 1-D or 2-D grids (numpy, float64).

Two annual steps:
- vl_step: Veit & Lewis 1996 (Am Nat 148:255) eq. 16 exactly -- pairs N^2/(r+2N), nest sites delta,
  dispersing fraction p(N) = min(N/N2*, 1), separate adult / juvenile Weibull kernels.
- hofi_step: the age model's yearling life cycle (age_forward.py step: departure on N/K before
  survival, adult kernel, juvenile kernel x journey survival Q, F = Fmax / (1 + c (N/K)^power) x
  clumped mate-finding 1 - (1 + gamma N / k)^-k, c = Fmax Sj / (1 - Sa) - 1).

Mechanism cards enter through `mech` (a dict of optional switches); every switch off reproduces
the plain step exactly. Densities are birds / km^2; distances km; one step = one year.
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammainc, gamma as gamma_fn

# ---------------------------------------------------------------- grids and kernels

def grid(ndim, dx, extent):
    x = np.arange(-extent, extent + dx / 2, dx)
    if ndim == 1:
        return dict(ndim=1, dx=dx, x=x, r=np.abs(x), cell=dx)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return dict(ndim=2, dx=dx, x=x, r=np.hypot(X, Y), cell=dx * dx)


def weibull_kernel(g, a, b, stay=0.0, sub=4):
    """Cell-integrated Weibull dispersal kernel, mass 1, with a point mass `stay` at zero.

    1-D: density a / (2 b G(1/a)) exp(-(|z|/b)^a) (Veit & Lewis eq. 17), integrated exactly per cell.
    2-D: the SAME distance distribution, f(r) = a / (b G(1/a)) exp(-(r/b)^a) (mean b G(2/a) / G(1/a):
    329 / 98 km for the paper's juvenile / adult fits), spread isotropically; cell masses from
    sub x sub point sampling of f(r) / (2 pi r), the central cell given the exact mass below dx/2.
    """
    dx = g['dx']
    if g['ndim'] == 1:
        x = g['x']; lo = np.maximum(np.abs(x) - dx / 2, 0.); hi = np.abs(x) + dx / 2
        m = .5 * (gammainc(1 / a, (hi / b) ** a) - gammainc(1 / a, (lo / b) ** a))
        c = len(x) // 2; m[c] *= 2
    else:
        x = g['x']; off = (np.arange(sub) + .5) / sub - .5
        m = np.zeros_like(g['r'])
        for ox in off:
            for oy in off:
                X = x[:, None] + ox * dx; Y = x[None, :] + oy * dx
                r = np.maximum(np.hypot(X, Y), dx / (4 * sub))
                f_r = a / (b * gamma_fn(1 / a)) * np.exp(-(r / b) ** a)  # pdf of distance
                m += f_r / (2 * np.pi * r)
        c = (len(x) // 2, len(x) // 2)
        # the central cell holds the distance mass below dx/2 exactly
        m[c] = 0.
        inner = gammainc(1 / a, ((dx / 2) / b) ** a)
        m = m / m.sum() * (1 - inner); m[c] = inner
    m = m / m.sum()
    k = (1 - stay) * m
    k[c] += stay
    return k


def weibull_mean(a, b):
    """Mean distance under the Veit & Lewis form exp(-(r/b)^a) (paper: 329 / 98 km)."""
    return b * gamma_fn(2 / a) / gamma_fn(1 / a)


def convolve(field, kernel):
    out = fftconvolve(field, kernel, mode='same')
    return np.maximum(out, 0.)


# ---------------------------------------------------------------- release and readouts

def release_patch(g, density, radius):
    """Initial density `density` within `radius` of the origin (cell-integrated in 1-D)."""
    if g['ndim'] == 1:
        x = g['x']; dx = g['dx']
        overlap = np.maximum(np.minimum(x + dx / 2, radius) - np.maximum(x - dx / 2, -radius), 0.)
        return density * overlap / dx
    return np.where(g['r'] <= max(radius, g['dx'] / 2), density, 0.)


def readout(g, N, thr):
    occ = N >= thr
    if g['ndim'] == 1:
        rad = float(g['r'][occ].max()) if occ.any() else 0.
        area = float(occ.sum() * g['dx'])
    else:
        area = float(occ.sum() * g['cell'])
        rad = float(g['r'][occ].max()) if occ.any() else 0.
    r90 = float(np.percentile(g['r'][occ], 90)) if occ.any() else 0.
    r_area = float(np.sqrt(area / np.pi)) if g['ndim'] == 2 else area / 2
    return dict(radius_km=rad, r90=r90, r_area=r_area, occupied=area, total=float(N.sum() * g['cell']),
                core=float(N.flat[N.size // 2]))


# ---------------------------------------------------------------- Veit & Lewis 1996

def vl_params(r=.15, c=1.5, s=.67, delta=1.5):
    b = (c / (1 - s) - 2) * delta
    K = .5 * (b + np.sqrt(b * b - 4 * r * delta)) if b * b > 4 * r * delta else np.nan
    return dict(r=r, c=c, s=s, delta=delta, K=K)


def vl_births(N, P):
    den = P['r'] + 2 * N + N * N / P['delta']
    return np.where(N > 0, P['c'] * N * N / np.where(den > 0, den, 1.), 0.)


def kernel_mean_distance(g, k, stay=0.):
    """Mean displacement of movers (excluding the point mass at zero) -- checks discretisation."""
    m = k.copy(); c = np.unravel_index(np.argmax(g['r'] == 0), g['r'].shape) if g['ndim'] == 2 else len(k) // 2
    m[c] -= stay
    return float((m * g['r']).sum() / m.sum())


def vl_step(N, P, kA, kJ, mech, t):
    p = departure(N / P['K'], mech)
    adult = P['s'] * N; young = vl_births(N, P)
    moved = convolve(adult * p, kA) + convolve(young * p, kJ)
    return adult * (1 - p) + young * (1 - p) + moved


# ---------------------------------------------------------------- age-model yearling cycle

def hofi_params(Sa=.55, Sj=.30, Fmax=6., K=4., n50=.05, kmate=3., power=1., Q=1.):
    """n50: density (birds/km^2) at which about half of breeders are mated (gamma = ln 2 / n50)."""
    c = Fmax * Sj / (1 - Sa) - 1
    return dict(Sa=Sa, Sj=Sj, Fmax=Fmax, K=K, n50=n50, kmate=kmate, power=power, Q=Q, c=max(c, 0.))


def mated_fraction(N, P):
    if not np.isfinite(P['n50']) or P['n50'] <= 0:
        return np.ones_like(N)
    x = np.log(2) / P['n50'] * N
    return 1 - (1 + x / P['kmate']) ** (-P['kmate'])


def departure(x, mech):
    form = mech.get('departure', 'veit')
    if form == 'veit':
        return np.minimum(x, 1.)
    if form == 'power':
        return np.minimum(x, 1.) ** mech.get('theta', 1.)
    if form == 'const':
        return np.full_like(x, mech.get('p', .5))
    raise ValueError(form)


def hofi_step(state, P, kA, kJ, mech, t):
    """state = (A, J): post-breeding census of adults and this year's young."""
    A, J = state
    Ntot = A + J
    p = departure(Ntot / P['K'], mech)
    kJt = kJ
    if 'kernel_mix' in mech:  # B1 ramp: mix of a short (base) and long juvenile kernel by year
        w = mech['kernel_mix'](t); kJt = (1 - w) * kJ + w * mech['kJ_long']
    A_stay = A * (1 - p); A_move = convolve(A * p, kA)
    J_stay = J * (1 - p); J_move = convolve(J * p, kJt)
    if 'establish' in mech:  # A2: arrivals into unoccupied cells establish with prob 1 - exp(-m / b)
        J_move, A_move = establish(J_move, A_move, A_stay + J_stay, mech['establish'], mech['cell_area'])
    D = A_stay + A_move + J_stay + J_move  # post-dispersal, pre-survival (as in the model)
    B = P['Sa'] * (A_stay + A_move) + P['Sj'] * J_stay + P['Q'] * P['Sj'] * J_move
    Fx = P['Fmax'] / (1 + P['c'] * (D / P['K']) ** P['power']) * mated_fraction(D, P)
    if 'local_sink' in mech:  # A5: fecundity penalty in recently colonised cells
        d0, tau = mech['local_sink']; age = mech['_age']
        Fx = Fx * (1 - d0 * np.exp(-age / tau))
    young = B * Fx
    if 'floor' in mech:  # A2 variant: density floor
        eps = mech['floor']; B = np.where(B + young < eps, 0., B); young = np.where(B > 0, young, 0.)
    if 'local_sink' in mech:
        occ = (B + young) >= mech.get('age_thr', 1e-3)
        mech['_age'] = np.where(occ, mech['_age'] + 1, 0.)
    return B, young


def establish(J_move, A_move, resident, b, cell_area):
    """Arrivals in cells with no resident birds succeed as a group with probability 1-exp(-m/b),
    m = expected arriving birds in the cell; failed arrivals die (mean-field closure)."""
    empty = resident * cell_area < 1e-9
    m = (J_move + A_move) * cell_area
    pe = np.where(empty, -np.expm1(-m / b), 1.)
    return J_move * pe, A_move * pe


# ---------------------------------------------------------------- stochastic twin (A3/A6)

def hofi_step_stochastic(state, P, kA, kJ, mech, t, rng, cell_area):
    """Integer birds per cell, same cycle and same mean rates as hofi_step.

    Departure and survival are binomial; arrivals in each cell are Poisson with the kernel-convolved mean (independent
    arrivals) or, with mech['group'] = b, Poisson(mean / b) groups of b birds (clustered arrivals, same mean); births
    Poisson(B F(D)); no breeding in a cell with fewer than two breeders (pairs). Densities enter F and Allee as
    birds / cell_area, exactly as in the deterministic step.
    """
    A, J = state
    Ntot = (A + J) / cell_area
    p = departure(Ntot / P['K'], mech)
    A_mv = rng.binomial(A, p); J_mv = rng.binomial(J, p)
    A_st, J_st = A - A_mv, J - J_mv
    kJt = kJ
    if 'kernel_mix' in mech:
        w = mech['kernel_mix'](t); kJt = (1 - w) * kJ + w * mech['kJ_long']
    lamA = convolve(A_mv.astype(float), kA); lamJ = convolve(J_mv.astype(float), kJt)
    b = mech.get('group')
    if b:
        A_in = rng.poisson(lamA / b) * b; J_in = rng.poisson(lamJ / b) * b
    else:
        A_in = rng.poisson(lamA); J_in = rng.poisson(lamJ)
    D = (A_st + A_in + J_st + J_in) / cell_area
    B = rng.binomial(A_st + A_in, P['Sa']) + rng.binomial(J_st, P['Sj']) + rng.binomial(J_in, P['Q'] * P['Sj'])
    Fx = P['Fmax'] / (1 + P['c'] * (D / P['K']) ** P['power']) * mated_fraction(D, P)
    young = rng.poisson(np.where(B >= 2, B * Fx, 0.))
    return B, young


def simulate_stochastic(P, g, kA, kJ, years=51, N0=None, thr=.1, mech=None, seed=0):
    rng = np.random.default_rng(seed); mech = dict(mech or {})
    cell = g['cell']
    A = rng.poisson(.5 * N0 * cell); J = rng.poisson(.5 * N0 * cell)
    hist = []
    for t in range(years):
        hist.append(dict(year=t, **readout(g, (A + J) / cell, thr)))
        A, J = hofi_step_stochastic((A, J), P, kA, kJ, mech, t, rng, cell)
    return hist


# ---------------------------------------------------------------- simulation driver

def simulate(model, P, g, kA, kJ, years=81, N0=None, thr=.1, mech=None, keep=()):
    mech = dict(mech or {}); mech.setdefault('cell_area', g['cell'])
    if 'local_sink' in mech:
        mech['_age'] = np.zeros_like(N0)
    hist = []; snaps = {}
    if model == 'vl':
        N = N0.copy()
        for t in range(years):
            hist.append(dict(year=t, **readout(g, N, thr)))
            if t in keep: snaps[t] = N.copy()
            N = vl_step(N, P, kA, kJ, mech, t)
    else:
        A = .5 * N0; J = .5 * N0
        for t in range(years):
            N = A + J
            hist.append(dict(year=t, **readout(g, N, thr)))
            if t in keep: snaps[t] = N.copy()
            A, J = hofi_step((A, J), P, kA, kJ, mech, t)
    return hist, snaps


def lag_and_speeds(hist, lag_km=100., early=(5, 20), late=(30, 50), key='radius_km'):
    yrs = np.array([h['year'] for h in hist]); rad = np.array([h[key] for h in hist])
    above = np.nonzero(rad > lag_km)[0]
    lag = float(yrs[above[0]]) if len(above) else None
    def speed(a, b):
        if b >= len(rad): return None
        return float((rad[b] - rad[a]) / (b - a))
    return dict(lag_year=lag, early_speed=speed(*early), late_speed=speed(*late),
                radius_at={int(y): float(rad[y]) for y in (10, 15, 20, 25, 30, 40, 50) if y < len(rad)})
