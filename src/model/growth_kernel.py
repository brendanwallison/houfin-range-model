"""population_model.growth_kernel: a DATA-DRIVEN local growth reference (research prototype, default OFF).

The mechanistic local update (density-braked adult survival Sa / (1 + c_s crowd(N/K)), Beverton-Holt fecundity
Fmax / (1 + c crowd(N/K)), absolute aggregated Allee mate finding, K) is replaced by two LEARNED functions of habitat Z
and current density, with an explicit Z x density interaction (a varying-coefficient / ANOVA product kernel):

    g_f(z, u) = a_f + x(z).beta_f0                          habitat main effect (the rate as density -> 0)
              + sum_p c_fp phi_p(u)                         density main effect (one shape everywhere)
              + sum_p ((z - z_c).beta_fp) phi_p(u)          PRODUCT interaction: habitat modulates the density response
             [+ sum_q d_fq psi_q(u_tot - k(z))]             optional RELATIVE axis (relative_axis = true): a learned
                                                            habitat density scale k(z) = k0 + (z - z_c).kappa, shared by
                                                            s and r; kappa = 0 removes the habitat shift
    adult survival   s   = sigmoid(g_s)
    births / breeder rho = exp(softcap(g_r))                (softcap = smooth ceiling at log(rho_cap))

Why the optional relative axis: a product kernel linear in z can shift a density response along u only to FIRST order
(f(u) + k(z) f'(u)); the mechanistic brake acts on N/K with K varying ~100-fold over habitat, i.e. a shift of several
knot intervals, which the product kernel cannot carry (an additive or product start from a fitted mechanistic model
explodes -- see research/exp/invasion_sandbox/growth_kernel_init.py). psi(u - k(z)) carries a habitat-scaled density
response EXACTLY while k, the shape d and the product terms stay free; kappa = 0, d = 0 nests the product kernel.

phi_p / psi_q are clamped cubic B-splines with the first function DROPPED: every column is exactly 0 at the lower end of
its range, so with N -> 0 the absolute terms vanish (u -> u_min) and g_f -> a_f + x(z).beta_f0 (+ the relative terms at
u_min - k(z), which vanish when k(z) >= u_min - rel_min). Density input (per land cell, the post-dispersal pre-breeding
pools the mechanistic update uses):
    stage: u_a = log(N_a_post + e^u_min), u_j = log(N_j_post + e^u_min), phi = [phi(u_a), phi(u_j)]  (additive in age)
    total: u   = log(N_a_post + N_j_post + e^u_min)
    relative axis: u_tot - k(z), u_tot = log(N_a_post + N_j_post + e^u_min), clipped to [rel_min, rel_max]
Inputs are clipped at the top (flat extrapolation above the range).

Priors (non-centred, so the MAP has no variance funnel): a_f ~ N(loc_f, a_scale); beta_f0 = w_f0 b_f0, b ~ N(0, 1),
w_f0 = softplus(N(main_amp_loc, 1)); c_fp ~ N(0, c_scale); beta_fp = w_fi B_fp, B ~ N(0, 1), w_fi = softplus(N(int_amp_loc,
1)) -- the data decide how much product interaction exists (w_fi -> 0 is the additive kernel f(Z) + h(N)); relative axis:
k0 ~ N(k0_loc, k0_scale), kappa = w_k b_k, b_k ~ N(0, 1), w_k = softplus(N(0, 1)), d_fq ~ N(0, c_scale).

What stays mechanistic: juvenile overwinter survival Sj (stayers) and journey survival Q (dispersers) -- both are
density-independent and Q is part of the dispersal step, which must stay exactly as it is -- so potential growth on
the breeder census is lambda = s + Sj rho (yearling breeding). Dispersal (kernels, departure on N/K with the mechanistic
K field), disease added mortality, the native seed and the observation model are untouched.

The forward model receives growth_kernel = (eta_s (T, L), A_s (T, L, P), eta_r, A_r, k (T, L) | None, d_s (Q,) | None,
d_r (Q,) | None, spec).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import numpyro
import numpyro.distributions as dist

DEFAULTS = dict(
    enabled=False,
    n_basis=8,              # B-spline functions per density axis (first dropped -> n_basis - 1 active)
    density_input="stage",  # "stage" (additive in log adults, log juveniles) | "total"
    u_min=math.log(1e-5),   # log model density; 1e-5 units = 2e-4 route counts (below the Allee transition)
    u_max=math.log(10.0),   # above the observed max (~5.8 units at cp3)
    features="z",           # main effect on "z" (Z_t - z_c) | "split" ([Z_ref - z_c, Z_t - Z_ref], as readout.split)
    a_s_loc=-0.5, a_r_loc=0.0, a_scale=3.0,
    main_amp_loc=0.0,       # softplus(0) = 0.69, as manifold_prior amplitudes
    c_scale=3.0,
    int_amp_loc=-1.0,       # softplus(-1) = 0.31: product-interaction amplitude prior median
    rho_cap=50.0,           # smooth ceiling on births per breeder
    relative_axis=False,
    rel_min=-6.0, rel_max=4.0,   # log(N / e^k) range of the relative axis
    k0_loc=math.log(0.3), k0_scale=3.0,
)


def spec_from_config(cfg: dict | None) -> dict:
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (cfg or {}).items() if not k.startswith("_")})
    s["n_basis"] = int(s["n_basis"])
    for k in ("u_min", "u_max", "rel_min", "rel_max"):
        s[k] = float(s[k])
    s["relative_axis"] = bool(s["relative_axis"])
    if s["density_input"] not in ("stage", "total"):
        raise ValueError(f"growth_kernel.density_input must be 'stage' or 'total', got {s['density_input']!r}")
    if s["features"] not in ("z", "split"):
        raise ValueError(f"growth_kernel.features must be 'z' or 'split', got {s['features']!r}")
    if s["n_basis"] < 4:
        raise ValueError("growth_kernel.n_basis must be >= 4 (cubic B-splines)")
    return s


def n_density_terms(spec) -> int:
    return (spec["n_basis"] - 1) * (2 if spec["density_input"] == "stage" else 1)


def n_relative_terms(spec) -> int:
    return spec["n_basis"] - 1 if spec["relative_axis"] else 0


def _knots(u_min, u_max, n_basis, degree=3):
    n_int = n_basis - degree - 1
    inner = np.linspace(u_min, u_max, n_int + 2)
    return np.concatenate([[u_min] * degree, inner, [u_max] * degree])


def bspline_basis(u, u_min, u_max, n_basis, degree=3, drop_first=True):
    """Clamped uniform B-spline basis, (..., n_basis[-1]). u is clipped to [u_min, u_max]; Cox-de Boor recursion
    (static loops; differentiable inside each knot interval)."""
    t = _knots(u_min, u_max, n_basis, degree)
    span = u_max - u_min
    u = jnp.clip(u, u_min, u_max - 1e-9 * span)
    B = [jnp.where((u >= t[i]) & (u < t[i + 1]), 1.0, 0.0).astype(u.dtype) for i in range(len(t) - 1)]
    for p in range(1, degree + 1):
        nb = []
        for i in range(len(t) - 1 - p):
            d1, d2 = t[i + p] - t[i], t[i + p + 1] - t[i + 1]
            term = jnp.zeros_like(u)
            if d1 > 0:
                term = term + (u - t[i]) / d1 * B[i]
            if d2 > 0:
                term = term + (t[i + p + 1] - u) / d2 * B[i + 1]
            nb.append(term)
        B = nb
    out = jnp.stack(B[:n_basis], axis=-1)
    return out[..., 1:] if drop_first else out


def density_basis(Na, Nj, spec):
    """phi (..., P) from post-dispersal adult / juvenile densities (any matching shapes)."""
    e = math.exp(spec["u_min"])
    bb = lambda u: bspline_basis(u, spec["u_min"], spec["u_max"], spec["n_basis"])
    if spec["density_input"] == "stage":
        return jnp.concatenate([bb(jnp.log(jnp.maximum(Na, 0.0) + e)), bb(jnp.log(jnp.maximum(Nj, 0.0) + e))], axis=-1)
    return bb(jnp.log(jnp.maximum(Na + Nj, 0.0) + e))


def relative_basis(Na, Nj, k, spec):
    """psi (..., Q) on log total density relative to the habitat density scale k (same shape as Na)."""
    u = jnp.log(jnp.maximum(Na + Nj, 0.0) + math.exp(spec["u_min"]))
    return bspline_basis(u - k, spec["rel_min"], spec["rel_max"], spec["n_basis"])


def softcap(x, cap, sharpness=5.0):
    """Smooth ceiling: ~x well below cap (error e^{-5 (cap - x)} / 5: < 1e-7 at rho = cap / 25), -> cap above."""
    return cap - jnn.softplus(sharpness * (cap - x)) / sharpness


def rates_at(gk_t, Na, Nj):
    """(s, rho) for one year's coefficients gk_t = (eta_s, A_s, eta_r, A_r, k | None, d_s, d_r, spec) at densities
    Na, Nj (shapes matching eta)."""
    eta_s, A_s, eta_r, A_r, k, d_s, d_r, spec = gk_t
    phi = density_basis(Na, Nj, spec)
    g_s = eta_s + jnp.sum(A_s * phi, axis=-1)
    g_r = eta_r + jnp.sum(A_r * phi, axis=-1)
    if k is not None:
        psi = relative_basis(Na, Nj, k, spec)
        g_s = g_s + psi @ d_s
        g_r = g_r + psi @ d_r
    return jnn.sigmoid(g_s), jnp.exp(softcap(g_r, math.log(float(spec["rho_cap"]))))


def year_slice(gk, t):
    """gk tuple at year t (the per-year arrays indexed, constants passed through)."""
    eta_s, A_s, eta_r, A_r, k, d_s, d_r, spec = gk
    return (eta_s[t], A_s[t], eta_r[t], A_r[t], None if k is None else k[t], d_s, d_r, spec)


def sample_growth_kernel(spec, M, prior_scale=1.0):
    """NumPyro sites (names gk_*). Returns {f: (a, beta0, c, beta_int)} for f in s, r, plus 'rel': (k0, kappa, d_s, d_r)
    or None."""
    P = n_density_terms(spec)
    M_main = 2 * M if spec["features"] == "split" else M
    out = {}
    for f, loc in (("s", spec["a_s_loc"]), ("r", spec["a_r_loc"])):
        a = numpyro.sample(f"gk_{f}_a0", dist.Normal(float(loc), float(spec["a_scale"])))
        w0 = numpyro.deterministic(f"gk_{f}_w0", jnn.softplus(numpyro.sample(
            f"gk_{f}_w0_raw", dist.Normal(float(spec["main_amp_loc"]), 1.0))))
        b0 = numpyro.sample(f"gk_{f}_b0", dist.Normal(0.0, 1.0).expand([M_main]).to_event(1))
        c = numpyro.sample(f"gk_{f}_c", dist.Normal(0.0, float(spec["c_scale"])).expand([P]).to_event(1))
        wi = numpyro.deterministic(f"gk_{f}_wi", jnn.softplus(numpyro.sample(
            f"gk_{f}_wi_raw", dist.Normal(float(spec["int_amp_loc"]), 1.0))))
        bi = numpyro.sample(f"gk_{f}_bi", dist.Normal(0.0, 1.0).expand([P, M]).to_event(2))
        out[f] = (a, w0 * b0, c, wi * bi)
    out["rel"] = None
    if spec["relative_axis"]:
        Q = n_relative_terms(spec)
        k0 = numpyro.sample("gk_k0", dist.Normal(float(spec["k0_loc"]), float(spec["k0_scale"])))
        wk = numpyro.deterministic("gk_wk", jnn.softplus(numpyro.sample("gk_wk_raw", dist.Normal(0.0, 1.0))))
        bk = numpyro.sample("gk_bk", dist.Normal(0.0, 1.0).expand([M]).to_event(1))
        d_s = numpyro.sample("gk_s_d", dist.Normal(0.0, float(spec["c_scale"])).expand([Q]).to_event(1))
        d_r = numpyro.sample("gk_r_d", dist.Normal(0.0, float(spec["c_scale"])).expand([Q]).to_event(1))
        out["rel"] = (k0, wk * bk, d_s, d_r)
    return out


def coefficient_fields(Z_gathered, coefs, spec, z_center=None, Z_ref=None):
    """The forward model's growth_kernel tuple: per (time, land cell) eta_f (T, L), A_f (T, L, P), k (T, L) | None."""
    M = Z_gathered.shape[-1]
    zc = jnp.zeros(M) if z_center is None else jnp.asarray(z_center)
    split = spec["features"] == "split"
    if split and Z_ref is None:
        raise ValueError("growth_kernel.features = split needs data['Z_ref'] (readout.split inputs)")
    (a_s, b0_s, c_s, bi_s), (a_r, b0_r, c_r, bi_r) = coefs["s"], coefs["r"]
    rel = coefs.get("rel")
    dt = jnp.result_type(b0_s.dtype, jnp.float32)
    if split:
        lvl_s = jnp.dot(Z_ref - zc, b0_s[:M])
        lvl_r = jnp.dot(Z_ref - zc, b0_r[:M])

    def year(t):
        z = jnp.take(Z_gathered, t, axis=0).astype(dt)
        zc_t = z - zc
        if split:
            dz = z - Z_ref
            eta_s = a_s + lvl_s + jnp.dot(dz, b0_s[M:])
            eta_r = a_r + lvl_r + jnp.dot(dz, b0_r[M:])
        else:
            eta_s = a_s + jnp.dot(zc_t, b0_s)
            eta_r = a_r + jnp.dot(zc_t, b0_r)
        A_s = c_s[None, :] + zc_t @ bi_s.T
        A_r = c_r[None, :] + zc_t @ bi_r.T
        k = (rel[0] + jnp.dot(zc_t, rel[1])) if rel is not None else jnp.zeros_like(eta_s)
        return eta_s, A_s, eta_r, A_r, k
    eta_s, A_s, eta_r, A_r, k = jax.lax.map(year, jnp.arange(Z_gathered.shape[0]))
    if rel is None:
        return (eta_s, A_s, eta_r, A_r, None, None, None, spec)
    return (eta_s, A_s, eta_r, A_r, k, rel[2], rel[3], spec)


def low_density_rates(gk):
    """(s0, rho0) as N -> 0, (T, L): the potential (low-density) rates."""
    z = jnp.zeros_like(gk[0])
    return rates_at(gk, z, z)


# ---------------------------------------------------------------------------------------------------------------
# Readouts (outside the model; numpy/jax, chunk by year): potential growth, the data's implied equilibrium, sinks.
# ---------------------------------------------------------------------------------------------------------------

def lambda_on_ray(gk_t, Sj, B, n_fixed=8):
    """Growth on the breeder census along the equilibrium stage ray: breeders B (pre-breeding adults present), juveniles
    present rho(B) B (the yearlings the same breeders produced; exact AT an equilibrium). rho solved by fixed point.
    gk_t: one year's (or any leading shape's) coefficients, eta (...), A (..., P), k (...); Sj (...), B (G,)
    -> lambda, s, rho, each (..., G)."""
    eta_s, A_s, eta_r, A_r, k, d_s, d_r, spec = gk_t
    ex = lambda x: None if x is None else x[..., None]
    g = (ex(eta_s), A_s[..., None, :], ex(eta_r), A_r[..., None, :], ex(k), d_s, d_r, spec)
    Bb = jnp.broadcast_to(B, eta_s.shape + B.shape)
    rho = jnp.ones_like(Bb)
    for _ in range(n_fixed):
        s, rho = rates_at(g, Bb, rho * Bb)
    return s + Sj[..., None] * rho, s, rho


def equilibrium_readout(gk_t, Sj, n_grid=80):
    """Per cell: lambda0 (N -> 0), lambda_peak over density, N* (largest stable equilibrium, TOTAL density B(1 + rho),
    model units), Allee threshold (lowest unstable crossing) and status codes:
    0 = stable positive equilibrium inside the basis range, 1 = no positive equilibrium (lambda < 1 at every density),
    2 = lambda > 1 at the top of the range (equilibrium above u_max: unbounded within the observed range)."""
    spec = gk_t[-1]
    B = jnp.exp(jnp.linspace(spec["u_min"], spec["u_max"], n_grid))
    lam, s, rho = lambda_on_ray(gk_t, Sj, B)
    z = jnp.zeros_like(gk_t[0])
    s0, r0 = rates_at(gk_t, z, z)
    lam0 = s0 + Sj * r0
    above = lam > 1.0
    down = above[..., :-1] & ~above[..., 1:]          # stable crossings (lambda falls through 1)
    up = ~above[..., :-1] & above[..., 1:]            # unstable (Allee) crossings
    idx = jnp.arange(n_grid - 1)
    last_down = jnp.max(jnp.where(down, idx, -1), axis=-1)
    first_up = jnp.min(jnp.where(up, idx, n_grid), axis=-1)

    def interp(i):          # log-linear interpolation of the crossing between grid i and i + 1, on total density
        i = jnp.clip(i, 0, n_grid - 2)
        l0, l1 = jnp.take_along_axis(lam, i[..., None], -1)[..., 0], jnp.take_along_axis(lam, i[..., None] + 1, -1)[..., 0]
        w = jnp.clip((l0 - 1.0) / jnp.where(l0 != l1, l0 - l1, 1.0), 0.0, 1.0)
        tot = B * (1.0 + rho)
        t0, t1 = jnp.take_along_axis(tot, i[..., None], -1)[..., 0], jnp.take_along_axis(tot, i[..., None] + 1, -1)[..., 0]
        return jnp.exp(jnp.log(t0) + w * (jnp.log(t1) - jnp.log(t0)))
    n_star = jnp.where(last_down >= 0, interp(last_down), jnp.nan)
    allee = jnp.where(first_up < n_grid, interp(first_up), jnp.nan)
    status = jnp.where(above[..., -1], 2, jnp.where(jnp.any(above, axis=-1), 0, 1))
    status = jnp.where((status == 0) & (last_down < 0), 2, status)
    return dict(lambda0=lam0, lambda_peak=jnp.max(lam, axis=-1), n_star=n_star, allee=allee, status=status)
