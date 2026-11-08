"""Start values for population_model.growth_kernel from a fitted MECHANISTIC parameter set (run under the FLAG-OFF overlay).

  python growth_kernel_init.py PARAMS.npz GK_OVERLAY.json OUT_PREFIX [--subsample 60000 --grid_rows 20000 --grid_n 12]

1. Replays the mechanistic model at PARAMS and records (ordered debug callbacks), every year and land cell, the
   post-dispersal pools the reproduction step sees (N_a_post, J_stay, J_arrive) and its brakes (c, c_s, K).
2. Mechanistic rates there (disease-free): s = Sa / (1 + c_s crowd(N/K)), births per breeder
   F = Fmax / (1 + c crowd(N/K)) * allee(gamma N), N = total post-dispersal density (+ e^u_min, the learned floor) --
   at every REALISED state and on a density GRID for a random subsample of cell-years (so the start is also right
   off the trajectory: colonisation of empty cells, crashes), each grid row at its cell-year's stage share.
3. GK_OVERLAY's relative_axis: k(z) = least squares of the mechanistic half-brake density log(K / c^(1/power)) on z.
4. Least squares of logit s and log F on the kernel's MAIN effects [1, x(z), phi(u) (, psi(u_tot - k(z)))]
   -> a, beta0, c (, d); product-interaction coefficients 0: OUT_PREFIX_neutral.npz = PARAMS + gk_* sites.
5. For comparison, the same plus the product block (z - z_c) (x) phi(u) on a random subsample of rows, with the ridge
   the interaction prior implies (rmse^2 / softplus(int_amp_loc)^2): OUT_PREFIX_lsfull.npz.
R^2 by density tier (realised states and grid rows separately) is printed and written to OUT_PREFIX_report.json.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402
from harness import Sandbox, ap  # noqa: E402
from src.model import age_forward as af  # noqa: E402
from src.model import growth_kernel as gkm  # noqa: E402
from src.model.density_regulation import CROWDING_POWER, crowding_argument  # noqa: E402


def softplus_inv(y):
    y = max(float(y), 1e-6)
    return y + math.log(-math.expm1(-y))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('params'); p.add_argument('gk_overlay'); p.add_argument('out_prefix')
    p.add_argument('--subsample', type=int, default=60000)
    p.add_argument('--grid_rows', type=int, default=20000)
    p.add_argument('--grid_n', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if ap.GROWTH_KERNEL:
        raise SystemExit('run under the FLAG-OFF overlay (the replay records the mechanistic rates)')
    spec = gkm.spec_from_config(json.load(open(a.gk_overlay))['population_model']['growth_kernel'])
    t0 = time.time()
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        params = {k: np.asarray(z[k]) for k in z.files}
    lat.update({k: jnp.asarray(v) for k, v in params.items() if not k.startswith('laplace_')})

    rec = []
    orig = af.reproduction_age_structured

    def recorder(N_a_post, N_j_stayers, N_j_arrivers, S_a, S_j, F_max, K, c, allee_gamma, **kw):
        cs = kw.get('c_s')
        jax.debug.callback(lambda *xs: rec.append(tuple(np.asarray(x) for x in xs)), N_a_post, N_j_stayers, N_j_arrivers,
                           K, c, (cs if cs is not None else jnp.zeros_like(K)), ordered=True)
        return orig(N_a_post, N_j_stayers, N_j_arrivers, S_a, S_j, F_max, K, c, allee_gamma, **kw)
    af.reproduction_age_structured = recorder
    try:
        tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(
            data=sb.data, prior_scale=1.)
        jax.effects_barrier()
    finally:
        af.reproduction_age_structured = orig
    T = int(sb.data['time']); rows, cols = np.asarray(sb.rows), np.asarray(sb.cols); L = rows.size
    rec = rec[-T:]                                       # the main run (a spin-up, if any, comes first)
    print(json.dumps(dict(stage='replay', calls=len(rec), sec=round(time.time() - t0))), flush=True)
    G = lambda i: np.stack([r[i][rows, cols] for r in rec])        # (T, L)
    Na, Js, Ja, K, c, cs = (G(i) for i in range(6))
    Nj = Js + Ja

    def land(k):
        v = np.asarray(tr[k]['value']).reshape(T, -1)
        return v if v.shape[1] == L else v[:, rows * sb.Nx + cols]
    Sa, Fmax = land('Sa_flat'), land('Fmax_flat')
    gamma = float(tr['allee_gamma']['value'])
    e = math.exp(spec['u_min'])
    n = T * L
    fl = lambda X: np.asarray(X, np.float64).reshape(n)
    Na_f, Nj_f, K_f, c_f, cs_f, Sa_f, Fm_f = (fl(X) for X in (Na, Nj, K, c, cs, Sa, Fmax))

    def mech(Na_, Nj_, idx):     # disease-free mechanistic logit s and log births per breeder at these densities
        Nt_ = np.maximum(Na_ + Nj_, 0.) + e
        r_ = np.asarray(crowding_argument(Nt_ / np.maximum(K_f[idx], 1e-12)))
        s_ = np.clip(Sa_f[idx] / (1. + cs_f[idx] * r_), 1e-9, 1 - 1e-9)
        F_ = Fm_f[idx] / (1. + c_f[idx] * r_) * np.asarray(af.allee_mated_fraction(jnp.asarray(gamma * Nt_)))
        return np.log(s_) - np.log1p(-s_), np.log(np.maximum(F_, 1e-12))
    rng = np.random.default_rng(a.seed)
    S = rng.choice(n, size=min(a.grid_rows, n), replace=False)
    Bg = np.exp(np.linspace(spec['u_min'], spec['u_max'] - 0.5, a.grid_n))
    tot_S = Na_f[S] + Nj_f[S]
    jf = np.clip(np.where(tot_S > 1e-6, Nj_f[S] / np.maximum(tot_S, 1e-300), 0.4), 0.05, 0.95)
    idx = np.concatenate([np.arange(n), np.repeat(S, a.grid_n)])
    Na_all = np.concatenate([Na_f, (Bg[None, :] * (1 - jf[:, None])).ravel()])
    Nj_all = np.concatenate([Nj_f, (Bg[None, :] * jf[:, None]).ravel()])
    real = np.arange(idx.size) < n
    y_s, y_r = mech(Na_all, Nj_all, idx)

    zc = (np.asarray(sb.data['habitat_center_occupied']) if ap._HABITAT_CENTERING == 'occupied'
          else np.zeros(sb.data['Z_gathered'].shape[-1]))
    Zc = (np.asarray(sb.data['Z_gathered'], np.float64) - zc).reshape(n, -1)      # (n, M)
    M = Zc.shape[-1]
    if spec['features'] == 'split':
        Zr = np.broadcast_to(np.asarray(sb.data['Z_ref'], np.float64) - zc, (T, L, M)).reshape(n, M)
        Xm = np.concatenate([Zr, Zc - Zr], 1)
    else:
        Xm = Zc
    rel = None
    if spec['relative_axis']:
        ok = c_f > 1e-6
        lh = np.log(np.maximum(K_f[ok], 1e-12)) - np.log(c_f[ok]) / CROWDING_POWER
        Dk = np.concatenate([np.ones((int(ok.sum()), 1)), Zc[ok]], 1)
        bk = np.linalg.lstsq(Dk, lh, rcond=None)[0]
        kf = bk[0] + Zc @ bk[1:]
        rel = dict(k0=bk[0], kappa=bk[1:], r2=float(1 - np.var(lh - Dk @ bk) / np.var(lh)),
                   range=[float(np.quantile(kf, q)) for q in (.01, .5, .99)])
        print(json.dumps(dict(stage='relative_k', r2=rel['r2'], k_q01_50_99=rel['range'])), flush=True)
    phi = np.asarray(gkm.density_basis(jnp.asarray(Na_all), jnp.asarray(Nj_all), spec))
    P = phi.shape[-1]
    cols_ = [np.ones((idx.size, 1)), Xm[idx], phi]
    if rel is not None:
        cols_.append(np.asarray(gkm.relative_basis(jnp.asarray(Na_all), jnp.asarray(Nj_all), jnp.asarray(kf[idx]), spec)))
    D = np.concatenate(cols_, axis=1); del cols_
    Mm = Xm.shape[-1]
    tier = np.digitize(np.log10(np.maximum(Na_all + Nj_all, 0.) + e), [-4, -2, -1])   # <1e-4, ..1e-2, ..0.1, >0.1 units

    def fit(Dm, y, ridge):
        return np.linalg.solve(Dm.T @ Dm + np.diag(ridge), Dm.T @ y)

    def r2(y, yh, m):
        return float(1 - np.sum((y[m] - yh[m]) ** 2) / max(np.sum((y[m] - y[m].mean()) ** 2), 1e-30))
    out_main, out_full, report = {}, {}, {}
    sub = rng.choice(idx.size, size=min(a.subsample, idx.size), replace=False)
    I = (Zc[idx[sub]][:, :, None] * phi[sub][:, None, :]).reshape(sub.size, M * P)   # (z - z_c) (x) phi, p fastest
    tau_i = float(np.log1p(np.exp(spec['int_amp_loc'])))                                 # prior median interaction amplitude
    for f, y in (('s', y_s), ('r', y_r)):
        rid = np.full(D.shape[1], 1e-6); rid[0] = 0.
        b = fit(D, y, rid); yh = D @ b
        a0, beta0, cc = b[0], b[1:1 + Mm], b[1 + Mm:1 + Mm + P]
        dd = b[1 + Mm + P:] if rel is not None else None
        w0 = max(float(np.sqrt(np.mean(beta0 ** 2))), 1e-3)
        out_main[f] = dict(a0=a0, w0_raw=softplus_inv(w0), b0=beta0 / w0, c=cc,
                           wi_raw=float(spec['int_amp_loc']), bi=np.zeros((P, M)))
        if dd is not None:
            out_main[f]['d'] = dd
        rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
        Df = np.concatenate([D[sub], I], axis=1)
        ridf = np.concatenate([rid, np.full(I.shape[1], rmse ** 2 / tau_i ** 2)])        # MAP-scaled ridge on the product block
        bf = fit(Df, y[sub], ridf)
        yhf_sub = Df @ bf
        kk = D.shape[1]
        bint = bf[kk:].reshape(M, P).T                                                   # (P, M)
        wi = max(float(np.sqrt(np.mean(bint ** 2))), 1e-3)
        bm = bf[:kk]; w0f = max(float(np.sqrt(np.mean(bm[1:1 + Mm] ** 2))), 1e-3)
        out_full[f] = dict(a0=bm[0], w0_raw=softplus_inv(w0f), b0=bm[1:1 + Mm] / w0f, c=bm[1 + Mm:1 + Mm + P],
                           wi_raw=softplus_inv(wi), bi=bint / wi)
        if dd is not None:
            out_full[f]['d'] = bm[1 + Mm + P:]
        everything = np.ones(sub.size, bool)
        report[f] = dict(r2_main_real=r2(y, yh, real), r2_main_grid=r2(y, yh, ~real),
                         r2_main_real_by_tier=[r2(y, yh, real & (tier == k)) for k in range(4)],
                         r2_main_grid_by_tier=[r2(y, yh, ~real & (tier == k)) for k in range(4)],
                         rmse_main_real=float(np.sqrt(np.mean((y[real] - yh[real]) ** 2))),
                         r2_full_sub=r2(y[sub], yhf_sub, everything), r2_main_sub=r2(y[sub], yh[sub], everything),
                         r2_full_sub_by_tier=[r2(y[sub], yhf_sub, tier[sub] == k) for k in range(4)],
                         a0=float(a0), w0=w0, c=[round(float(x), 3) for x in cc],
                         d=None if dd is None else [round(float(x), 3) for x in dd], wi_full=wi)
    report['tier_counts_real'] = [int((real & (tier == k)).sum()) for k in range(4)]
    report['relative'] = None if rel is None else dict(r2=rel['r2'], k_q01_50_99=rel['range'], k0=float(rel['k0']))
    report['spec'] = dict(spec)
    print(json.dumps(report, default=float, indent=1), flush=True)
    for tag, o in (('neutral', out_main), ('lsfull', out_full)):
        vals = {k: v for k, v in params.items() if k != 'dd_survival_share'}
        for f in ('s', 'r'):
            for k, v in o[f].items():
                vals[f'gk_{f}_{k}'] = np.asarray(v, np.float64)
        if rel is not None:
            wk = max(float(np.sqrt(np.mean(rel['kappa'] ** 2))), 1e-3)
            vals.update(gk_k0=np.asarray(rel['k0'], np.float64), gk_wk_raw=np.asarray(softplus_inv(wk), np.float64),
                        gk_bk=np.asarray(rel['kappa'] / wk, np.float64))
        np.savez_compressed(f'{a.out_prefix}_{tag}.npz', **vals)
        print(f'wrote {a.out_prefix}_{tag}.npz', flush=True)
    Path(f'{a.out_prefix}_report.json').write_text(json.dumps(report, default=float, indent=1) + '\n')


if __name__ == '__main__':
    main()
