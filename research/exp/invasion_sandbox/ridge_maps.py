"""Validation + lambda maps along the toll-capped ridge (user, after R90: "show the validation maps from these runs").

One column per arm (refit params + its overlay, the arm's own structure), rows:
  1 potential (low-density) lambda 1966      2 potential lambda 2010      3 change 2010 - 1966
  4 realised disease mortality 2011-25 (amplitude x hill(N) at the arm's densities)
  5 held-out 31 BLOCKS, all years: log(sum obs / sum expected) per block (blue = over-predicted, red = under)
  6 held-out 2021-25 rows: log(sum obs / sum expected) per cell (cells with >= 3 rows)
Rows 5-6 are the never-fitted data; route effects are as the fit has them (held-out block routes ~0, NB-CAL).
Usage: one arm per call (each needs its own overlay through sbx_env), then --plot to assemble:
  sbx_env.sh STEP python ridge_maps.py --arm NAME --params NPZ --out DIR
  python ridge_maps.py --plot --out DIR --arms base,c0.35,c0.25,c0.15
"""
import argparse, json, os
from pathlib import Path
import numpy as np


def compute(a):
    from harness import Sandbox, capture, to_land, potential_lambda
    sb = Sandbox(os.environ['SBX_REFINED'])
    p = np.load(a.params)
    sb.lat = {**{k: np.asarray(v, float) for k, v in sb.lat.items()}, **{k: np.asarray(p[k], float) for k in p.files}}   # ALL saved sites (F2's capacity_factor is not in the base latents)
    sb.base_abund = sb.base_x = sb.base_med = None
    x, _ = capture(sb.data, sb.lat)
    T = len(sb.years); yrs = np.asarray(sb.years)
    Sa, Sj, F = (to_land(x[k], sb.rows, sb.cols, T, sb.Nx) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'))
    lam = potential_lambda(Sa, Sj, F, bool(x.get('yearling_breeding')))
    fit = sb.run(dict(name=a.arm), keep_density=True); dens = fit.pop('_density')
    amp = np.asarray(x['disease_m_amp']).reshape(T, -1)
    amp = amp if amp.shape[1] == len(sb.rows) else to_land(np.asarray(x['disease_m_amp']), sb.rows, sb.cols, T, sb.Nx)
    hn = float(np.ravel(np.asarray(x['disease_hill_n']))[0]); kh = float(np.ravel(np.asarray(x['disease_k_half']))[0])
    h = np.power(np.maximum(dens, 0) + 1e-12, hn); m = amp * h / (h + kh ** hn)
    # observation rows at this arm's state
    from numpyro import handlers
    from harness import ap
    tr = handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d, data=sb.lat), rng_seed=104)).get_trace(data=sb.data, prior_scale=1.)
    ok = np.asarray(sb.obs_ok)
    y = np.asarray(sb.data['observed_results'])[ok]; mu = np.asarray(tr['expected_obs']['value'])[ok]
    held = (np.asarray(sb.data['obs_weight']) <= 0)[ok]
    oy = yrs[np.asarray(sb.data['obs_time_indices'])][ok]
    orow, ocol = np.asarray(sb.data['obs_rows'])[ok], np.asarray(sb.data['obs_cols'])[ok]
    t66, t10 = int(np.flatnonzero(yrs == 1966)[0]), int(np.flatnonzero(yrs == 2010)[0])
    late = (yrs >= 2011) & (yrs <= 2025)
    np.savez_compressed(Path(a.out) / f'{a.arm}.npz', rows=sb.rows, cols=sb.cols, shape=np.asarray(np.asarray(sb.data['land_mask']).shape),
                        lam66=lam[t66], lam10=lam[t10], toll=m[late].mean(0), occ=np.isfinite(sb.first_obs),
                        y=y, mu=mu, held=held, oy=oy, orow=orow, ocol=ocol)
    print('saved', a.arm)


def plot(a):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    arms = a.arms.split(','); D = {k: np.load(Path(a.out) / f'{k}.npz') for k in arms}
    d0 = D[arms[0]]; shape = tuple(d0['shape'])

    def grid(v, d):
        g = np.full(shape, np.nan); g[d['rows'], d['cols']] = v; return g

    def block_ratio(d, sel, b):   # sum obs / sum expected over b x b cell blocks, drawn on the block's cells
        g = np.full(shape, np.nan)
        br, bc = d['orow'][sel] // b, d['ocol'][sel] // b
        keys, inv = np.unique(np.c_[br, bc], axis=0, return_inverse=True)
        so = np.bincount(inv, d['y'][sel]); se = np.bincount(inv, d['mu'][sel]); n = np.bincount(inv)
        for (r, c), o, e, k in zip(keys, so, se, n):
            if k >= 3 and e > 0:
                g[r * b:(r + 1) * b, c * b:(c + 1) * b] = np.log((o + .5) / (e + .5))
        return g
    land = np.zeros(shape, bool); land[d0['rows'], d0['cols']] = True
    titles = ['potential lambda 1966', 'potential lambda 2010', 'change 2010 - 1966', 'disease mortality 2011-25',
              'held-out BLOCKS: log(obs/exp)', 'held-out 2021-25: log(obs/exp)']
    fig, axes = plt.subplots(6, len(arms), figsize=(3.3 * len(arms), 15.5), squeeze=False)
    for j, k in enumerate(arms):
        d = D[k]
        blocks = d['held'] & (d['oy'] < 2021); h21 = d['held'] & (d['oy'] >= 2021)
        tot_b = np.log(d['y'][blocks].sum() / d['mu'][blocks].sum()); tot_21 = np.log(d['y'][h21].sum() / d['mu'][h21].sum())
        occ = d['occ']
        panels = [(grid(d['lam66'], d), dict(cmap='RdBu_r', norm=TwoSlopeNorm(1., .2, 3.)), f'occupied median (all regions) {np.median(d["lam66"][occ]):.2f}'),
                  (grid(d['lam10'], d), dict(cmap='RdBu_r', norm=TwoSlopeNorm(1., .2, 3.)), f'occ. median {np.median(d["lam10"][occ]):.2f}'),
                  (grid(d['lam10'] - d['lam66'], d), dict(cmap='PuOr_r', norm=TwoSlopeNorm(0., -1., 2.)), f'occ. median {np.median((d["lam10"] - d["lam66"])[occ]):+.2f}'),
                  (grid(d['toll'], d), dict(cmap='magma', vmin=0, vmax=.5), f'occ. mean {d["toll"][occ].mean():.2f}'),
                  (block_ratio(d, blocks, 4), dict(cmap='RdBu_r', norm=TwoSlopeNorm(0., -1.5, 1.5)), f'all blocks {tot_b:+.2f}'),
                  (block_ratio(d, h21, 1), dict(cmap='RdBu_r', norm=TwoSlopeNorm(0., -1.5, 1.5)), f'all rows {tot_21:+.2f}')]
        for i, (g, kw, note) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(np.where(land, 0., np.nan), cmap='Greys', vmin=0, vmax=4, interpolation='nearest')
            im = ax.imshow(g, interpolation='nearest', **kw)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{k}: {titles[i]}\n{note}' if i == 0 else f'{titles[i]}\n{note}', fontsize=8)
            if j == len(arms) - 1:
                fig.colorbar(im, ax=ax, fraction=.04)
    fig.suptitle('Toll-capped ridge (R90): lambda, disease and never-fitted data by arm (rows 5-6: red = observed > expected, i.e. under-predicted; blue = over-predicted)', fontsize=10)
    fig.tight_layout(); fig.savefig(Path(a.out) / 'ridge_maps.png', dpi=130); print('wrote', Path(a.out) / 'ridge_maps.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm'); p.add_argument('--params'); p.add_argument('--out', required=True)
    p.add_argument('--plot', action='store_true'); p.add_argument('--arms', default='base,c0.35,c0.25,c0.15')
    a = p.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    plot(a) if a.plot else compute(a)
