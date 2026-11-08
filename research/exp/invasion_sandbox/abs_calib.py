"""Absolute-abundance bridge (user, 2026-11-08): calibrate the model's route-count density to birds/km2 with the Partners in
Flight continental total (~40 million House Finches, US + Canada), then express the Allee n50 pin and K in V&L's units.
Route counts per unit model density: q = expected_obs_before_observer x exp(sigma^2/2) / simulated_density (mean route).
Usage (via research/bin/local_takeover_run.sh, AGE_MODEL_CONFIG = the fit's overlay): python abs_calib.py RUN/best_params.npz
First result (LC3, 2026-11-08): 1 mean-route count = .735 [.367] birds/km2 at PIF 40 M [20 M]. See CONSIDERATIONS.md G14
and SANDBOX_INVASION_SHAPE_2026-11.md ("ABSOLUTE BRIDGE"). Rerun it whenever pop_scalar, route sigma or the readout changes."""
import os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
import jax.numpy as jnp
from numpyro import handlers
from harness import Sandbox, ap
from src.config_utils import load_age_model_config
sb = Sandbox(os.environ['SBX_REFINED'])
lat, lon = np.asarray(sb.latd), np.asarray(sb.lon); rows, cols = np.asarray(sb.rows), np.asarray(sb.cols)
B = np.array([(-118., 32.5), (-117.1, 32.5), (-114.7, 32.7), (-111.0, 31.3), (-108.2, 31.3), (-108.2, 31.8), (-106.5, 31.8), (-104.5, 29.6),
              (-103.0, 29.0), (-101.4, 29.8), (-99.5, 27.5), (-97.2, 25.9)])
mex = (lat < np.interp(lon, B[:, 0], B[:, 1])) & (lon < -97.)
z = np.load(sys.argv[1]); lt = {k: jnp.asarray(v) for k, v in sb.lat.items()}; lt.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_')})
sig = float(z['laplace_route_sigma'])
tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lt), 1)).get_trace(data=sb.data, prior_scale=1.)
yrs = np.asarray(sb.years).astype(int); T = len(yrs)
dens = np.asarray(tr['simulated_density']['value'])[:, rows, cols]
ti = np.asarray(sb.data['obs_time_indices']); orow, ocol = np.asarray(sb.data['obs_rows']), np.asarray(sb.data['obs_cols'])
mu0 = np.asarray(tr['expected_obs_before_observer']['value']); d_row = np.asarray(tr['simulated_density']['value'])[ti, orow, ocol]
ok = (np.asarray(sb.data['obs_obsroute_idx']) >= 0) & (d_row > 1e-6) & (yrs[ti] >= 2010)
q = np.median(mu0[ok] / d_row[ok]) * np.exp(.5 * sig ** 2)     # route count (mean route) per unit model density
w = (yrs >= 2010) & (yrs <= 2019)
rc = dens[w].mean(0) * q                                         # mean route count per cell, 2010-19
A = sb.cell_km2; us = ~mex
S = float((rc[us] * A).sum())                                    # route-count x km2, US + Canada cells in the domain
for Ntot in (20e6, 40e6):
    X = Ntot / S                                                 # birds / km2 per route count
    print(f'PIF total {Ntot / 1e6:.0f} M -> 1 route count = {X:.3f} birds/km2 (= {1 / X:.1f} route counts per bird/km2)')
cfg = load_age_model_config()['population_model']
n50_int = float(np.logaddexp(0., cfg['pinned_sites']['n50_raw'])); pop = float(np.asarray(sb.data['pop_scalar']))
print(f'q (route counts per model unit) {q:.3f} | cell area {A:.0f} km2 | US+CA cells {int(us.sum())} | sum rc*A {S:.3e} | pop_scalar {pop:.3f}')
print(f'pinned n50 = {n50_int:.4f} internal (softplus) ; floor {cfg.get("capacity_floor_internal")} internal')
K = np.asarray(tr['K_flat']['value']).reshape(T, -1); K = K if K.shape[1] == len(rows) else K[:, rows * sb.Nx + cols]
occ = dens[w].mean(0) > .01 * np.median(K[w].mean(0))
for nm, val in (('median K, occupied US+CA cells 2010-19', np.median(K[w].mean(0)[occ & us])), ('median density, occupied US+CA 2010-19', np.median(dens[w].mean(0)[occ & us]))):
    print(f'{nm}: {val:.4f} model units = {val * q:.3f} route counts = {val * q * 40e6 / S:.3f} birds/km2 (PIF 40 M)')
