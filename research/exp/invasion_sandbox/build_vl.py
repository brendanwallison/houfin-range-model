"""Build the V&L-faithful juvenile kernel input dir ("_vl"): a 4-direction x 3-band stack from the 2-D density
exp(-(r/b)^a)/r (V&L's 1-D distance density), with matching edge corrections and Z_disp, everything else symlinked.
Validates the re-implementation by rebuilding the parent 4 x 3 stack, its edge corrections and two years of Z_disp first.
Usage: build_vl.py IN_DIR OUT_DIR VL_B [--validate-only] [--land_share_floor F]
History: the 2026-11-07 _vl inputs (inputs_shape/*_vl) were built with the worktree integrate_paths, which has NO land-share
floor, although their FIELD54 parent Z_disp is floored (1e-4 on land share; 172 cell-band pairs in 164 coastal cells
differ by up to .31; docs/model_spec/DESK_PROVENANCE.md item 4). --land_share_floor 1e-4 (2026-11-08) applies the parent
lineage's rule; it needs CODE_ROOT_FLOOR (a src/ whose integrate_paths accepts land_share_floor, e.g. the main checkout)."""
import os, sys, pickle, json, time
import numpy as np
LSF = float(sys.argv[sys.argv.index('--land_share_floor') + 1]) if '--land_share_floor' in sys.argv else None
sys.path.insert(0, os.environ['CODE_ROOT_FLOOR'] if LSF is not None else os.environ['TAKEOVER_CODE_ROOT'])
import jax, jax.numpy as jnp
from jax.numpy.fft import fft2
from src.model.build_kernels import edge_correction_from_fft, toroidal_distance_grid, radial_kernel_grid
from src.model.build_path_features import integrate_paths

IN, OUT = sys.argv[1], sys.argv[2]; SHAPE = 0.468; VL_B = float(sys.argv[3]); NDIR = 4   # VL_B = V&L Weibull b (km)
VALIDATE_ONLY = '--validate-only' in sys.argv
print('land_share_floor', LSF, '| code root', sys.path[0])
m = pickle.load(open(os.path.join(IN, 'metadata.pkl'), 'rb'))
Ny, Nx = int(m['Ny']), int(m['Nx']); T, NL, M = int(m['time']), int(m['N_land']), int(m['M'])
fk12 = np.asarray(m['juvenile_fft_kernel_stack']); K12, Ly, Lx = fk12.shape
stack12 = np.real(np.fft.ifft2(fk12)).astype(np.float64)
master = stack12.sum(0)
lr, lc = np.asarray(m['land_rows']), np.asarray(m['land_cols'])
land = np.asarray(m['land_mask']).astype(np.float64)
spec = m.get('dispersal_spec') or {}
cell = 54.0
DSPEC = dict(m['dispersal_spec']); DISC = m.get('kernel_discretization'); SPL = [float(v) for v in DSPEC['juvenile_radial_splits_km']]
SPL = [min(v, 1e9) for v in SPL]
rebuilt = np.asarray(radial_kernel_grid(Lx, Ly, 54.0, DSPEC['juvenile_mdd_km'], DSPEC['juvenile_shape'], DISC), np.float64)
print('VALIDATE master: max |rebuilt - stored| =', float(np.abs(rebuilt - master).max()), '| stored max', float(master.max()), '| splits', SPL)
print('master mass', master.sum(), '| K12', K12, '| grid', (Ny, Nx), (Ly, Lx), '| T, NL, M', T, NL, M, '| spec keys', list(spec)[:8] if isinstance(spec, dict) else type(spec))


def wedges(master, ndir, splits):
    y, x = np.meshgrid(np.arange(Ly), np.arange(Lx), indexing='ij')
    dx = np.where(x <= Lx // 2, x, x - Lx); dy = np.where(y <= Ly // 2, y, y - Ly)
    ang = np.arctan2(dy, dx)
    if ndir == 4:   # the stored order and targets: to_NORTH, to_SOUTH, to_EAST, to_WEST
        names, targets = ['to_NORTH', 'to_SOUTH', 'to_EAST', 'to_WEST'], [-np.pi / 2, np.pi / 2, 0.0, np.pi]
    else:
        targets = [k * 2 * np.pi / ndir for k in range(ndir)]; names = [f'dir{int(round(np.degrees(t)))}' for t in targets]
    half = 2 * np.pi / ndir   # raised-cosine half-width = sector spacing -> partition of unity
    r = np.asarray(toroidal_distance_grid(Lx, Ly, cell))
    slope = 4.0 / (2.0 * cell)
    cdf = [np.ones_like(r) if b <= 1e-6 else (np.zeros_like(r) if b >= 1e9 else 1 / (1 + np.exp(-slope * (r - b)))) for b in splits]
    out, labels = [], []
    for nm, tgt in zip(names, targets):
        diff = np.mod(ang - tgt + np.pi, 2 * np.pi) - np.pi
        w = 0.5 * (1 + np.cos(np.pi * np.clip(diff, -half, half) / half))
        w = np.where((dx == 0) & (dy == 0), 1.0 / ndir, w)
        for i in range(len(splits) - 1):
            out.append(master * w * (cdf[i] - cdf[i + 1])); labels.append(f'{nm}_{splits[i]:.0f}-{min(splits[i + 1], 99999):.0f}')
    return np.stack(out), labels


def edge_corr(stack):
    padded = np.zeros((Ly, Lx)); padded[:Ny, :Nx] = land
    fl = fft2(jnp.asarray(padded)); res = []
    for k in stack:
        wsum = k.sum(); kn = k / wsum if wsum > 1e-12 else k
        res.append(np.asarray(edge_correction_from_fft(fl, fft2(jnp.asarray(kn)), jnp.asarray(land > 0.5), Ny, Nx)))
    return np.stack(res)


zg = np.memmap(os.path.join(IN, m['z_gathered_path']), dtype='float32', mode='r', shape=(T, NL, M))
zd12 = np.memmap(os.path.join(IN, m['z_disp_gathered_path']), dtype='float32', mode='r', shape=(T, NL, K12, M))
steps = int(spec.get('path_integration_steps', 20)) if isinstance(spec, dict) else 20


def zdisp_year(t, stack):
    Zg = np.zeros((Ny, Nx, M), np.float32); Zg[lr, lc] = zg[t]
    kw = {} if LSF is None else dict(land_share_floor=LSF)
    out = integrate_paths(jnp.asarray(Zg)[None], jnp.asarray(stack, jnp.float32), jnp.asarray(land, jnp.float32), steps=steps, **kw)
    return np.transpose(np.asarray(out[0])[lr, lc], (0, 2, 1))   # integrate_paths -> (Ny, Nx, M, K); gather -> (NL, K, M)


# ---- validation against the stored 4 x 3 inputs ----
re12, _ = wedges(master, 4, [0.0, 155.36162529769288, 482.7446923028151, 1e9])
print('VALIDATE wedges: max |rebuilt - stored| =', float(np.abs(re12 - stack12).max()), '| stored max', float(stack12.max()))
ec12 = edge_corr(re12)
print('VALIDATE edge corr: max |rebuilt - stored| =', float(np.abs(ec12[:, land > 0.5] - np.asarray(m['juvenile_edge_correction_stack'])[:, land > 0.5]).max()))
for t in (0, 60):
    z = zdisp_year(t, re12)
    ref = np.asarray(zd12[t])
    d = np.abs(z - ref); rel = d / (np.abs(ref) + 1e-3)
    bad = np.unravel_index(np.argmax(d), d.shape)
    print(f'VALIDATE Z_disp year idx {t}: max abs {float(d.max()):.3e}; median rel {float(np.median(rel)):.2e}; p99 rel {float(np.percentile(rel, 99)):.2e}; p99.9 rel {float(np.percentile(rel, 99.9)):.2e}; worst cell idx {bad[0]} kernel {bad[1]}')
if VALIDATE_ONLY:
    sys.exit(0)

# ---- build the finer stack ----
def vl_master(b, a=0.468, cell=54.0, S=int(DISC['subsamples'])):
    # source-faithful V&L: their K is a density of DISTANCE, f(r) ~ exp(-(r/b)^a) (1-D). The 2-D area density with that
    # distance distribution is k(r) = f(r) / (2 pi r); cell-integrated exactly as radial_kernel_grid (tent sub-offsets).
    x = np.concatenate([np.arange(Lx // 2 + 1), -np.arange(Lx // 2, 0, -1)]).astype(np.float64)
    y = np.concatenate([np.arange(Ly // 2 + 1), -np.arange(Ly // 2, 0, -1)]).astype(np.float64)
    ks = np.arange(-(S - 1), S); sub = ks / S; wts = (S - np.abs(ks)) / float(S * S)
    rmin = cell / (2.0 * S)          # regularise the integrable 1/r singularity at zero offset (sub-cell scale)
    k = np.zeros((Ly, Lx))
    for oy, wy in zip(sub, wts):
        dy2 = (y + oy)[:, None] ** 2
        for ox, wx in zip(sub, wts):
            r = np.sqrt(dy2 + ((x + ox)[None, :]) ** 2) * cell
            k += wy * wx * np.exp(-(r / b) ** a) / np.maximum(r, rmin)
    return k / k.sum()
master_new = vl_master(VL_B)
r_ = np.asarray(toroidal_distance_grid(Lx, Ly, 54.0))
print('new master: V&L faithful b', VL_B, '| own-cell mass', float(master_new[0, 0]), '| P(r > 27)', float(master_new[r_ > 27].sum()), '| shape', SHAPE, '| mean km', float((master_new * r_).sum()), '| old mean km', float((master * r_).sum()),
      '| mass > 155 / > 483:', float(master_new[r_ > 155].sum()), float(master_new[r_ > 483].sum()), '| old', float(master[r_ > 155].sum()), float(master[r_ > 483].sum()))
st, labels = wedges(master_new, NDIR, SPL)
K = st.shape[0]
print('fine stack K', K, 'mass', st.sum(), 'labels', labels[:6], '...')
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(IN):
    if f in ('metadata.pkl',) or f == m['z_disp_gathered_path']:
        continue
    dst = os.path.join(OUT, f)
    if not os.path.exists(dst):
        os.symlink(os.path.realpath(os.path.join(IN, f)), dst)
zname = f'Z_disp_gathered_vl{VL_B:g}.dat'
mm = np.memmap(os.path.join(OUT, zname), dtype='float32', mode='w+', shape=(T, NL, K, M))
t0 = time.time()
for t in range(T):
    mm[t] = zdisp_year(t, st)
    if t % 20 == 0:
        print(f'year idx {t} done ({time.time() - t0:.0f} s)', flush=True)
mm.flush(); del mm
m2 = dict(m)
m2.update(K=K, juvenile_fft_kernel_stack=np.asarray(fft2(jnp.asarray(st))).astype(np.complex64),
          juvenile_edge_correction_stack=edge_corr(st), juvenile_kernel_labels=[str(x) for x in labels], z_disp_gathered_path=zname,
          dispersal_spec=dict(DSPEC))
pickle.dump(m2, open(os.path.join(OUT, 'metadata.pkl'), 'wb'))
json.dump(dict(parent=IN, kernel='V&L faithful: 2-D density exp(-(r/b)^a)/r', vl_b=VL_B, vl_a=SHAPE, ndir=NDIR, splits_km=SPL, K=K, steps=steps, land_share_floor=LSF, built=time.strftime('%Y-%m-%d %H:%M')),
          open(os.path.join(OUT, 'KERNEL_DISCRETISATION.json'), 'w'), indent=1)
print('DONE', OUT, 'K', K, f'{time.time() - t0:.0f} s')
