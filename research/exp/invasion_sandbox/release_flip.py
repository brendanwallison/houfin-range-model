"""RELEASE FLIP (2026-11-06): what turns the release region from sink to source in the 1950s, on a Laplace fit?

Replays the fitted point (no refitting) with counterfactual edits and decomposes the release area year by year
(release_area.py components). Arms: fitted; rates_1965 (release-area Sa, Sj, Fmax, K for 1940-64 set to their 1965
values, c recomputed); allee_off (gamma x 1e-6); no_drain (departure suppressed everywhere before 1965); both.
Usage (takeover runner, the fit's overlay): release_flip.py --params NPZ --out JSON [--radius 160]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--radius', type=float, default=160.)
    p.add_argument('--zarms', action='store_true', help='run only the habitat-input (Z) arms on the release area')
    a = p.parse_args()
    import jax.numpy as jnp
    from harness import Sandbox, capture, to_land, potential_lambda
    from src.model import age_forward as af
    try:
        from src.model.density_regulation import CROWDING_POWER as POWER
    except ImportError:
        POWER = 1.0
    sb = Sandbox(os.environ['SBX_REFINED'])
    with np.load(a.params) as z:
        sb.lat = dict(sb.lat); sb.lat.update({k: np.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
    years = np.asarray(sb.years).astype(int); T = len(years)
    fitted = sb.run(dict(name='fitted'), keep_density=True); N0 = fitted.pop('_density')
    x, _ = capture(sb.data, sb.lat)
    m = sb.dist_km <= a.radius; rel = sb.rel
    r0, c0 = int(sb.rows[rel]), int(sb.cols[rel])
    block = (np.abs(sb.rows - r0) <= 2) & (np.abs(sb.cols - c0) <= 2)
    i65 = int(np.nonzero(years == 1965)[0][0]); early = (years >= 1940) & (years < 1965)

    # rates_1965: multipliers (T, N_land) = value(1965) / value(t) on the release area, early years
    def freeze_mult(key):
        v = np.asarray(x[key]).reshape(T, -1); mult = np.ones_like(v)
        mult[np.ix_(early, m)] = v[i65, m][None, :] / np.maximum(v[np.ix_(early, m)], 1e-30)
        return mult
    freeze = {f: freeze_mult(f + '_flat') for f in ('Sa', 'Sj', 'Fmax', 'K')}
    # Departure is Veit's p = N/K here (dispersal_random does not reach it): no_drain = static departure_scale .1 on the
    # release area (whole run; read the block only -- it also slows the area's later export).
    ds0 = x.get('departure_scale'); nl = len(sb.rows)
    ds = np.ones(nl) if ds0 is None else np.asarray(ds0, float).copy(); ds[m] *= .1
    dsj = jnp.asarray(ds, dtype=jnp.asarray(x['Sa_flat']).dtype)
    arms = [dict(name='rates_1965', rate_array=freeze), dict(name='allee_off', allee_gamma_scale=1e6),
            dict(name='no_drain', forward_args=dict(departure_scale=dsj)),
            dict(name='rates_1965_no_drain', rate_array=freeze, forward_args=dict(departure_scale=dsj)),
            dict(name='allee_off_no_drain', allee_gamma_scale=1e6, forward_args=dict(departure_scale=dsj)),
            dict(name='vital_1965', rate_array={f: freeze[f] for f in ('Sa', 'Sj', 'Fmax')}),
            dict(name='K_1965', rate_array={'K': freeze['K']})]
    if a.zarms:   # habitat-input edits on the release area (local Z only; Z_disp / journey untouched), 1951-75
        arms = []
    dens = {'fitted': N0}
    for arm in arms:
        row = sb.run(arm, keep_density=True); dens[arm['name']] = row.pop('_density')
    if a.zarms:
        Z0 = np.asarray(sb.data['Z_gathered']); dt = Z0.dtype
        i51 = int(np.nonzero(years == 1951)[0][0]); win = (years >= 1951) & (years <= 1975)
        east = sb.region['east']; mE = Z0[:, east].mean(1)                      # (T, M) eastern mean z
        zedits = {
            'Z_frozen1951': lambda Z: Z[i51, m][None],
            'Z_continental_only': lambda Z: Z[i51, m][None] + (mE[win] - mE[i51])[:, None, :],
            'Z_local_only': lambda Z: Z[np.ix_(win, m)] - (mE[win] - mE[i51])[:, None, :],
        }
        data0 = sb.data
        zruns = [(n, fn, {}) for n, fn in zedits.items()]
        for ex in (10, 25):   # release cells exempt from the Allee for ex years after release (existing forward option)
            zruns.append((f'exempt{ex}', None, dict(allee_exempt_years=ex)))
            zruns.append((f'Z_frozen1951_exempt{ex}', zedits['Z_frozen1951'], dict(allee_exempt_years=ex)))
        for name, fn, fa in zruns:
            Z = Z0.copy()
            if fn is not None:
                Z[np.ix_(win, m)] = fn(Z0).astype(dt)
            sb.data = dict(data0); sb.data['Z_gathered'] = jnp.asarray(Z)
            try:
                row = sb.run(dict(name=name, forward_args=fa), keep_density=True); dens[name] = row.pop('_density')
            finally:
                sb.data = data0

    # decomposition on the fitted run (as release_area.py)
    Nx = sb.Nx
    Sa, Sj, F, K, c = (to_land(x[k], sb.rows, sb.cols, T, Nx) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat', 'K_flat', 'c_flat'))
    yb = bool(x.get('yearling_breeding')); gamma = float(np.asarray(x['allee_gamma'])); relA = bool(x.get('allee_relative'))
    lam = potential_lambda(Sa, Sj, F, yb)
    R0 = (F * Sj / np.maximum(1 - Sa, 1e-9)) if yb else (F * Sa * Sj / np.maximum(1 - Sa, 1e-9))
    out = dict(params=str(a.params), radius_km=a.radius, n_area_cells=int(m.sum()), n_block_cells=int(block.sum()),
               decomposition={}, arms={})
    for i, y in enumerate(years):
        if y < 1945 or y > 1975:
            continue
        n = N0[i][m]; w = n / max(n.sum(), 1e-30); nk = n / np.maximum(K[i][m], 1e-12)
        mated = np.asarray(af.allee_mated_fraction(gamma * (nk if relA else n)))
        crowd = 1. / (1. + c[i][m] * nk ** POWER)
        lam_b = float(np.median(lam[i][block]))
        out['decomposition'][int(y)] = dict(total=float(n.sum()), lambda0_w=float((w * lam[i][m]).sum()), lambda0_block_median=lam_b,
                                            mated_w=float((w * mated).sum()), N_over_K_w=float((w * nk).sum()),
                                            crowding_w=float((w * crowd).sum()), sink_share=float(w[R0[i][m] < 1].sum()),
                                            K_block_median=float(np.median(K[i][block])),
                                            Sa_block=float(np.median(Sa[i][block])), Sj_block=float(np.median(Sj[i][block])),
                                            F_block=float(np.median(F[i][block])))
    i52 = int(np.nonzero(years == 1952)[0][0])
    for name, D in dens.items():
        b = D[:, block].sum(1); e = D[:, sb.region['east']].sum(1)
        thr = 2. * b[i52]; ok = [i for i in range(i52, T - 5) if (b[i:i + 5] > thr).all()]
        out['arms'][name] = dict(takeoff_year=int(years[ok[0]]) if ok else None,
                                 block={int(y): float(b[i]) for i, y in enumerate(years) if y in (1950, 1952, 1955, 1958, 1960, 1965, 1970, 1975, 1980)},
                                 east_total={int(y): float(e[i]) for i, y in enumerate(years) if y in (1960, 1970, 1980, 1990)})
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    for y, r in out['decomposition'].items():
        if y % 2 == 1:
            print(y, {k: round(v, 3) for k, v in r.items()})
    for name, r in out['arms'].items():
        print(name, 'takeoff', r['takeoff_year'], 'block', {k: round(v, 3) for k, v in r['block'].items()},
              'east', {k: round(v, 1) for k, v in r['east_total'].items()})


if __name__ == '__main__':
    main()
