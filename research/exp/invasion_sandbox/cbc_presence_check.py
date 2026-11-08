"""Early-presence check with the FIT'S OWN CBC observation model (2026-11-07; user: "the historic western range looks tiny ...
missing core presences even in the 1940s? must be worse early").

For every POSITIVE CBC circle-year (<= 1965): mu = exp(s + b * log effort) x model abundance available to the circle (local +
seasonal sources), circle effect 0, effort = recorded or the year trend m(t) (as --cbc_count_mode latent); P(count > 0) =
1 - (1 + mu / phi)^-phi with the fit's phi. Reports by region x era: n positives, median P, share with P < .1 ("impossible").
Also the model's occupied area by decade: western land cells (col < 48) with density >= 1% of the modern occupied median.
Usage (takeover runner, fit overlay): cbc_presence_check.py --params NPZ --cbc cbc_early.npz --out JSON
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--cbc', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import jax.numpy as jnp
    from numpyro import handlers
    from harness import Sandbox, ap
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_')})
        cc = np.asarray(z['laplace_cbc_counts']) if 'laplace_cbc_counts' in z.files else None
    s, b, phi = (float(cc[0]), float(cc[1]), float(np.exp(cc[2]))) if cc is not None else (np.nan, .6, 1.)
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(data=sb.data, prior_scale=1.)
    N = (np.asarray(tr['Na_grid']['value']) + np.asarray(tr['Nj_grid']['value']))[:, np.asarray(sb.rows), np.asarray(sb.cols)]
    N = N * float(sb.data['pop_scalar'])
    yrs = np.asarray(sb.years).astype(int)
    c = np.load(a.cbc); cy = yrs[c['cbc_t']]
    rec = np.asarray(c['cbc_effort_missing']) == 0; le = np.asarray(c['cbc_log_effort'], float)
    kn = np.array([1920., 1930., 1940., 1950., 1960., 1966.])
    B = lambda yy: np.maximum(0., 1. - np.abs(np.clip(yy, kn[0], kn[-1])[:, None] - kn[None, :]) / 10.)
    rk = rec & (cy <= 1965); coef, *_ = np.linalg.lstsq(B(cy[rk].astype(float)), le[rk], rcond=None)
    le_use = np.where(rec, le, B(cy.astype(float)) @ coef)
    pos = (np.asarray(c['cbc_count']) > 0) & (cy <= 1965)
    ci = c['cbc_circle'][pos]; src = c['cbc_source_idx'][ci]
    w = np.where(src >= 0, np.maximum(c['cbc_w_local'][ci], c['cbc_w_seasonal'][ci]), 0.)
    A = (N[c['cbc_t'][pos][:, None], np.maximum(src, 0)] * w).sum(1)
    if not np.isfinite(s):   # fit without a CBC count term: calibrate s on the recorded-effort positives (median match)
        s = float(np.median(np.log(np.asarray(c['cbc_count'])[pos] + 1e-9) - np.log(np.maximum(A, 1e-12)) - b * le_use[pos]))
    mu = np.exp(s + b * le_use[pos]) * A
    P = 1. - (1. + mu / phi) ** (-phi)
    lon = sb.lon[np.maximum(src[:, 0], 0)]; reg = np.where(lon < -100, 'west', np.where(lon < -95, 'plains', 'east'))
    yr = cy[pos]; eras = [(1902, 1919), (1920, 1929), (1930, 1939), (1940, 1949), (1950, 1959), (1960, 1965)]
    out = dict(params=str(a.params), s=s, b=b, phi=phi, presence={}, west_occupied_cells={})
    for r in ('west', 'plains', 'east'):
        rows = []
        for e0, e1 in eras:
            m = (reg == r) & (yr >= e0) & (yr <= e1)
            if m.sum() == 0:
                continue
            rows.append(dict(era=f'{e0}-{e1}', n=int(m.sum()), median_P=round(float(np.median(P[m])), 3),
                             share_P_lt_01=round(float((P[m] < .1).mean()), 3)))
        out['presence'][r] = rows
    west = np.asarray(sb.cols) < 48
    occ_ref = np.median(N[(yrs >= 2010) & (yrs <= 2020)][:, west][N[(yrs >= 2010) & (yrs <= 2020)][:, west] > 0])
    for y in (1902, 1915, 1930, 1940, 1950, 1966, 1990, 2010):
        i = int(np.flatnonzero(yrs == y)[0]); out['west_occupied_cells'][y] = int((N[i, west] >= .01 * occ_ref).sum())
    out['west_land_cells'] = int(west.sum())
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    for r, rows in out['presence'].items():
        print(r, ' | '.join(f"{q['era']} n{q['n']} medP {q['median_P']:.2f} P<.1 {q['share_P_lt_01']:.2f}" for q in rows))
    print('west occupied cells', out['west_occupied_cells'], 'of', out['west_land_cells'])


if __name__ == '__main__':
    main()
