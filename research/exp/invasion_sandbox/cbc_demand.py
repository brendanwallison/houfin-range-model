"""What do the early CBC counts demand of a fit? (2026-11-07, eta-free line: early CBC data widen the swing -- where?)

At a fit, for CBC circle-years with RECORDED effort (cbc_effort_missing == 0, <= 1965), model abundance available to the circle
A = sum over its local + seasonal source cells of density x weight (as laplace_fit's CBC terms). Per region (circle's first
source cell: West < 100W, Plains 100-95W, East >= 95W) x period, the scale-free ratio
    R = log( sum observed count / sum A x effort^b ),  b = the fit's CBC effort exponent (laplace_cbc_counts[1]) or .6,
reported RELATIVE to the region's first period with data. A rising R means the birds grew faster than the model's colony; a
falling R, slower. Also: positive-share observed vs model P(count > 0) under NB(phi = 1) at the same scale.
Usage (takeover runner, fit overlay): cbc_demand.py --params NPZ --cbc cbc_early.npz --out JSON
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
        b = float(np.asarray(z['laplace_cbc_counts'])[1]) if 'laplace_cbc_counts' in z.files else .6
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(data=sb.data, prior_scale=1.)
    N = (np.asarray(tr['Na_grid']['value']) + np.asarray(tr['Nj_grid']['value']))[:, np.asarray(sb.rows), np.asarray(sb.cols)]
    N = N * float(sb.data['pop_scalar'])
    c = np.load(a.cbc); yrs = np.asarray(sb.years).astype(int)[c['cbc_t']]
    m = (yrs <= 1965) & (np.asarray(c['cbc_effort_missing']) == 0) & (np.asarray(c['cbc_train']) > 0)
    ci = c['cbc_circle'][m]; src = c['cbc_source_idx'][ci]; w = np.where(src >= 0, np.maximum(c['cbc_w_local'][ci], c['cbc_w_seasonal'][ci]), 0.)
    A = (N[c['cbc_t'][m][:, None], np.maximum(src, 0)] * w).sum(1)
    E = A * np.exp(b * c['cbc_log_effort'][m]); y = c['cbc_count'][m]; yr = yrs[m]
    lon = sb.lon[np.maximum(src[:, 0], 0)]
    reg = np.where(lon < -100, 'west', np.where(lon < -95, 'plains', 'east'))
    periods = [(1920, 1939), (1940, 1949), (1950, 1954), (1955, 1959), (1960, 1965)]
    out = dict(params=str(a.params), effort_exponent=b, regions={})
    for r in ('west', 'plains', 'east'):
        rows, ref = [], None
        for p0, p1 in periods:
            s = (reg == r) & (yr >= p0) & (yr <= p1)
            if s.sum() < 5 or E[s].sum() <= 0:
                continue
            R = float(np.log(max(y[s].sum(), .5) / E[s].sum())); ref = R if ref is None else ref
            sc = y[s].sum() / E[s].sum()
            pz = float(np.mean(1. - 1. / (1. + sc * E[s])))   # P(count > 0) under NB(phi = 1) at the period's own scale
            rows.append(dict(period=f'{p0}-{p1}', n=int(s.sum()), obs=float(y[s].sum()), log_ratio_rel=round(R - ref, 3),
                             pos_share_obs=round(float((y[s] > 0).mean()), 3), pos_share_model=round(pz, 3)))
        out['regions'][r] = rows
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    for r, rows in out['regions'].items():
        print(r, ' | '.join(f"{q['period']} n{q['n']} R{q['log_ratio_rel']:+.2f} pos {q['pos_share_obs']:.2f}/{q['pos_share_model']:.2f}" for q in rows))


if __name__ == '__main__':
    main()
