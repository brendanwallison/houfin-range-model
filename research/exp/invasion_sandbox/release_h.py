"""Where does the release-site habitat rise come from? (2026-11-06, follows RELEASE FLIP)

H_f = (z_t - z_center).beta_f (age_fields.project_and_scatter_age_structured). For each field (s, sj, r, k):
change of H between two years at the release block vs region medians (East / Plains / West, 25-49N), and the
per-z-dimension contributions dz_d * beta_d at the release block (which DESK dims carry it). Also the share of
eastern land cells whose H rises more than the release block's rise (is the release site special or continental?).
Usage (takeover runner, the fit's overlay): release_h.py --params NPZ --out JSON [--y0 1951 --y1 1965]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--y0', type=int, default=1951); p.add_argument('--y1', type=int, default=1965)
    a = p.parse_args()
    from harness import Sandbox, capture
    from numpyro import handlers
    from src.model import age_priors as ap
    sb = Sandbox(os.environ['SBX_REFINED'])
    with np.load(a.params) as z:
        sb.lat = dict(sb.lat); sb.lat.update({k: np.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
    got = {}
    orig = ap.project_and_scatter_age_structured

    def rec(*aa, **kw):
        import inspect
        b = inspect.signature(orig).bind(*aa, **kw); b.apply_defaults(); got.update(dict(b.arguments)); return orig(*aa, **kw)
    ap.project_and_scatter_age_structured = rec
    try:
        x, _ = capture(sb.data, sb.lat)
    finally:
        ap.project_and_scatter_age_structured = orig
    if not got:
        raise SystemExit('project_and_scatter_age_structured not called through age_priors; patch the right module')
    years = np.asarray(sb.years).astype(int); i0, i1 = list(years).index(a.y0), list(years).index(a.y1)
    Z = np.asarray(got['Z_gathered']); zs = got.get('z_shrink'); Zm = got.get('Z_mean')
    if zs is not None:
        Z = np.asarray(Zm) + float(np.asarray(zs)) * (Z - np.asarray(Zm))
    if got.get('split_readout') is not None:
        raise SystemExit('split readout: extend')
    il = np.asarray(x['inv_locations']).reshape(-1, 2)[0]
    rel = int(sb.lookup[il[0], il[1]]); r0, c0 = int(sb.rows[rel]), int(sb.cols[rel])
    block = (np.abs(sb.rows - r0) <= 2) & (np.abs(sb.cols - c0) <= 2)
    dZ = Z[i1] - Z[i0]
    out = dict(y0=a.y0, y1=a.y1, z_shrink=None if zs is None else float(np.asarray(zs)), fields={})
    for f in ('beta_s', 'beta_sj', 'beta_r', 'beta_k'):
        beta = np.asarray(got[f]); dH = dZ @ beta
        contrib = (dZ[block] * beta[None, :]).mean(0)
        order = np.argsort(-np.abs(contrib))[:6]
        reg = {k: float(np.median(dH[m])) for k, m in sb.region.items()}
        east = sb.region['east']
        out['fields'][f] = dict(dH_block=float(dH[block].mean()), dH_region_median=reg,
                                share_east_cells_rising_more=float(np.mean(dH[east] > dH[block].mean())),
                                H_sd_space_y0=float(np.std((Z[i0] @ beta)[east])),
                                top_dims=[(int(d), float(contrib[d]), float(dZ[block, d].mean()), float(beta[d])) for d in order],
                                block_dz_norm=float(np.linalg.norm(dZ[block].mean(0))))
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    for f, r in out['fields'].items():
        print(f, 'dH block', round(r['dH_block'], 3), 'region med', {k: round(v, 3) for k, v in r['dH_region_median'].items()},
              'share East rising more', round(r['share_east_cells_rising_more'], 2), 'spatial sd', round(r['H_sd_space_y0'], 3))
        print('   top dims (d, contrib, dz, beta):', [(d, round(c, 3), round(z, 3), round(b, 3)) for d, c, z, b in r['top_dims']])


if __name__ == '__main__':
    main()
