"""Within-route residual autocorrelation of BBS counts at a Laplace fit (2026-11-06; user: are the BBS rows overweighted?).

At the fit's state, per observer x route unit: mean count mu_i (route effect included at its conditional mode, as saved
in best_params: observer_route_eps), NB2 concentration phi_i. Pearson residual r_i = (y - mu) / sqrt(mu + mu^2/phi).
Within each unit, residuals ordered by year; lag-1 and lag-2 autocorrelation pooled over units with >= 5 consecutive
years. Also the variance inflation of unit-year sums vs independence and an AR(1) effective-sample-size factor
(1 - rho) / (1 + rho) for long series. Split by region and era.
Usage (takeover runner, fit overlay): resid_autocorr.py --params NPZ --out JSON
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import jax.numpy as jnp
    from numpyro import handlers
    from harness import Sandbox, ap
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(data=sb.data, prior_scale=1.)
    mu = np.asarray(tr['expected_obs']['value'], float)
    so = tr['obs']; fn = so['fn']
    conc = np.asarray(getattr(fn, 'concentration', getattr(getattr(fn, 'base_dist', fn), 'concentration', np.nan)), float)
    conc = np.broadcast_to(conc, mu.shape) if conc.size > 1 else np.full(mu.shape, float(conc))
    y = np.asarray(sb.data['observed_results'], float); idx = np.asarray(sb.data['obs_obsroute_idx'])
    w = np.asarray(sb.data['obs_weight'], float) if sb.data.get('obs_weight') is not None else np.ones_like(y)
    yrs = np.asarray(sb.years)[np.asarray(sb.data['obs_time_indices'])].astype(int)
    lon = sb.lon[sb.lookup[np.asarray(sb.data['obs_rows']), np.asarray(sb.data['obs_cols'])]]
    ok = np.isfinite(y) & (idx >= 0) & (w > 0) & np.isfinite(mu) & (mu > 0)
    var = mu + mu ** 2 / np.maximum(conc, 1e-9)
    r = np.where(ok, (y - mu) / np.sqrt(np.maximum(var, 1e-12)), np.nan)
    out = {}
    for reg, m in dict(all=np.ones_like(ok), west=lon < -100, plains=(lon >= -100) & (lon < -95), east=lon >= -95).items():
        for era, (y0, y1) in dict(all=(1966, 2025), pre1994=(1966, 1993), post1994=(1994, 2025)).items():
            sel = ok & m & (yrs >= y0) & (yrs <= y1)
            pairs1, pairs2 = [], []
            order = np.lexsort((yrs[sel], idx[sel])); ii, yy, rr = idx[sel][order], yrs[sel][order], r[sel][order]
            same1 = (ii[1:] == ii[:-1]) & (yy[1:] - yy[:-1] == 1)
            same2 = (ii[2:] == ii[:-2]) & (yy[2:] - yy[:-2] == 2)
            if same1.sum() < 50:
                continue
            a1 = np.corrcoef(rr[:-1][same1], rr[1:][same1])[0, 1]
            a2 = np.corrcoef(rr[:-2][same2], rr[2:][same2])[0, 1] if same2.sum() > 50 else np.nan
            out[f'{reg}_{era}'] = dict(n_rows=int(sel.sum()), n_pairs=int(same1.sum()), rho1=round(float(a1), 3),
                                       rho2=round(float(a2), 3), ess_factor_ar1=round(float((1 - a1) / (1 + a1)), 3),
                                       pearson_var=round(float(np.nanvar(rr)), 3))
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out.items():
        print(k, v)


if __name__ == '__main__':
    main()
