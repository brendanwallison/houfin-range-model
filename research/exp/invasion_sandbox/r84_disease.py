"""R84 (user: "you haven't explored the disease model at all"): what does the fit's disease process do?

[cur]'s disease acts on SURVIVAL (disease_prior.target = survival): added mortality m = amplitude(cell, t) x hill(local N)
applied to adults, juvenile stayers and arrivers (age_forward), mortality ceiling .8. Readouts on the fit:
(a) disease amplitude by region x era (mean over occupied cells) and the realised added mortality at the fit's own
    densities (amplitude x hill(N)), i.e. how many survival points disease removes, where and when;
(b) replay with disease OFF (amplitude 0): BBS log-lik and abundance by region x era vs the fit -- how much of the modern
    abundance pattern disease carries.
Usage (inside sbx_env.sh): python r84_disease.py --out FILE.json
"""
import argparse, json, os
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from harness import Sandbox, capture, to_land


def kq_stats(sb, x, occ, k_land=None, q_land=None):
    """Occupied-cell medians of K (route counts) and mean journey survival Q by region at 1966 / 1993 / 2010 / 2020,
    plus dlogK and dQ over 1993 -> 2010."""
    T, yrs = len(sb.years), np.asarray(sb.years)
    K = np.asarray(k_land) if k_land is not None else to_land(x['K_flat'], sb.rows, sb.cols, T, sb.Nx)
    if q_land is None:
        q = np.asarray(x['Q_flat']); q = q.reshape(T, -1, q.shape[-1]).mean(-1) if q.ndim >= 2 else q.reshape(T, -1)
        q_land = q if q.shape[1] == len(sb.rows) else to_land(q.reshape(-1), sb.rows, sb.cols, T, sb.Nx)
    Q = np.asarray(q_land)
    res = {}
    for reg in ('east', 'plains', 'west'):
        mo = sb.region[reg] & occ
        ti = {y: int(np.flatnonzero(yrs == y)[0]) for y in (1966, 1993, 2010, 2020)}
        kmed = {str(y): float(np.median(K[i][mo])) for y, i in ti.items()}
        qmed = {str(y): float(np.mean(Q[i][mo])) for y, i in ti.items()}
        res[reg] = dict(K_median=kmed, Q_mean=qmed,
                        dlogK_1993_2010=float(np.median(np.log(np.maximum(K[ti[2010]][mo], 1e-9)) - np.log(np.maximum(K[ti[1993]][mo], 1e-9)))),
                        dQ_1993_2010=float(np.mean(Q[ti[2010]][mo] - Q[ti[1993]][mo])))
    return res


def disease_stats(sb, arm_extra=None, off=True):
    """r84 readouts at sb.lat under whatever forward is installed (refit_fields --disease_readout passes its arm keys)."""
    arm_extra = dict(arm_extra or {}); k_flat = arm_extra.pop('_k_flat', None)
    x, _ = capture(sb.data, sb.lat)
    keys = sorted(k for k in x if 'disease' in k)
    amp = np.asarray(x['disease_m_amp']); T, N = len(sb.years), len(sb.rows)
    amp = amp.reshape(T, -1)[:, :N] if amp.reshape(T, -1).shape[1] == N else to_land(amp, sb.rows, sb.cols, T, sb.Nx)
    fit = sb.run(dict(name='fitted', **arm_extra), keep_density=True); dens = fit.pop('_density')
    hn = float(np.ravel(np.asarray(x['disease_hill_n']))[0]); kh = float(np.ravel(np.asarray(x['disease_k_half']))[0])
    h = np.power(np.maximum(dens, 0) + 1e-12, hn); m = amp * h / (h + kh ** hn)
    occ = np.isfinite(sb.first_obs); yrs = sb.years
    out = dict(disease_args=keys, hill_n=hn, k_half=kh, by_region={})
    for reg in ('east', 'plains', 'west'):
        mo = sb.region[reg] & occ; row = {}
        for y0, y1 in ((1966, 1993), (1994, 2000), (2001, 2010), (2011, 2025)):
            t = (yrs >= y0) & (yrs <= y1)
            row[f'{y0}-{y1}'] = dict(amplitude=float(amp[t][:, mo].mean()), realised_mortality=float(m[t][:, mo].mean()),
                                     realised_mortality_q90=float(np.quantile(m[t][:, mo], .9)))
        out['by_region'][reg] = row
    # effective (disease-inclusive) potential lambda: Sa (1 - m) + Sj (1 - m) Fmax (yearling), occupied cells
    from harness import potential_lambda
    Sa, Sj, F = (to_land(x[k], sb.rows, sb.cols, T, sb.Nx) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat'))
    yb = bool(x.get('yearling_breeding'))
    lam_pot = potential_lambda(Sa, Sj, F, yb); lam_eff = potential_lambda(Sa * (1 - m), Sj * (1 - m), F, yb)
    out['lambda_occupied'] = {}
    for reg in ('east', 'plains', 'west'):
        mo = sb.region[reg] & occ
        out['lambda_occupied'][reg] = {str(y): dict(potential=float(np.median(lam_pot[yrs == y][0][mo])),
                                                    effective=float(np.median(lam_eff[yrs == y][0][mo])))
                                       for y in (1940, 1966, 1990, 2000, 2010, 2020)}
    # REVIEW 8: 'effective' above is Sa(1-m) + Sj(1-m)Fmax at the fit's OWN densities WITHOUT crowding -- neither the
    # low-density rate (disease is density-dependent: m -> 0 as N -> 0, so low-density lambda = potential) nor realised
    # growth. Realised per-capita growth N_{t+1} / N_t (crowding, Allee, disease, net immigration all included):
    out['effective_is'] = 'Sa(1-m)+Sj(1-m)Fmax at fit densities, no crowding (NOT low-density, NOT realised)'
    out['lowdensity_lambda_is'] = 'equal to potential (hill(N) -> 0 as N -> 0)'
    ratio = np.where((dens[:-1] > 1e-6) & (dens[1:] > 1e-6), dens[1:] / np.maximum(dens[:-1], 1e-12), np.nan)
    out['realised_growth_occupied_median'] = {}
    for reg in ('east', 'plains', 'west'):
        mo = sb.region[reg] & occ
        out['realised_growth_occupied_median'][reg] = {f'{y0}-{y1}': float(np.nanmedian(ratio[(yrs[:-1] >= y0) & (yrs[:-1] <= y1)][:, mo]))
                                                       for y0, y1 in ((1966, 1979), (1980, 1993), (1994, 2000), (2001, 2010), (2011, 2024))}
    bbl_npz = os.environ.get('SBX_BBL_NPZ')
    if bbl_npz:   # adult survival at the BANDED cells (release-weighted, years by .. by + 4), without and with disease
        z = np.load(bbl_npz); out['sa_banded'] = {}
        for g_, gname in ((0, 'west'), (1, 'central'), (2, 'east')):
            for lab, lo, hi in (('band 1960-89', 1960, 1989), ('band 1994-2019', 1994, 2019)):
                sel = (z['grp'] == g_) & (z['by'] >= lo) & (z['by'] <= hi)
                w = s0 = s1 = 0.
                for c_, b_, n_ in zip(z['cell'][sel], z['by'][sel], z['n'][sel]):
                    ti = np.flatnonzero((yrs >= b_) & (yrs <= b_ + 4))
                    if ti.size:
                        s0 += n_ * Sa[ti, c_].mean(); s1 += n_ * (Sa[ti, c_] * (1 - m[ti, c_])).mean(); w += n_
                out['sa_banded'][f'{gname} {lab}'] = dict(released=float(w), Sa=(s0 / w if w else None), Sa_with_disease=(s1 / w if w else None))
    # REVIEW 9 C1: does capacity K or journey survival Q carry a modern change the rate readouts miss? K / Q as the
    # FORWARD receives them (decoupled arms: K from the real-Z pass) -- pass k_land / q_land to override.
    out['K_Q'] = kq_stats(sb, x, occ, (to_land(k_flat, sb.rows, sb.cols, len(sb.years), sb.Nx) if k_flat is not None else None))
    out['latents'] = {k: np.asarray(v).round(4).tolist() for k, v in sb.lat.items() if k.startswith('disease')}
    out['loglik_fit'] = {k: v['loglik'] for k, v in fit['abundance'].items()}
    if off:
        fa = dict(arm_extra.pop('forward_args', {}) or {}); fa['disease_m_amp'] = jnp.zeros_like(jnp.asarray(x['disease_m_amp']))
        offr = sb.run(dict(name='disease_off', forward_args=fa, **arm_extra))
        out['loglik_disease_off'] = {k: v['loglik'] for k, v in offr['abundance'].items()}
        out['delta_nll_off_minus_fit'] = {k: -(out['loglik_disease_off'][k] - out['loglik_fit'][k]) for k in out['loglik_fit']}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sb = Sandbox(os.environ['SBX_REFINED'])
    out = disease_stats(sb)
    a.out.write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps({k: out[k] for k in ('by_region', 'delta_nll_off_minus_fit')}, indent=1))


if __name__ == '__main__':
    main()
