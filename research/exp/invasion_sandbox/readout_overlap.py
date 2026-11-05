"""Does a fitted Z readout (R16 / zreadout.py) just re-use the fit's own habitat-change signal?

For a zreadout summary (its b and intercept), rebuild the readout field log G(x, t) = g0 + z(x, t) . b on the model's
own standardised Z, and compare it with the FITTED model's potential log lambda0(x, t) (unfrozen habitat, harness
potential_lambda), and with each fitted rate (log Fmax, logit Sa, logit Sj, log K). Pearson over eastern (and western)
land cell-years, over cell-mean-removed values (the TIME part), and over the 1940 -> 2010 change per cell.
High overlap = the readout re-routes the same Z direction the fit uses for inflation.
Usage (inside sbx_env.sh): python readout_overlap.py SUMMARY.json --out DIR
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from harness import Sandbox, capture, to_land, potential_lambda


def corr(a, b):
    a, b = a.ravel(), b.ravel(); m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary', type=Path); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    s = json.loads(a.summary.read_text())
    sb = Sandbox(os.environ['SBX_REFINED'])
    Z = np.asarray(sb.data['Z_gathered'], float)
    mu, sd = Z.reshape(-1, Z.shape[-1]).mean(0), Z.reshape(-1, Z.shape[-1]).std(0) + 1e-9
    z = (Z - mu) / sd
    logG = s['intercept'] + z @ np.asarray(s['b'])                      # (T, N_land)
    x, _ = capture(sb.data, sb.lat)
    T, Nx = len(sb.years), int(np.asarray(sb.data['land_mask']).shape[1])
    R = {k: to_land(x[f'{k}_flat'], sb.rows, sb.cols, T, Nx) for k in ('Sa', 'Sj', 'Fmax', 'K')}
    fields = dict(log_lambda0=np.log(np.maximum(potential_lambda(R['Sa'], R['Sj'], R['Fmax'], bool(x.get('yearling_breeding'))), 1e-9)),
                  log_Fmax=np.log(np.maximum(R['Fmax'], 1e-9)), logit_Sa=np.log(R['Sa'] / (1 - R['Sa'])),
                  logit_Sj=np.log(R['Sj'] / (1 - R['Sj'])), log_K=np.log(np.maximum(R['K'], 1e-12)))
    i0, i1 = int(np.flatnonzero(sb.years == 1940)[0]), int(np.flatnonzero(sb.years == 2010)[0])
    out = dict(summary=str(a.summary), mode=s['mode'], tag=a.summary.name)
    for reg in ('east', 'west'):
        m = sb.region[reg]; r = {}
        for k, f in fields.items():
            g, h = logG[:, m], f[:, m]
            r[k] = dict(all=corr(g, h), time_part=corr(g - g.mean(0), h - h.mean(0)),
                        change_1940_2010=corr(logG[i1, m] - logG[i0, m], f[i1, m] - f[i0, m]))
        out[reg] = r
    (a.out / f'overlap_{a.summary.stem}.json').write_text(json.dumps(out, indent=1) + '\n')
    for reg in ('east', 'west'):
        print(reg, {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in out[reg].items()})


if __name__ == '__main__':
    main()
