"""Quick point-estimate preview of cached z_assim_evaluate raw scores (no CIs; the combine step has them)."""
import pickle, sys
from pathlib import Path
raw = Path(sys.argv[1])
f = lambda v: f"{v[0]:.3f}"
print('arm | panel | stratum | all | regional | local | slope | mse_gain | hist_mse | mod_mse | teacher all/reg | winter gain 1966-85 / 1920-39')
for p in sorted(raw.glob('*.pkl')):
    sp, te, lam, w = pickle.loads(p.read_bytes())
    for panel in ('known', 'spatial'):
        for s in ('sed_common', 'all'):
            e = sp[f'k2_{panel}_{s}']
            t = te[panel]
            tt = f"{t['all']['corr']['point']:.3f}/{t.get('regional', {}).get('corr', {}).get('point', float('nan')):.3f}" if s == 'all' else ''
            ww = ''
            if w is not None and s == 'all' and panel == 'known':
                ww = ' / '.join(f"{w[0][e_]['metrics']['backcast_minus_modern']['value']:+.4f}" for e_ in ('1966_1985', '1920_1939'))
            print(f"{p.stem} | {panel} | {s} | {f(e['all']['corr'])} | {f(e['regional']['corr'])} | {f(e['local']['corr'])} | "
                  f"{f(e['all']['truth_on_prediction_slope'])} | {f(e['all']['mse_gain_over_no_change'])} | "
                  f"{e['historical_mse']['corr'][0]:.4f} | {e['modern_mse']['corr'][0]:.4f} | {tt} | {ww}")
