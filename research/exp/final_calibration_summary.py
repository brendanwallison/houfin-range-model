"""Summarize the completed fixed calibration-size experiment; no fitting or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_clean_compare import components
from final_holdout_summary import correlation, estimate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
p.add_argument('--figure', required=True, type=Path)
a = p.parse_args()
r = a.root
arms = ('v1_r64', 'v2_r64', 'v2_r128')
entries = []
for fraction, seeds in [('0.03', [1, 2]), ('0.1', [1, 2]), ('1', [1])]:
    for seed in seeds:
        directories = {arm: (r / 'calibration_curve/f0.1' / f'{arm}_s{seed}' if fraction == '0.1'
                            else r / 'summer' / f'{arm}_e1966_1986_f{fraction}_s{seed}') for arm in arms}
        if not all((path / 'DONE').exists() for path in directories.values()):
            raise SystemExit(f'Waiting for complete fraction{fraction}/seed{seed}; no partial plot written')
        values = {arm: components(path) for arm, path in directories.items()}
        meta = {arm: json.loads((path / 'fair_rivals_incr_final.json').read_text()) for arm, path in directories.items()}
        reference = values['v1_r64']
        entry = {'fraction': float(fraction), 'thinning_seed': seed,
                 'fit_rows': meta['v1_r64']['n_fit_rows'], 'test_cells': meta['v1_r64']['n_cells'], 'differences': {}}
        for arm, value in values.items():
            assert np.array_equal(value['columns'], reference['columns']) and np.array_equal(value['vt'], reference['vt'])
            assert meta[arm]['n_fit_rows'] == entry['fit_rows'] and meta[arm]['n_cells'] == entry['test_cells']
            if arm == 'v1_r64':
                continue
            entry['differences'][arm] = {}
            for readout in ('desk_within', 'desk_pooled'):
                current, baseline = correlation(value, readout), correlation(reference, readout)
                assert np.isclose(current[0], meta[arm]['models'][readout]['corr'], atol=1e-6)
                assert np.isclose(baseline[0], meta['v1_r64']['models'][readout]['corr'], atol=1e-6)
                entry['differences'][arm][readout] = estimate(current-baseline)
        entries.append(entry)
result = {'interpretation': 'Original four-seed encoders; fixed3%,10%,100% calibration, reused heldouts. Confidence intervals condition on each fitted readout; this is not clean training or fresh confirmation.',
          'evaluation_windows': {'early': [1966, 1986], 'modern': [2005, 2025], 'calibration': [1996, 2025]},
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'entries': entries}
from research.adopt import record_read
record_read(str(r / 'calibration_curve'), 'FINAL_CALIBRATION: fixed calibration-size summary of previously opened holdout score components')
a.out.write_text(json.dumps(result, indent=2, allow_nan=False))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3), sharex=True, sharey=True, layout='constrained')
colors = {1: '#2563a6', 2: '#d07418'}
for row, arm in enumerate(('v2_r64', 'v2_r128')):
    for col, readout in enumerate(('desk_within', 'desk_pooled')):
        axis = axes[row, col]
        axis.axhline(0, color='0.4', linewidth=.8, linestyle='--')
        for seed in (1, 2):
            subset = [e for e in entries if e['thinning_seed'] == seed and e['fraction'] < 1]
            full = next(e for e in entries if e['fraction'] == 1)
            line = subset+[full]
            axis.plot([e['fraction'] for e in line], [e['differences'][arm][readout]['value'] for e in line], color=colors[seed], alpha=.65, linewidth=1)
            for entry in subset:
                x = entry['fraction']; metric = entry['differences'][arm][readout]
                lo, hi = metric['ci95']
                axis.vlines(x, lo, hi, color=colors[seed], linewidth=1.4)
                axis.scatter(x, metric['value'], color=colors[seed], marker='o' if seed == 1 else 's', s=25, label=f'Thinning seed {seed}' if row == 0 and col == 0 and x == .03 else None)
        metric = full['differences'][arm][readout]; lo, hi = metric['ci95']
        axis.vlines(1., lo, hi, color='black', linewidth=1.4)
        axis.scatter(1., metric['value'], color='black', marker='D', s=28, label='Full calibration (shared)' if row == 0 and col == 0 else None)
        axis.set_xscale('log'); axis.set_xlim(.022, 1.3)
        axis.set_xticks([.03, .1, 1.], ['3%', '10%', '100%'])
        axis.grid(axis='y', alpha=.18)
        axis.set_title(('32 summer + 32 winter' if row == 0 else '64 summer + 64 winter')+' | '+('within-cell' if col == 0 else 'pooled'))
        if col == 0:
            axis.set_ylabel('Δ noise-corrected correlation vs v1')
        if row == 1:
            axis.set_xlabel('Modern calibration rows retained')
axes[0, 0].legend(fontsize=8, loc='lower left')
fig.suptitle('Original four-seed DESK: calibration-size sensitivity\nPaired 95% block-bootstrap intervals; fixed fitted encoders', fontsize=12)
fig.savefig(a.figure, dpi=200)
plt.close(fig)
print('Saved', a.out, a.figure)
for entry in entries:
    print(entry['fraction'], entry['thinning_seed'], entry['fit_rows'], entry['differences']['v2_r128'])
