"""Quantify original mean2 versus mean4 sensitivity using completed, fixed scores."""
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
from final_winter_score import ERAS, paired_circle_summary

p = argparse.ArgumentParser(description=__doc__)
for name in ('mean4', 'mean2', 'cbc', 'out'):
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
from research.adopt import record_read
record_read(str(a.cbc), 'FINAL_ENSEMBLE: original mean2 versus mean4 sensitivity on previously opened observations')
for root, count in [(a.mean4, 4), (a.mean2, 2)]:
    for family in ('final', 'final_v2'):
        meta = json.loads((root / 'features' / family / 'DONE.json').read_text())
        assert len(meta['members']) == count, (root, family)
result = {'interpretation': 'Original encoders only. Mean2 minus mean4; no clean-completeness intervention.',
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'summer': {}, 'winter': {}}
arms = ('v1_r64', 'v2_r64', 'v2_r128')
for thin, seed in [('1', 1), ('0.03', 1), ('0.03', 2)]:
    block = {}
    for arm in arms:
        x = components(a.mean4 / 'summer' / f'{arm}_e1966_1986_f{thin}_s{seed}')
        y = components(a.mean2 / 'summer' / f'{arm}_f{thin}_s{seed}')
        assert np.array_equal(x['columns'], y['columns'])
        np.testing.assert_allclose(x['vt'][0], y['vt'][0], rtol=1e-6, atol=1e-8)
        paired = np.array_equal(x['vt'], y['vt'])
        block[arm] = {}
        for rd in ('desk_within', 'desk_pooled'):
            old, new = correlation(x, rd), correlation(y, rd)
            delta = estimate(new-old) if paired else {'value': float(new[0]-old[0]), 'ci95': None,
                     'note': 'Truth bootstrap draws differ across environments; no paired interval reported.'}
            block[arm][rd] = {'mean4': float(old[0]), 'mean2': float(new[0]), 'mean2_minus_mean4': delta}
    result['summer'][f'f{thin}/s{seed}'] = block
old_meta, new_meta = [json.loads((root / 'winter/result.json').read_text()) for root in (a.mean4, a.mean2)]
for key in ('species', 'calibration_rows', 'test_rows'):
    assert old_meta[key] == new_meta[key], key
values = {}; rows = None
for arm in arms:
    with np.load(a.mean4 / f'winter/lists_{arm}.npz') as old, np.load(a.mean2 / f'winter/lists_{arm}.npz') as new:
        assert np.array_equal(old['test_rows'], new['test_rows']) and np.array_equal(old['species'], new['species'])
        if rows is not None:
            assert np.array_equal(rows, old['test_rows'])
        rows = old['test_rows'].copy()
        values[f'{arm}/backcast'] = new['backcast']-old['backcast']
        values[f'{arm}/temporal_gain'] = (new['backcast']-new['modern'])-(old['backcast']-old['modern'])
with np.load(a.cbc) as data:
    year, circle = data['year_end'][rows], data['circle_idx'][rows]
for name, lo, hi in ERAS:
    mask = (year >= lo) & (year <= hi)
    result['winter'][name] = paired_circle_summary(circle[mask], {key: value[mask] for key, value in values.items()})
a.out.write_text(json.dumps(result, indent=2, allow_nan=False))
for regime, block in result['summer'].items():
    print(regime, {arm: {rd: metric['mean2_minus_mean4'] for rd, metric in scores.items()} for arm, scores in block.items()})
