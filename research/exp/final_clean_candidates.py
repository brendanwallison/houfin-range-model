"""Compare completed clean candidates on the same previously opened observations."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_clean_compare import components
from final_holdout_summary import correlation, estimate
from final_winter_score import ERAS, paired_circle_summary
from research.adopt import record_read

p = argparse.ArgumentParser(description=__doc__)
for name in ('root', 'cbc', 'out'):
    p.add_argument('--' + name, required=True, type=Path)
p.add_argument('--arms', nargs='+', default=['v2_r64', 'v2_r128'])
p.add_argument('--winter-only', action='store_true')
a = p.parse_args()
assert (a.root / 'EVALUATION_DONE').exists()
record_read(str(a.cbc), 'FINAL_CLEAN_CANDIDATES: paired comparisons of completed reused holdouts')
result = {'timestamp': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'root': str(a.root), 'summer': {}, 'winter': {},
          'interpretation': 'Clean seeds1/2; v2 minus v1. Fixed models and reused observations, not fresh confirmation.'}
arms = tuple(a.arms)
for thin, seed in ([] if a.winter_only else [('1', 1), ('0.03', 1), ('0.03', 2)]):
    ref = components(a.root / 'summer' / f'v1_r64_f{thin}_s{seed}')
    block = {}
    for arm in arms:
        new = components(a.root / 'summer' / f'{arm}_f{thin}_s{seed}')
        assert np.array_equal(ref['columns'], new['columns'])
        assert np.array_equal(ref['vt'], new['vt'])
        block[arm] = {rd: estimate(correlation(new, rd) - correlation(ref, rd))
                      for rd in ('desk_within', 'desk_pooled')}
    result['summer'][f'f{thin}/s{seed}'] = block
with np.load(a.root / 'winter/lists_v1_r64.npz') as data:
    ref = {k: data[k].copy() for k in data.files}
values = {}
for arm in arms:
    with np.load(a.root / f'winter/lists_{arm}.npz') as new:
        assert np.array_equal(ref['test_rows'], new['test_rows'])
        assert np.array_equal(ref['species'], new['species'])
        values[arm + '/backcast'] = new['backcast'] - ref['backcast']
        values[arm + '/temporal_gain'] = (new['backcast'] - new['modern']) - (ref['backcast'] - ref['modern'])
with np.load(a.cbc) as data:
    rows = ref['test_rows']
    year, circle, counters = data['year_end'][rows], data['circle_idx'][rows], data['field_counters'][rows]
    modern = np.bincount(circle[year >= 2005], minlength=len(data['circles']))
for era, lo, hi in ERAS:
    primary = (year >= lo) & (year <= hi)
    counts = np.bincount(circle[primary], minlength=len(modern))
    masks = {'primary': primary, 'same_circles': primary & (counts[circle] >= 3) & (modern[circle] >= 3),
             'counters_1_6': primary & (counters > 0) & (counters <= 6)}
    result['winter'][era] = {name: paired_circle_summary(circle[mask], {k: v[mask] for k, v in values.items()})
                             for name, mask in masks.items()}
tmp = a.out.with_suffix('.tmp.json')
tmp.write_text(json.dumps(result, indent=2, allow_nan=False)); tmp.replace(a.out)
print(json.dumps({'summer': result['summer'], 'winter_primary': {
    era: block['primary'] for era, block in result['winter'].items()}}, indent=2))
