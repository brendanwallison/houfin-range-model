"""Paired summaries of fixed, matched old/clean two-seed evaluations."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_holdout_summary import correlation, estimate
from final_winter_score import ERAS, paired_circle_summary

ARMS = ('v1_r64', 'v2_r64', 'v2_r128')


def components(directory):
    with np.load(directory / 'species_components_final.npz') as source:
        keys = ['columns', 'vt'] + [f'{arm}__{field}' for arm in ('desk_within', 'desk_pooled') for field in ('cv', 'vp')]
        return {key: source[key] if key == 'columns' else source[key].astype('float64') for key in keys}


def compare(reference, candidate, cbc):
    result = {'timestamp': datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds'),
              'provenance': {'reference': str(reference.resolve()), 'candidate': str(candidate.resolve()),
                             'summary_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'helper_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                               for name in ('final_holdout_summary.py', 'final_winter_score.py')},
                             'versions': {'numpy': np.__version__, 'python': sys.version},
                             'feature_manifests': {str(root / 'features' / family / 'DONE.json'):
                                 hashlib.sha256((root / 'features' / family / 'DONE.json').read_bytes()).hexdigest()
                                 for root in (reference, candidate) for family in ('final', 'final_v2')}},
              'summer': {}, 'winter': {}, 'interpretation':
              'Matched seeds1/2, fixed scoring and reused heldouts. Completeness intervention is not fresh confirmation; encoder training spans the full timeline.'}
    for thin, seed in [('1', 1), ('0.03', 1), ('0.03', 2)]:
        block = {}
        for arm in ARMS:
            name = f'{arm}_f{thin}_s{seed}'
            old, new = components(reference / 'summer' / name), components(candidate / 'summer' / name)
            if not np.array_equal(old['columns'], new['columns']) or not np.array_equal(old['vt'], new['vt']):
                raise ValueError(f'unpaired summer targets/truth/bootstrap: {name}')
            block[arm] = {}
            for readout in ('desk_within', 'desk_pooled'):
                a, b = correlation(old, readout), correlation(new, readout)
                for directory, value in [(reference, a), (candidate, b)]:
                    direct = json.loads((directory / 'summer' / name / 'fair_rivals_incr_final.json').read_text())
                    if not np.isclose(value[0], direct['models'][readout]['corr'], atol=1e-6):
                        raise ValueError(f'component score differs from direct scorer: {name}/{readout}')
                block[arm][readout] = {'original': estimate(a), 'clean': estimate(b),
                                       'clean_minus_original': estimate(b-a)}
        result['summer'][f'f{thin}/s{seed}'] = block
    old_meta = json.loads((reference / 'winter/result.json').read_text())
    new_meta = json.loads((candidate / 'winter/result.json').read_text())
    for key in ('species', 'calibration_rows', 'test_rows', 'relocated_test_circles_removed_from_calibration'):
        if old_meta[key] != new_meta[key]:
            raise ValueError(f'winter supports/calibration differ: {key}')
    values = {}
    rows = None
    for arm in ARMS:
        with np.load(reference / f'winter/lists_{arm}.npz') as old, np.load(candidate / f'winter/lists_{arm}.npz') as new:
            if not np.array_equal(old['test_rows'], new['test_rows']) or not np.array_equal(old['species'], new['species']):
                raise ValueError(f'unpaired winter support: {arm}')
            if rows is not None and not np.array_equal(rows, old['test_rows']):
                raise ValueError('winter arms have different rows')
            rows = old['test_rows'].copy()
            values[f'{arm}/clean_minus_original_backcast'] = new['backcast']-old['backcast']
            values[f'{arm}/clean_minus_original_temporal_gain'] = (new['backcast']-new['modern'])-(old['backcast']-old['modern'])
    with np.load(cbc) as data:
        year, circle, counters = data['year_end'][rows], data['circle_idx'][rows], data['field_counters'][rows]
        n_circles = len(data['circles'])
    modern = np.bincount(circle[year >= 2005], minlength=n_circles)
    for era, lo, hi in ERAS:
        primary = (year >= lo) & (year <= hi)
        counts = np.bincount(circle[primary], minlength=n_circles)
        masks = {'primary': primary,
                 'same_circles': primary & (counts[circle] >= 3) & (modern[circle] >= 3),
                 'counters_1_6': primary & (counters > 0) & (counters <= 6)}
        result['winter'][era] = {label: paired_circle_summary(circle[mask], {k: v[mask] for k, v in values.items()})
                                  for label, mask in masks.items()}
    result['winter_support'] = {key: old_meta[key] for key in ('species', 'calibration_rows', 'test_rows')}
    return result


def report(result, destination):
    def fmt(metric):
        if not metric or metric.get('value') is None:
            return 'unsupported'
        if metric.get('ci95') is None:
            return f"{metric['value']:+.3f} [interval unavailable]"
        lo, hi = metric['ci95']
        return f"{metric['value']:+.3f} [{lo:+.3f},{hi:+.3f}]"
    lines = ['# Matched clean-completeness evaluation', '', f"Updated {result['timestamp']}.", '', result['interpretation'], '',
             'Positive differences favor the clean encoder. Both ensembles use seeds1/2; this does not compare a two-seed model to a four-seed model.', '',
             '## Summer', '', '1966–1986 to2005–2025; modern target calibration1996+. Noise-corrected change correlation with the existing paired6×6 block bootstrap.', '',
             '| Calibration / thinning seed | Representation | Within clean−original [95%] | Pooled clean−original [95%] |', '|---|---|---|---|']
    for regime, arms in result['summer'].items():
        for arm, scores in arms.items():
            lines.append(f"|{regime}|{arm}|{fmt(scores['desk_within']['clean_minus_original'])}|{fmt(scores['desk_pooled']['clean_minus_original'])}|")
    lines += ['', '## Winter', '', 'Conditional list-composition score with the existing circle bootstrap. Temporal gain subtracts each fitted model’s own fixed modern-feature placebo; it is not abundance truth.', '',
              '| Era | Lists / circles | v1 clean−original temporal gain | v2 32+32 | v2 64+64 |', '|---|---|---|---|---|']
    for era, summaries in result['winter'].items():
        item = summaries['primary']; metrics = item.get('metrics', {})
        values = [fmt(metrics.get(f'{arm}/clean_minus_original_temporal_gain')) for arm in ARMS]
        lines.append(f"|{era}|{item['lists']}/{item['circles']}|"+'|'.join(values)+'|')
    lines += ['', 'JSON retains absolute summer scores, winter absolute backcast differences, same-circle and low-counter sensitivities. Supports, species, calibration counts and paired bootstrap truth must match or this summary fails. Sparse repeats share test observations; they are not independent ecological replications. No production adoption is automated.', '']
    temporary = destination.with_suffix('.tmp.md')
    temporary.write_text('\n'.join(lines), encoding='utf-8'); temporary.replace(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('reference', 'candidate', 'cbc', 'out'):
        p.add_argument('--'+flag, required=True, type=Path)
    p.add_argument('--report', type=Path)
    p.add_argument('--identity-check', action='store_true')
    a = p.parse_args()
    from research.adopt import record_read
    record_read(str(a.cbc), 'FINAL_CLEAN: paired summaries of previously opened holdout scores; no new confirmation')
    result = compare(a.reference, a.candidate, a.cbc)
    if a.identity_check:
        assert a.reference.resolve() == a.candidate.resolve()
        for block in result['summer'].values():
            for scores in block.values():
                for metric in scores.values():
                    assert metric['clean_minus_original']['value'] == 0
        for summaries in result['winter'].values():
            for item in summaries.values():
                for metric in item.get('metrics', {}).values():
                    assert metric['value'] == 0 and metric['ci95'] == [0., 0.]
    temporary = a.out.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)); temporary.replace(a.out)
    if a.report:
        report(result, a.report)
    print('Saved paired comparison', a.out)
