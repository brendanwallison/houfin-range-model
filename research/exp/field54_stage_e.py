"""FIELD54 Stage E helpers: the pre-registered inner-grid choice and the pre-registered twin decision rules.

Rules and grid are fixed in research/experiments/FIELD54_FIELD_2026-10-30.md BEFORE any score exists.

    python research/exp/field54_stage_e.py select --root R            # inner grid -> FIELD54_INNER_CHOICE.json
    python research/exp/field54_stage_e.py rules --root R --block-cells 3   # ASSIM_F54_*.json -> FIELD54_RULES_b3.json
"""
import argparse
import json
from pathlib import Path

TASK = 'backcast40'
DEFAULT = {'lam_lap': 1.0, 'lam_u': 1e-3}       # the 27 km field_d_interp setting (z_assim.DEFAULT)
LAM_LAP = [0.25, 0.5, 1.0, 2.0, 4.0]
LAM_U = [1e-4, 1e-3, 1e-2]
TIE = 0.005                                     # inner regional corr: within this of the best, keep the 27 km default
STRATA = ('sed_common', 'sedentary', 'all')


def inner_grid():
    """The pre-registered inner grid: arm field_d over lam_lap x lam_u, everything else the 27 km default."""
    return [['field_d', {'lam_lap': l, 'lam_u': u}] for l in LAM_LAP for u in LAM_U]


def select(results, tie=TIE):
    """Best inner REGIONAL corr for field_d; the 27 km default wins any tie within `tie` of the best (pure)."""
    rows = [r for r in results if r['arm'] == 'field_d' and r['cfg'] is not None]
    if not rows:
        raise ValueError('No field_d inner results')
    best = max(rows, key=lambda r: r['regional'])
    default = [r for r in rows if all(r['cfg'].get(k) == v for k, v in DEFAULT.items())]
    if default and best['regional'] - default[0]['regional'] <= tie:
        return dict(DEFAULT), {'reason': f'27 km default within {tie} of the best inner regional', 'best': best,
                               'default': default[0]}
    return {k: best['cfg'][k] for k in DEFAULT}, {'reason': 'best inner regional', 'best': best,
                                                 'default': default[0] if default else None}


def get(res, ref, arm, stratum, comp='regional'):
    d = res['paired_vs_' + ref][arm][f'k2_spatial_{stratum}'][comp]['corr']
    a = res['arms'][arm]['species'][f'k2_spatial_{stratum}'][comp]['corr']
    r = res['arms'][ref]['species'][f'k2_spatial_{stratum}'][comp]['corr']
    return {'arm': a['point'], 'ref': r['point'], 'diff': d['point'], 'ci95': d['ci95']}


def rules(root, bc):
    """Apply plan rules 1-3 to the scored JSONs (withheld places, k2, regional)."""
    a = root/'assim'/TASK
    sfx = f'_b{bc}'
    main = json.loads((a/f'ASSIM_F54_MAIN{sfx}.json').read_text())          # ref desk_interp_s1
    vsema = json.loads((a/f'ASSIM_F54_VSEMA{sfx}.json').read_text())        # ref desk_interp_s1_ema
    ema = json.loads((a/f'ASSIM_F54_EMAVSRAW{sfx}.json').read_text())       # ref field_d_interp_s1
    out = {'block_cells': bc, 'rule2_D_choice': {}, 'rule1_native_vs_averaged': {}, 'rule3_sanity': {}}
    # Rule 2: D = EMA vs raw. Larger regional wins; |diff| < .01 -> EMA. Decided on sedentary common; all species
    # reported; if the two strata disagree the rule is ambiguous and is flagged, not resolved here.
    picks = {}
    for s in STRATA:
        g = get(ema, 'field_d_interp_s1', 'field_d_interp_ema_s1', s)
        pick = 'ema' if g['diff'] is None or g['diff'] > -0.01 else 'raw'
        out['rule2_D_choice'][s] = {**g, 'pick': pick}
        picks[s] = pick
    winner = picks['sed_common']
    out['rule2_D_choice']['decision'] = winner
    out['rule2_D_choice']['strata_agree'] = picks['sed_common'] == picks['all']
    field = 'field_d_interp_ema_s1' if winner == 'ema' else 'field_d_interp_s1'
    # Rule 1: native vs block-averaged 27 km field_d_interp on the same 54 km panel: native "at least as good" if
    # diff >= -0.02 for sedentary common AND all species.
    r1 = json.loads((a/f'ASSIM_F54_NATIVEVSAVG{sfx}.json').read_text())    # ref fdinterp27_avg54_s1
    for s in STRATA:
        out['rule1_native_vs_averaged'][s] = get(r1, 'fdinterp27_avg54_s1', field, s)
    ok = [out['rule1_native_vs_averaged'][s]['diff'] is not None and out['rule1_native_vs_averaged'][s]['diff'] >= -0.02
          for s in ('sed_common', 'all')]
    out['rule1_native_vs_averaged']['native_at_least_as_good'] = all(ok)
    # Rule 3: the field beats the 54 km control DESK, raw and EMA, at withheld places (point estimates; CI reported).
    for s in STRATA:
        out['rule3_sanity'][s] = {'vs_raw': get(main, 'desk_interp_s1', field, s),
                                  'vs_ema': get(vsema, 'desk_interp_s1_ema', field, s)}
        f, e = out['rule3_sanity'][s]['vs_raw'], out['rule3_sanity'][s]['vs_ema']
        out['rule3_sanity'][s]['field_beats_raw'] = f['diff'] is not None and f['diff'] > 0
        out['rule3_sanity'][s]['field_beats_ema'] = e['diff'] is not None and e['diff'] > 0
    out['field_arm_used'] = field
    (a/f'FIELD54_RULES{sfx}.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('stage', choices=['grid', 'select', 'rules'])
    p.add_argument('--root', type=Path)
    p.add_argument('--block-cells', type=int, default=3)
    a = p.parse_args()
    if a.stage == 'grid':
        print(json.dumps(inner_grid()))
        return
    out = a.root/'assim'/TASK
    if a.stage == 'select':
        results = []
        for f in sorted(out.glob('INNER_*of*.json')):
            results += json.loads(f.read_text())['results']
        cfg, why = select(results)
        (out/'FIELD54_INNER_CHOICE.json').write_text(json.dumps({'cfg': cfg, **why, 'n_results': len(results)},
                                                                 indent=2))
        print(json.dumps(cfg))
    else:
        rules(a.root, a.block_cells)


if __name__ == '__main__':
    main()
