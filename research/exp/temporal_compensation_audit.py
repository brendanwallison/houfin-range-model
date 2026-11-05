"""Fixed-support historical growth changes; descriptive accounting, not causal attribution.

Read already collected frozen trajectory budgets. No fitting, source modification,
population weighting, or new biological assumptions. Report adult survival and
the combined juvenile-survival/reproductive contribution separately: their sum
is the current yearling model's potential growth multiplier.
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform

REPO = Path('/mnt/c/Dev/houfin-range-model')
ROOT = Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')
EXP = REPO / 'research/experiments'
STEPS = [
    'step38_cbc1951_noncbc_free_s41', 'step38_cbc1951_noncbc_free_s42',
    'step38_cbc1951_free_s41',
    'step39_veit_fixed_s41', 'step39_veit_fixed_s42',
    'step39_kfloor1p25_s41', 'step39_kfloor1p25_s42',
    'step39_kfloor4p0_s41', 'step39_kfloor4p0_s42',
    'step41_rational_adult20_s41',
    'step39_adult_priority_fixed_s41',
    'step40_rangev2_historical_s41', 'step40_rangev2_historical_s42',
    'step40_rangev2_invasion_s41', 'step40_rangev2_invasion_s42',
]
ERAS = {'1902-1915': (1902, 1915), '1951-1965': (1951, 1965),
        '1966-1985': (1966, 1985), '1986-2009': (1986, 2009),
        '2010-2025': (2010, 2025)}


def mean_change_parts(lam0, sa0, lam1, sa1):
    """Exact additive identity, without separating the unidentifiable Sj/F pair."""
    adult = float(np.mean(sa1) - np.mean(sa0))
    reproductive = float(np.mean(lam1 - sa1) - np.mean(lam0 - sa0))
    delta = float(np.mean(lam1) - np.mean(lam0))
    if not np.isclose(delta, adult + reproductive, atol=1e-12, rtol=1e-12):
        raise ValueError('potential growth decomposition does not close')
    return dict(delta=delta, adult_survival=adult,
                juvenile_survival_times_reproduction=reproductive)


def main():
    inventory = {r['step']: r for r in json.loads(
        (EXP / 'FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json').read_text())['runs']}
    steps = list(STEPS)
    for filename in ['GRADUAL_DEPARTURE_RUNS_20261102.json', 'CROWDING_CURVATURE_RUNS_20261102.json',
                     'KFLOOR_COMBINATION_RUNS_20261102.json','MINOR_FLEXIBILITY_RUNS_20261103.json','OUTSTANDING_TRANSFER_RUNS_20261103.json']:
        path = EXP / filename
        if path.exists():
            for spec in json.loads(path.read_text()):
                if spec['step'] not in steps:
                    steps.append(spec['step'])
    records, missing, definitions, grid_ref = [], [], {}, None
    max_cached_error = 0.0
    for step in steps:
        path = ROOT / step / 'annual_budget.npz'
        if not path.exists() or step not in inventory or not inventory[step].get('valid'):
            missing.append(step)
            continue
        if not inventory[step]['config']['population_model']['reproduction'].get('yearling_breeding'):
            raise ValueError(f'unsupported life cycle: {step}')
        with np.load(path) as z:
            years, rr, cc = z['years'], z['land_rows'], z['land_cols']
            lam, sa = z['potential_lambda'], z['local_adult_survival']
            full_lam = z['potential_lambda_with_process'] if 'potential_lambda_with_process' in z else lam
        if grid_ref is None:
            grid_ref = (years.copy(), rr.copy(), cc.copy())
            with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
                xx, yy = rasterio.transform.xy(src.transform, rr, cc)
                lon, lat = map(np.asarray, transform(src.crs, 'EPSG:4326', xx, yy))
            regions = {'all_land': np.ones(len(rr), bool), 'east': (lon >= -80) & (lat >= 25) & (lat < 49),
                       'plains': (lon >= -100) & (lon < -95), 'west': lon < -100}
            # Also provide an exhaustive three-region partition of one common
            # latitude belt, rather than mislabeling the old NE mask all East.
            belt = (lat >= 25) & (lat < 49)
            regions.update(east_belt=(lon >= -95) & belt,
                           plains_belt=(lon >= -100) & (lon < -95) & belt,
                           west_belt=(lon < -100) & belt)
            if not np.array_equal(sum(regions[k].astype(int) for k in
                                      ['east_belt', 'plains_belt', 'west_belt']), belt.astype(int)):
                raise ValueError('regional belt masks must partition latitude support')
            definitions = {k: dict(n_cells=int(v.sum())) for k, v in regions.items()}
            definitions.update(belt_description='25<=latitude<49; West<-100, Plains[-100,-95), East>=-95. Legacy east is NE>=-80, not the entire East.')
        elif not all(np.array_equal(a, b) for a, b in zip(grid_ref, (years, rr, cc))):
            raise ValueError(f'time/grid mismatch: {step}')
        if not (np.isfinite(lam).all() and np.isfinite(sa).all()):
            raise ValueError(f'nonfinite fields: {step}')
        for region, mask in regions.items():
            fields = {}
            for era, (lo, hi) in ERAS.items():
                tt = (years >= lo) & (years <= hi)
                if int(tt.sum()) != hi - lo + 1:
                    raise ValueError(f'incomplete era: {step} {era}')
                fields[era] = (lam[tt][:, mask], sa[tt][:, mask])
            early, modern = fields['1902-1915'], fields['2010-2025']
            total = mean_change_parts(*early, *modern)
            # Independently average each cell first and then difference. This also
            # retains spatial variation hidden by a continental mean.
            cell_delta = modern[0].mean(0) - early[0].mean(0)
            np.testing.assert_allclose(cell_delta.mean(), total['delta'], atol=1e-12)
            cached = inventory[step].get('pathology', {}).get('regions', {}).get(region, {})
            errors = []
            for key, era in [('early', '1902-1915'), ('modern', '2010-2025')]:
                value = cached.get(key, {}).get('lambda0', {}).get('land', {}).get('mean')
                if value is not None:
                    errors.append(float(fields[era][0].mean() - value))
            max_cached_error = max(max_cached_error, *(abs(v) for v in errors), 0.)
            transitions = []
            names = list(ERAS)
            for first, last in zip(names[:-1], names[1:]):
                transitions.append(dict(first=first, last=last,
                                        **mean_change_parts(*fields[first], *fields[last])))
            np.testing.assert_allclose(sum(t['delta'] for t in transitions), total['delta'], atol=1e-12)
            records.append(dict(step=step, region=region, **total,
                                full_process_era_means={era:float(full_lam[(years>=lo)&(years<=hi)][:,mask].mean()) for era,(lo,hi) in ERAS.items()},
                                full_process_delta=float(full_lam[(years>=2010)&(years<=2025)][:,mask].mean()-full_lam[(years>=1902)&(years<=1915)][:,mask].mean()),
                                era_means={k: float(v[0].mean()) for k, v in fields.items()},
                                cell_delta_q10_50_90=np.quantile(cell_delta, [.1, .5, .9]).tolist(),
                                transitions=transitions, cached_lambda_errors=errors))
    out = dict(definition='Potential lambda=Sa+Sj*Fmax. Equal fixed land-cell weights in every era/model.',
               caveat='Additive rate accounting is not causal attribution. Productive growth, movement and observed counts require separate diagnostics.',
               regions=definitions, eras=ERAS, missing=missing,
               max_cached_lambda_error=max_cached_error, records=records)
    (EXP / 'TEMPORAL_COMPENSATION_AUDIT_20261103.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = ['# Fixed-support temporal growth accounting', '', out['definition'], '', out['caveat'], '',
             'Belt regions exhaustively partition land within 25-49N: West west of 100W, Plains 100-95W, East east of 95W. Legacy east is only east of 80W within 25-49N; legacy Plains/West retain the full grid latitude range. These supports are reported separately, not pooled.', '',
             'Biological change excludes fitted abundance innovations. Full-process change includes their annual multipliers; their cumulative population effect is separately tested by zero-innovation replay. These are different quantities.', '',
             '| Fit | Region | Early | 1951-65 | 1966-85 | 1986-2009 | Modern | Biological change | Adult component | Juvenile/reproductive component | Full-process change |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        vals = [*r['era_means'].values(), r['delta'], r['adult_survival'], r['juvenile_survival_times_reproduction'],r['full_process_delta']]
        lines.append(f"| {r['step']} | {r['region']} | " + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    lines += ['', f'Missing collected budgets: {missing}.',
              f'Maximum discrepancy with independent cached biological replay means: {max_cached_error:.6g}.',
              'All era grids, finite values, additive identities and telescoping interval changes were checked. JSON retains cell-change quantiles and each adjacent-era contribution.',
              'Early fields are fitted extrapolations. Restricting their change does not independently validate a historical habitat trajectory.']
    (EXP / 'TEMPORAL_COMPENSATION_AUDIT_20261103.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(records=len(records), missing=missing, max_cached_lambda_error=max_cached_error)))


if __name__ == '__main__':
    main()
