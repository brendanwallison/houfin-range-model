"""Render all completed contribution audits; incomplete runs stay explicit."""
import argparse,json,os
from pathlib import Path


def render(root,out):
    records=[]
    for p in sorted(root.glob('*/READY.json')):
        try:r=json.loads(p.read_text())
        except json.JSONDecodeError:continue
        context=p.parent/'CONTEXT.json'
        if context.exists():
            for reg,v in json.loads(context.read_text()).items():r['regions'][reg]['modern'].update(v)
        records.append(r)
    f=lambda x:'missing' if x is None else f'{x:.3f}'
    lines=['# Contribution and population-budget results — November 1, 2026','',
      'Computed from each accepted fit with its own frozen source and float64 CPU replay. No demographic refitting. All completed runs are shown; this is not a selected-candidate table. Annual cell arrays and nonlinear regional/cell isolation replays are retained beside READY.json.','',
      'Runge C_breeding follows prior summer breeders and their actual offspring to the next summer breeding census, wherever they settle. R is the same-origin retained component (including same-cell returns), and export = C − R. Incoming birds are not credited to the receiving origin. C_state separately follows the model postbirth adult/juvenile census, weighting stage-specific contributions by actual stage proportions. The two censuses are not interchangeable. Sources: [Runge et al. 2006](https://doi.org/10.1086/503531), especially the multiple-stage section; [Diffendorfer et al. 2019](https://doi.org/10.1016/j.ecolmodel.2019.06.003) for density-dependent structured extensions.','',
      'Attribution holds departure and density-dependent demographic coefficients at the full fitted environment. The transpose acts only on tagged transport, not on density feedback. Independent matrix-enumeration tests verify origin/destination accounting and exporting-source classification. Every accepted audit also checks exact trajectory replay, BIDE closure and summed origin contributions against destination totals. Kernels retain their stored precision, explicitly recorded in receipts; classification tolerance sensitivities are provided. Initial seeded and exogenous-release cell transitions are excluded from C summaries. Era labels refer to the next breeding census year.','',
      '## Modern (2010–2025) contribution, retention and effective survival','',
      '| Fit | Region | C breeding | R breeding | Successful cell export/breeder | Cell immigrant share of next breeders | Breeder weight in cells with C<.99 | C state | Effective adult / juvenile survival |',
      '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in records:
        for region in ['west','plains','broad_plains','east']:
            b=r['regions'][region]['modern']
            lines.append(f"| {r['step']} | {region} | {f(b['C_breeding'])} | {f(b['R_breeding'])} | {f(b['successful_export_per_breeder'])} | {f(b['immigrant_share_of_next_breeders'])} | {f(b['breeder_weighted_sink_share']['0.01'])} | {f(b['C_state'])} | {f(b.get('effective_adult_survival'))} / {f(b.get('effective_juvenile_survival'))} |")
    lines+=['','C near one is expected in a slowly changing regional population; report the distribution and flows as well as the regional ratio. Effective survival uses each age class’s actual fitted origin abundance, so the two means have different spatial weights. Immigration fractions are gross, not net subsidies.','',
      '## Actual BIDE accounting and nonlinear isolation','',
      'Budget sums are model density units summed over years, not literal bird counts. Immigration/export refer to movement across CELL boundaries; summing them over a region includes within-region exchanges and does not equal external regional subsidy. Births/deaths are postbirth census events. Successful transport excludes journey mortality, which is attributed to its origin and kept separate from edge/numerical loss. Net immigration need not classify sources away from equilibrium. Regional isolation starts from the fitted end-2009 state, cuts incoming flow and retains ordinary emigration losses through 2025. Cell closure instead removes both immigration and emigration. Neither is the origin-attribution calculation.','',
      '| Fit | Region | Births | Ordinary deaths | Journey deaths | Gross immigration | Gross successful emigration | Isolation final/fitted | Cell closure final/fitted |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        for region in ['west','plains','broad_plains','east']:
            b=r['regions'][region]['modern'];i=r['isolation'][region]
            lines.append(f"| {r['step']} | {region} | {f(b['births'])} | {f(b['ordinary_deaths'])} | {f(b['journey_deaths'])} | {f(b['gross_immigration'])} | {f(b['gross_successful_emigration'])} | {f(i['regional_no_immigration_final_over_fitted'])} | {f(i['cell_no_movement_final_over_fitted'])} |")
    lines+=['','## Mechanism context and validation','',
      '| Fit | Region | Potential-sink land share | Positive-potential K/Allee-dead land share | C breeding in K/Allee-dead cells |',
      '|---|---|---:|---:|---:|']
    for r in records:
        for reg in ['west','plains','broad_plains','east']:
            b=r['regions'][reg]['modern']
            lines.append(f"| {r['step']} | {reg} | {f(b.get('potential_sink_land_share'))} | {f(b.get('K_Allee_dead_land_share'))} | {f(b.get('K_Allee_dead_C_breeding'))} |")
    lines+=['','These cached viability labels describe mean-environment local equilibria, not stochastic establishment probabilities or actual demographic contribution. Negative budget components above numerical tolerance are rejected; small signed roundoff remains visible in NPZ rather than silently converted into biological events.','']
    for r in records:lines.append(f"- {r['step']}: maximum adult/juvenile replay error {max(r['checks']['replay_adult_error'],r['checks']['replay_juvenile_error']):.3g}; BIDE residual {r['checks']['BIDE_error']:.3g}; contribution-total residual {r['checks']['breeder_origin_destination_sum_error']:.3g}. Receipt includes source/checkpoint/implementation hashes.")
    failures=root/'batch_failures.jsonl'
    if failures.exists():lines+=['','## Unfinished or rejected audits','',*['- '+line for line in failures.read_text().splitlines()]]
    lines+=['','## Remaining agreed work','',
      'S1 potential growth, S2 actual-state local replacement, S3 BIDE flows/quadrants, S4 breeder and stage-weighted contributions with retention/export, S6 apparent growth, effective movement-weighted age survival and nonlinear isolation are implemented here. S5 long-run reproductive value was optional/conditional and has not been implemented. Six later-established-state initialization fits have also been launched on TACC; their results are pending (LATER_ESTABLISHED_STATE_STATUS_20261101.md). Realized permanent-distance/tail diagnostics and finite local propagule checks are implemented and running across the archive (MOVEMENT_ESTABLISHMENT_RESULTS_20261101.md). Demographic-stochasticity checks, full joint observation fitting and P1–P15 configuration selection remain unfinished. These are not passes. All eras, cell distributions, flow quadrants and tolerance sensitivities are retained in machine-readable outputs.', '',f'Artifact root: `{root}`. Numerical clipping gains and unparented numerical births are explicitly retained as numerical adjustments; they are not credited as biological immigrants or breeder-origin production.']
    tmp=out.with_name(out.name+f'.{os.getpid()}.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(out)
    print('REPORT',len(records),'completed audits',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();render(a.root,a.out)
