"""Unified P1--P15 review of all later-state and matched nest comparisons."""
import argparse,json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def get(d,*keys):
    for k in keys:
        if d is None:return None
        try:d=d[k]
        except (KeyError,IndexError,TypeError):return None
    return d
def f(x,d=3):return 'pending' if x is None else f'{x:.{d}f}'
def pct(x):return 'pending' if x is None else f'{100*x:.2f}%'
def pair(x):return 'pending' if not x else f"{x['mean']:+.4f} [{x['lo90']:+.4f}, {x['hi90']:+.4f}]"

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();repo=a.repo;exp=repo/'research/experiments'
    audit=json.load(open(exp/'FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json'))
    regional={r['step']:r for r in json.load(open(exp/'REGIONAL_ESTABLISHMENT_LIVE_20261101.json'))}
    nests={r['step']:r for r in json.load(open(exp/'NEST_SITE_FECUNDITY_LIVE_20261101.json'))}
    reference=['step33_nestcbc_s41','step33_nestcbc_abs_s41','step34_abs_natdep_s41']
    rows=[r for r in audit['runs'] if r['step'] in reference or r['step'].startswith(('step35_','step36_'))]
    valid=[r for r in rows if r['valid'] and r.get('contribution_budget')]
    lines=['# Unified initialization and NestWatch results — November 1, 2026','',
      datetime.now(ZoneInfo('America/Chicago')).isoformat(),'',
      'All accepted later-established-state and matched no-NestWatch fits are included, with the earlier relative/absolute/native controls for context. Unfinished rows are listed explicitly. The whole archive now has '+str(audit['counts']['accepted_real'])+' accepted real-data fits; '+str(audit.get('contribution_coverage',0))+' have checked contribution budgets. No configuration is selected on a few favorable summaries.','',
      '## Interpretation so far','',
      'Later initialization is a biologically motivated coarse-grid alternative to simulating the point release with a ten-year Allee exemption. In the completed seed-41 fits it generally reduces the tiny-K/absolute-Allee nonviability mechanism, but the controls return to a juvenile-survival ceiling, and native-departure variants retain abundance-sensitive survival compensation. This is a change in the fitted explanation, not evidence against the initialization premise. Matched seed and later-state nesting ablations are now queued to distinguish basin sensitivity and reproductive-observation interaction.','',
      'The completed original-initialization no-nest controls have similar held-out BBS scores and resident Sj/Sa near .50. Their annual output evaluated at the same 1,089 nest locations is about 3.34–3.38, compared with 3.57–3.58 with NestWatch, although their Fmax is larger. Thus the nest information changes realized reproduction and spatial/density allocation; interpreting it as a simple penalty on the Fmax intercept is inadequate. This ablation remains conditional on the original initialization until the later-state matched controls finish.','',
      'Runge contribution, cell retention, gross cell exchange, and regional dependence on outside immigration are different readouts. Regional mean C near one can hide substantial cell heterogeneity in a slowly changing population; deficit distributions are reported below. The earlier favorable western fundamental-sink share does not replace these checks.','',
      '## Structure and demographic response','',
      'Sa and Sj/Sa are modern model-population-weighted medians. Nest output is realized annual output at identical nest cell-years, not fledglings per attempt. The no-nest readout reproduces production nest means in all checked with-nest fits to tolerance 1e-8.','',
      '| Fit | Init year / Allee exemption | Nests | Sa | Sj/Sa | Population ratio>.95 | Fmax median | Annual output at nest sites |',
      '|---|---|---|---:|---:|---:|---:|---:|']
    for r in valid:
        s=r['step'];pop=r['config']['population_model'];modern=get(r,'pathology','regions','all_land','modern')
        year=get(r,'config','_later_established_state','initialization_year') or 1940
        lines.append(f"| {s} | {year} / {pop.get('release_allee_exemption_years')} yr | {pop.get('nest_likelihood',{}).get('enabled',False)} | {f(get(modern,'Sa','population','q10_50_90',1))} | {f(get(modern,'Sj_over_Sa','population','q10_50_90',1))} | {pct(get(modern,'juvenile_ratio_above_095_population_share'))} | {f(get(modern,'Fmax','population','q10_50_90',1),2)} | {f(get(nests.get(s),'annual_offspring_at_nest_sites_mean'),2)} |")
    lines+=['','## Plains habitat mechanism and historical timing','',
      'The primary Plains band is 100°W–95°W; the live regional JSON also reports 105°W–95°W, 25°N–49°N. Allee-dead here means positive potential growth but no positive local equilibrium under K/Allee; it is not generic inhospitable habitat. Its prevalence alone does not determine biological credibility. Positive records are repeated modern BBS observations, not independent occupied cells.','',
      '| Fit | Plains K land median | Positive Plains records with tiny K | Plains K/Allee-dead cells | Positive Plains records in those cells | Early→modern East lambda | Fitted / frozen-release arrival error yr |',
      '|---|---:|---:|---:|---:|---|---|']
    for r in valid:
        s=r['step'];q=get(regional.get(s),'regions','plains_100W_95W') or {};pp=get(r,'pathology','regions')
        lines.append(f"| {s} | {f(get(pp,'plains','modern','K_relative','land','q10_50_90',1),5)} | {pct(q.get('K_low_positive_fraction'))} | {pct(q.get('dead_land_fraction'))} | {pct(q.get('dead_positive_fraction'))} | {f(get(pp,'east','early','lambda0','land','mean'))}→{f(get(pp,'east','modern','lambda0','land','mean'))} | {f(get(r,'front_speed','fitted','10pct','median_model_minus_observed_yr'),1)} / {f(get(r,'front_speed','frozen_release_only','10pct','median_model_minus_observed_yr'),1)} |")
    lines+=['','Moving initialization eleven years later mechanically changes available expansion time. Smaller frozen-release timing errors therefore cannot alone demonstrate a better dispersal brake. Fitted arrival medians remain about three years early in the completed new fits. Frozen modern environments are diagnostic counterfactuals, not historical reconstructions.','',
      '## Actual contribution, effective survival and immigration dependence','',
      'C/R refer to breeder-to-breeder origin attribution; export counts successful emigrants. Deficit shares use C<.95 (greater than 5% deficit), with .99 and 1−1e-6 sensitivities retained in JSON. Effective adult/juvenile survival uses each age class’s actual origin abundance, including movement and destination mortality, so the means have different spatial weights. Regional isolation ratios cut outside-region immigration from 2010–2025 while retaining normal outward losses.','',
      '| Fit | Region | C | R | Breeder weight in C<.95 cells | Effective adult / juvenile survival | No outside immigration: final/fitted |',
      '|---|---|---:|---:|---:|---|---:|']
    for r in valid:
        for reg in ['west','plains']:
            c=r['contribution_budget'];b=c['regions'][reg]['modern']
            lines.append(f"| {r['step']} | {reg} | {f(b['C_breeding'])} | {f(b['R_breeding'])} | {pct(b['breeder_weighted_sink_share']['0.05'])} | {f(b['effective_adult_survival'])} / {f(b['effective_juvenile_survival'])} | {f(c['isolation'][reg]['regional_no_immigration_final_over_fitted'])} |")
    lines+=['','Gross immigration/export summed over cells includes internal regional transfers. It cannot be called the external regional subsidy. Actual births, local deaths, journey mortality, numerical clipping/edge adjustments and all gross flows are retained in annual_budget.npz; their BIDE closure is checked.','',
      '## Matched prediction','',
      'Fresh scorer results below preserve each run’s own comparison reference, common-design SHA and production prediction parity. Intervals are exploratory 90% spatial-block bootstrap intervals conditional on fitted points, not demographic uncertainty intervals. Current new-run spatial contrasts use 30 supported real-data blocks (31 pinned blocks include the broader holdout design); temporal contrasts use 262 blocks. Do not combine old reported bootstrap summaries with these as if every earlier implementation detail were identical.','',
      '| Fit | Reference | Spatial real marginal gain | Temporal marginal gain |','|---|---|---|---|']
    for r in valid:
        if r['step'] in reference:continue
        p=repo/f".research_runtime/range_monitor/rescore_{r['step']}.json"
        score=json.load(open(p)) if p.exists() else {};q=score.get('paired_vs_ref',{});diff=q.get('diff',{})
        lines.append(f"| {r['step']} | {q.get('ref','pending')} | {pair(diff.get('S_real_marg'))} | {pair(diff.get('T_marg'))} |")
    lines+=['','## Complete P1–P15 assessment','',
      '| Concern | Evidence and current status |','|---|---|',
      '| P1 historical growth | All early/modern regional trajectories retained. Later-state controls infer lower early eastern growth and higher modern growth than the earlier absolute reference; independent habitat-history validation remains missing. |',
      '| P2 invasion timing | All new completed fitted medians remain ~3 years early. Frozen-release medians improve partly through the later calendar start and still arrive decades early. No timing clearance. |',
      '| P3 reproduction | Compare Fmax and realized annual nest-site output separately. NestWatch ablation changes them in different directions. Annual conversion remains pinned in these fits and needs uncertainty propagation. |',
      '| P4 juvenile ceiling | Later-state controls: ~66–69% of fitted modern population above Sj/Sa=.95. Native later-state low abundance also reaches ~60%; other brackets avoid the ceiling but have different survival balances. Effective movement-weighted survival is now measured. |',
      '| P5 compensation | Explicit initialization/abundance/nesting contrasts show changes in F/Sa/Sj/K with comparatively small count-score shifts. New matched cold seed-42 and later-state no-nest probes are queued; no unique explanation claimed. |',
      '| P6 source/sink and subsidy | Runge breeding and stage-weighted C, retention/export, BIDE and nonlinear isolation now available. Report distributions and scale; regional C near one is not an absence of local deficits. |',
      '| P7 tiny K | Regional land medians and positive-record context included above. Later-state controls reduce this symptom, with survival compensation. Spatial maps remain available; low K alone is not a pathology verdict. |',
      '| P8 K/Allee barrier artifact | Positive-growth/nonviable cells and observations distinguished from fundamentally unfavorable habitat. Finite deterministic propagule tests are being extended to the new fits; stochastic establishment and within-cell pairing remain unvalidated. |',
      '| P9 founders/init | Later-state initialization and zero exemption actually fitted; backcast constraint removed. Abundances remain conditional fixed scenarios and the inherited 50/50 age split/within-cell support are assumptions, not CBC likelihood estimates. |',
      '| P10 movement distances/tails | Actual survival-weighted off-cell endpoint moments implemented across the stored batch and queued for every newly accepted fit. Pending new cases remain pending; no substitution of nominal kernel bands. |',
      '| P11 exodus/redistribution | Retention/export and actual fluxes are measured. Full departure-curve and before/after geographic redistribution interpretation still incomplete; native retention benefits cannot be equated with better overall contribution. |',
      '| P12 mover survival/Q | Effective age-specific survival and journey mortality now accounted. Numerical/off-domain loss and clip gains kept separate. Independent movement/survival calibration remains unfinished. |',
      '| P13 projection flexibility | These cases retain the shared field; source/config/parameter norms are inventoried. Detailed geometry/extrapolation comparison remains incomplete. |',
      '| P14 scale inflation | Parameter norms and field outputs retained; no claim that coefficient/amplitude compensation is resolved. |',
      '| P15 flexible corrections | No new ecological field or process correction introduced by these probes. Existing structural switches are recorded; remaining habitat-history flexibility still needs independent evaluation. |','',
      'HMC geometry and sampler efficiency are a separate computational criterion. Four full geometry audits remain queued; a timing-calibrated, warmup-checkpointed NUTS comparison follows. No HMC-readiness claim is made from the smoke test.','',
      '## Unfinished comparisons','']
    for r in rows:
        if r not in valid:lines.append(f"- {r['step']}: {r.get('queue_state','not yet in accepted archive')}; no completed result substituted.")
    lines+=['','## Functional stability versus component stability','',
      'See [the direct functional comparison](FUNCTIONAL_STABILITY_20261101.md) for common-support contribution/local-replacement/flux/export comparisons across abundance brackets, initialization formulations, seeds and structural changes. Component compensation does not by itself establish non-identifiability of aggregate growth or movement. In the completed 1951 bracket, realized contribution maps are considerably more stable than survival allocation; changing native departure changes western contribution geography substantially despite similar regional averages. Cross-fit stability is evidence of robustness, not a posterior identification certificate.','',
      'Live machine-readable source: FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json, REGIONAL_ESTABLISHMENT_LIVE_20261101.json, NEST_SITE_FECUNDITY_LIVE_20261101.json, matched rescore receipts and INITIALIZATION_PROBE_RUNS_20261101.json.']
    out=exp/'LATER_INITIALIZATION_UNIFIED_REVIEW_20261101.md';tmp=out.with_suffix('.tmp.md');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(out)
    print('UNIFIED INITIALIZATION REVIEW',len(valid),'completed contextual configurations')


if __name__=='__main__':main()
