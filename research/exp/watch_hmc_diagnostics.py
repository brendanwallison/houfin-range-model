"""Persistent HMC audit collection; no model changes or sampler adoption."""
import json,subprocess,time
from pathlib import Path
REPO=Path('/mnt/c/Dev/houfin-range-model');STATE=REPO/'.research_runtime/hmc_diagnostic_status.json'
HOST='bwa386@ls6.tacc.utexas.edu';SOCKET=str(Path.home()/'.ssh/cm-tacc')
OUT=Path('/mnt/e/Datasets/houfin/hmc_diagnostics_20261101')
last_audit_state=None
while not (REPO/'.research_runtime/STOP_HMC_DIAGNOSTICS_WATCH').exists():
 try:
  raw=subprocess.check_output(['ssh','-S',SOCKET,'-o','BatchMode=yes',HOST,
      '/work/07980/bwa386/ls6/houfin/venv/bin/python /work/07980/bwa386/ls6/houfin/research/grid54/monitor/release_hmc_screens.py && /work/07980/bwa386/ls6/houfin/venv/bin/python /work/07980/bwa386/ls6/houfin/research/grid54/monitor/hmc_diagnostic_status.py'],timeout=45)
  d=json.loads(raw);temp=STATE.with_suffix('.tmp.json');temp.write_bytes(raw);temp.replace(STATE)
  lines=['# HMC geometry and performance diagnostics — live results','',
   'Full unconstrained nominal posterior, including transformation Jacobians and observer nuisance parameters. Geometry rows are matrix-free Krylov estimates with residuals, not certified full condition numbers or full negative-eigenvalue counts. A finite geometry result is not an HMC pass. All queued cases are shown.','',
   '| Fit | Diagnostic | Queue | Chain 0 iterations | Complete |','|---|---|---|---|---|']
  for key,b in d['runs'].items():
   progress=b.get('chain0_progress',{})
   count=f"{progress['completed_iterations']}/{progress['planned_iterations']}" if progress else 'not started'
   lines.append(f"| {b['step']} | {b['mode']} | {b['queue_state']} | {count} | {bool(b.get('report'))} |")
   dest=OUT/b['step']/b['mode']
   if b.get('report') and not (dest/'READY.json').exists():
    dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp','-r','-o','ControlPath='+SOCKET,f"{HOST}:{b['artifact_dir']}/.",str(dest)+'/'],check=True,timeout=180)
  lines+=['','One diagonal run provides timing calibration first. Other sampler comparisons are held for measured compute-cost support; held_cost is a scheduling decision, not a model-quality failure. Warmup and sampling checkpoint every four transitions with adaptation and RNG preserved. Six hours bounds each attempt, not promised completion.','',
    'Geometry measures raw and conditional-prior-standardized curvature, numerical derivative/roughness checks and nearby curvature variation. The sampler screens compare diagonal and block-dense ecological metrics with diagonal observer nuisance blocks. Block screens have longer warmup; wall-time efficiency includes initialization, warmup and compilation, while leapfrog efficiency counts retained-draw steps only.','',
    'NUTS screens report divergences, acceptance, adapted step sizes, depth saturation, E-BFMI, NumPyro ESS/split-Rhat, and ESS per leapfrog and total wall time. These bounded near-MAP pilots are not full inference: eventual adoption requires longer dispersed chains, rank-normalized bulk/tail diagnostics, Monte Carlo error for biological estimands, and the biological pathology checklist.','']
  lines+=['','## Compute-cost forecasts','',
    'Depth-saturated scenarios based on each GPU gradient timing and, once available, observed NUTS leapfrog timing. Compilation/setup overhead is additional.','',
    '| Fit | Metric | Saturated hours | Attempt budget hours | Timing calibration |','|---|---|---:|---:|---|']
  for f in d.get('cost_forecasts',[]):
   value='unmeasured' if f['saturated_trajectory_hours'] is None else f"{f['saturated_trajectory_hours']:.2f}"
   lines.append(f"| {f['step']} | {f['mode']} | {value} | {f['attempt_budget_hours']} | {f['timing_calibration']} |")
  for b in d['runs'].values():
   r=b.get('report')
   if not r:continue
   lines.append('## '+b['mode']+' / '+b['step'])
   if b['mode']=='geometry':
    lines+=['',f"Dimension {r['dimension']}; gradient norm {r['gradient_norm']:.4g}; gradient median {r['grad_seconds_median']:.4g} seconds.",'',
       '| Coordinates | Smallest Ritz estimate | Largest Ritz estimate |','|---|---:|---:|']
    for name in ['raw_unconstrained','conditional_prior_scaled']:
     q=r[name];lines.append(f"| {name} | {q['ritz_min']:.4g} | {q['ritz_max']:.4g} |")
   else:
    lines+=['',json.dumps(r['chains'],indent=2),'']
  path=REPO/'research/experiments/HMC_DIAGNOSTIC_RESULTS_20261101.md';tmp=path.with_suffix('.tmp.md');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(path)
  key=tuple((k,b['queue_state'],bool(b.get('report'))) for k,b in d['runs'].items())
  if key!=last_audit_state:
   live=REPO/'research/experiments/FULL_PATHOLOGY_AUDIT_LIVE_20261101.json'
   subprocess.run(['/home/breallis/houfin/venv/bin/python',str(REPO/'research/exp/merge_contribution_audit.py'),
     '--root','/mnt/e/Datasets/houfin/contribution_20261101_v3',
     '--audit',str(live if live.exists() else REPO/'research/experiments/FULL_PATHOLOGY_AUDIT_20261101_AFTERNOON.json'),
     '--out',str(REPO/'research/experiments/FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json')],check=True,timeout=45)
   last_audit_state=key
  if all(b['queue_state'] in ['done','failed'] for b in d['runs'].values()):break
 except Exception as e:
  (REPO/'.research_runtime/hmc_diagnostic_watch_error.json').write_text(json.dumps({'epoch':time.time(),'error':str(e)}))
 time.sleep(30)
