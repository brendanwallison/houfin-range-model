"""One bounded retry for nonterminal, still-improving block-correction fits."""
import json,re,time
from pathlib import Path
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
resumed=[]
for s in json.loads((R/'monitor/misspecification_20261104_specs.json').read_text()):
 if not s['step'].startswith('step48_ab_blocks15_weak_'):continue
 run=R/'runs'/s['step'];task=s['queue_priority']+'_'+s['step']+'.sh';failed=R/'queue/failed'/task
 if not failed.exists() or (R/'queue/running'/task).exists():continue
 ref=R.parents[1]/f"processed/model_results/refined_float64__age_map_float32_grid54_{s['step']}_quick90/refine_report.json"
 if not ref.exists():continue
 report=json.loads(ref.read_text());script=run/'grid54_fit.sh';text=script.read_text()
 if report.get('valid') or 'max_iter 16000 reached' not in report.get('reason','') or 'HOUFIN_REFINE_MAX_ITER=16000' not in text:continue
 drops=re.findall(r'drop over last 20=([-+\d.eE]+)',(run/'out/refine.log').read_text(errors='replace'))
 if not drops or float(drops[-1])<=.05:continue
 backup=run/'grid54_fit_before_automatic_cap_extension.sh'
 if backup.exists():continue
 backup.write_text(text)
 script.write_text(text.replace('HOUFIN_REFINE_MAX_ITER=16000','HOUFIN_REFINE_MAX_ITER=32000'))
 failed.rename(R/'queue/pending'/task)
 receipt=dict(epoch=time.time(),step=s['step'],old_max_iter=16000,new_max_iter=32000,last_20_iteration_drop=float(drops[-1]),
  action='One nonterminal checkpoint continuation; model and MAP unchanged. No retry of stalled or nonfinite fits.')
 (run/'AUTOMATIC_CAP_EXTENSION.json').write_text(json.dumps(receipt,indent=2)+'\n');resumed.append(receipt)
print(json.dumps(resumed))
