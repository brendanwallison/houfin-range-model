"""Read-only collection of every queued HMC diagnostic, including partial work."""
import json,time
from pathlib import Path
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');specs=json.load(open(R/'monitor/hmc_diagnostic_specs.json'))
out={'epoch':time.time(),'runs':{}}
for spec in specs:
 mode,step=spec['mode'],spec['step'];prefix=spec.get('queue_priority','0630' if mode=='geometry' else '0640')
 directory='hmc_geometry' if mode=='geometry' else 'hmc_screen_block' if mode=='nuts_block' else 'hmc_screen'
 d=R/'runs'/step/'out'/directory
 state=next((s for s in ['running','pending','held_geometry','held_cost','done','failed'] if (R/'queue'/s/f'{prefix}_hmc_{mode}_{step}.sh').exists()),'unknown')
 entry={'mode':mode,'step':step,'queue_state':state,'artifact_dir':str(d),'spec':spec}
 for filename,key in [('READY.json','report'),('PARTIAL.json','partial'),('RUN_RECEIPT.json','receipt')]:
  p=d/filename
  if p.exists():
   try:entry[key]=json.loads(p.read_text())
   except json.JSONDecodeError:pass
 out['runs'][mode+'/'+step]=entry
 for chain in [0,1]:
  progress=d/f'chain{chain}/PROGRESS.json'
  if progress.exists():entry[f'chain{chain}_progress']=json.loads(progress.read_text())
forecast=R/'monitor/hmc_cost_forecasts.json'
if forecast.exists():out['cost_forecasts']=json.loads(forecast.read_text())
print(json.dumps(out))
