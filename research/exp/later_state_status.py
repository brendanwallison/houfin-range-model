"""Small read-only status snapshot for the later-initialization experiments."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
specs=json.loads((a.root/'monitor/later_state_specs.json').read_text());out={'epoch':time.time(),'runs':{}}
extra=a.root/'monitor/nest_ablation_specs.json'
if extra.exists():specs+=json.loads(extra.read_text())
extra=a.root/'monitor/overflow_early_specs.json'
if extra.exists():specs+=json.loads(extra.read_text())
extra=a.root/'monitor/cbc_established_specs.json'
if extra.exists():specs+=json.loads(extra.read_text())
extra=a.root/'monitor/historical_range_specs.json'
if extra.exists():specs+=json.loads(extra.read_text())
extra=a.root/'monitor/gradual_departure_specs.json'
if extra.exists():specs+=json.loads(extra.read_text())
extra=a.root/'monitor/crowding_curvature_specs.json'
if extra.exists():specs+=json.loads(extra.read_text())
extra=a.root/'monitor/floor_combination_specs.json'
if extra.exists():specs+=json.loads(extra.read_text())
extra=a.root/'monitor/minor_flexibility_specs.json'
if extra.exists() and (a.root/'monitor/minor_flexibility_VALIDATED.json').exists():specs+=json.loads(extra.read_text())
extra=a.root/'monitor/outstanding_transfer_specs.json'
if extra.exists() and (a.root/'monitor/outstanding_transfer_VALIDATED.json').exists():specs+=json.loads(extra.read_text())
extra=a.root/'monitor/misspecification_20261104_specs.json'
if extra.exists() and (a.root/'monitor/misspecification_20261104_VALIDATED.json').exists():specs+=json.loads(extra.read_text())
for spec in specs:
 s=spec['step'];run=a.root/'runs'/s
 prefix=spec.get('queue_priority','0620' if s.startswith('step36_') else '0610')
 state=next((x for x in ['running','local_running','pending','held_early_design','held_new_priority','done','failed'] if (a.root/'queue'/x/f'{prefix}_{s}.sh').exists()),'unknown')
 ref=a.root.parents[1]/f'processed/model_results/refined_float64__age_map_float32_grid54_{s}_quick90/refine_report.json'
 b={'queue_state':state,'fit_done':(run/'out/FIT_DONE').exists(),'contribution_done':(run/'out/contribution/READY.json').exists()}
 local=a.root/'queue/local_running'/f'{prefix}_{s}.sh.local3090.json'
 if local.exists():b['local_map_reservation']=json.loads(local.read_text())
 if ref.exists():b['refine_report']=json.loads(ref.read_text())
 cbc=run/'out/cbc_score.json'
 if cbc.exists():b['cbc_score']=json.loads(cbc.read_text())
 for name in ['map','refine','contribution']:
  log=run/f'out/{name}.log'
  if log.exists():b[name+'_tail']=log.read_text(errors='replace')[-1000:]
 out['runs'][s]=b
print(json.dumps(out))
