"""Evidence inventory for outstanding fits; file presence is not scientific acceptance."""
import json,time
from pathlib import Path
R=Path('/mnt/c/Dev/houfin-range-model');E=R/'research/experiments';S=R/'.research_runtime'
def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def main():
 state=read(S/'later_state_status.json',{});runs=state.get('runs',{})
 inv={x['step']:x for x in read(E/'FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json',{'runs':[]})['runs']}
 hmc=read(S/'hmc_diagnostic_status.json',{}).get('runs',{})
 local=read(S/'local_overnight/status.json',{})
 records=[]
 for registry in ['MINOR_FLEXIBILITY_RUNS_20261103.json','OUTSTANDING_TRANSFER_RUNS_20261103.json','KFLOOR_COMBINATION_RUNS_20261102.json']:
  for spec in read(E/registry,[]):
   step=spec['step'];b=runs.get(step,{});v=inv.get(step,{})
   records.append(dict(step=step,registry=registry,reference=spec['reference'],seed=spec['seed'],family=spec['family'],queue_state=b.get('queue_state','not_in_live_snapshot'),
     accepted_refinement=bool(b.get('refine_report',{}).get('valid')),float32_handoff=step in local.get('completed',[]),
     contribution_budget=bool(b.get('contribution_done')),real_record_rescore=(S/f'range_monitor/rescore_{step}.done.json').exists(),
     movement_establishment=(Path('/mnt/e/Datasets/houfin/movement_followup_20261101')/step/'READY.json').exists(),
     front=bool(v.get('front_speed')),pathology=bool(v.get('pathology')),hmc=[x['mode'] for x in hmc.values() if x['step']==step and x.get('report')]))
 d=dict(epoch=time.time(),live_snapshot_epoch=state.get('epoch'),records=records,local_active=local.get('active'),local_handoffs=len(local.get('completed',[])),
   allocations=read(S/'overflow_resource_status.json',{}).get('jobs',[]),
   caveat='Presence/receipt inventory, not certification against every pathology or HMC suitability. Pending refinement is not a completed scientific comparison.')
 p=E/'OUTSTANDING_WORK_STATUS_20261103.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
 lines=['# Outstanding work: live execution and evidence','',d['caveat'],'',f"Snapshot epoch {d['live_snapshot_epoch']}; report epoch {d['epoch']}. Local verified MAP handoffs: {d['local_handoffs']}.",'',
 '| Fit | Queue | Refined | Local MAP handed off | Budget | Real-record score | Movement/establishment | Front | HMC report |',
 '|---|---|---|---|---|---|---|---|---|']
 for x in records:
  cells=[x['step'],x['queue_state']]+['yes' if x[k] else 'missing' for k in ['accepted_refinement','float32_handoff','contribution_budget','real_record_rescore','movement_establishment','front']]+[','.join(x['hmc']) or 'missing']
  lines.append('| '+' | '.join(cells)+' |')
 lines+=['','The complete scope, conditional experiments and observation-model work are in [OUTSTANDING_WORK_20261103.md](OUTSTANDING_WORK_20261103.md). Held obsolete designs are excluded from automatic execution. Local float32 completion never substitutes for TACC refinement.']
 p=E/'OUTSTANDING_WORK_STATUS_20261103.md';tmp=p.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(p)
 print('OUTSTANDING_INVENTORY',len(records),flush=True)
if __name__=='__main__':main()