"""Persistent TACC batch/resource watcher with a bounded development-wave limit."""
import fcntl,json,os,subprocess,time
from pathlib import Path
REPO=Path('/mnt/c/Dev/houfin-range-model');R='/work/07980/bwa386/ls6/houfin/research/grid54'
SSH=['ssh','-S',str(Path.home()/'.ssh/cm-tacc'),'-o','BatchMode=yes','bwa386@ls6.tacc.utexas.edu']
PY='/work/07980/bwa386/ls6/houfin/venv/bin/python'
lock=(REPO/'.research_runtime/overflow_resource_watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
REMOTE_SCRIPT='''import json,re,subprocess,time
from pathlib import Path
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
path=R/'monitor/overflow_early_DISPATCH.json';d=json.loads(path.read_text())
extra=R/'monitor/floor_combination_DISPATCH.json'
if extra.exists():
    for j in json.loads(extra.read_text()).get('regular_gpu_jobs',[]):
        if j not in d.setdefault('regular_gpu_jobs',[]):d['regular_gpu_jobs'].append(j)
d['dev_job']=str(d['dev_job']).splitlines()[-1]
jobs=d.setdefault('development_jobs',[d['dev_job']])
active=subprocess.check_output(['squeue','-h','-u','bwa386','-o','%i|%P|%T|%L|%N'],text=True).splitlines()
live={row.split('|')[0]:row.split('|') for row in active}
local_helper=R/'monitor/local_map_dispatch.py'
recovered_local=[]
if local_helper.exists():
    recovered_local=json.loads(subprocess.check_output(['/work/07980/bwa386/ls6/houfin/venv/bin/python',str(local_helper),'recover'],text=True,timeout=60))
pending=[p for prefix in range(39,49) for p in (R/'queue/pending').glob(f'*_step{prefix}_*.sh')]
recovered=[]
for owner in [p for prefix in range(39,49) for p in (R/'queue/running').glob(f'*_step{prefix}_*.owner')]:
    if owner.read_text().strip() not in live:
        task=owner.with_suffix('.sh')
        try:task.rename(R/'queue/pending'/task.name);recovered.append(task.name)
        except FileNotFoundError:pass
        owner.unlink(missing_ok=True)
# A pending regular request must not prevent a replacement development wave.
# Preserve the recorded wave budget; extensions are explicit dispatch decisions.
if not any(j in live for j in jobs) and len(pending)>=3 and len(jobs)<int(d.get('development_wave_limit',3)):
    reply=subprocess.check_output(['sbatch','--parsable',str(R/'monitor/overflow_dev.slurm')],cwd=R,text=True)
    jobs.append(re.findall(r'^([0-9]+)(?:;[^\\n]+)?$',reply,re.MULTILINE)[-1])
tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(d,indent=2)+'\\n');tmp.replace(path)
gpu={}
for node in sorted({r[4] for r in live.values() if r[2]=='RUNNING' and '[' not in r[4]}):
    try:
        out=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',node,'nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],timeout=15,text=True,stderr=subprocess.DEVNULL)
        gpu[node]=out.strip().splitlines()
    except Exception as e:gpu[node]={'error':str(e)}
states={}
specs=json.loads((R/'monitor/overflow_early_specs.json').read_text())
if (R/'monitor/historical_range_specs.json').exists():specs+=json.loads((R/'monitor/historical_range_specs.json').read_text())
if (R/'monitor/gradual_departure_specs.json').exists():specs+=json.loads((R/'monitor/gradual_departure_specs.json').read_text())
if (R/'monitor/crowding_curvature_specs.json').exists():specs+=json.loads((R/'monitor/crowding_curvature_specs.json').read_text())
if (R/'monitor/floor_combination_specs.json').exists():specs+=json.loads((R/'monitor/floor_combination_specs.json').read_text())
if (R/'monitor/minor_flexibility_specs.json').exists() and (R/'monitor/minor_flexibility_VALIDATED.json').exists():specs+=json.loads((R/'monitor/minor_flexibility_specs.json').read_text())
if (R/'monitor/outstanding_transfer_specs.json').exists() and (R/'monitor/outstanding_transfer_VALIDATED.json').exists():specs+=json.loads((R/'monitor/outstanding_transfer_specs.json').read_text())
if (R/'monitor/misspecification_20261104_specs.json').exists() and (R/'monitor/misspecification_20261104_VALIDATED.json').exists():specs+=json.loads((R/'monitor/misspecification_20261104_specs.json').read_text())
for s in specs:
    name=s['queue_priority']+'_'+s['step']+'.sh'
    state=next((x for x in ['pending','running','local_running','held_early_design','done','failed'] if (R/'queue'/x/name).exists()),'missing')
    rec={'state':state,'fit_done':(R/'runs'/s['step']/'out/FIT_DONE').exists()}
    if state=='failed':
        p=R/'queue/logs'/name.replace('.sh','.log')
        rec['error_tail']=p.read_text(errors='replace')[-1800:] if p.exists() else 'no log'
    states[s['step']]=rec
print(json.dumps(dict(epoch=time.time(),dispatch=d,jobs=list(live.values()),gpu=gpu,recovered=recovered,recovered_local=recovered_local,runs=states)))
'''
while not (REPO/'.research_runtime/STOP_OVERFLOW_WATCH').exists():
    try:
        raw=subprocess.check_output(SSH+[PY,'-'],input=REMOTE_SCRIPT.encode(),timeout=90)
        d=json.loads(raw);p=REPO/'.research_runtime/overflow_resource_status.json'
        tmp=p.with_suffix('.tmp.json');tmp.write_bytes(raw);tmp.replace(p)
        out=REPO/'research/experiments/OVERFLOW_EARLY_LIVE_20261102.md'
        lines=['# Overflow, K-floor and historical CBC: live execution','',
          'TACC jobs: '+', '.join(' / '.join(j) for j in d['jobs']),'',
          '| Fit | Queue | Optimizer complete |','|---|---|---|']
        lines += [f"| {s} | {b['state']} | {b['fit_done']} |" for s,b in d['runs'].items()]
        lines+=['','GPU observations (index, model, used/total MiB, utilization %):','',json.dumps(d['gpu'],indent=2)]
        out.write_text('\n'.join(lines)+'\n')
        if all(x['state'] in ['done','failed','held_early_design'] for x in d['runs'].values()):break
    except Exception as e:
        (REPO/'.research_runtime/overflow_resource_error.json').write_text(json.dumps(dict(epoch=time.time(),error=str(e))))
    time.sleep(45)
