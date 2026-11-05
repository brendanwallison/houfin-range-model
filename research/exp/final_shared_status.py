"""Read-only round1 status; runs on either local machine or TACC."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--processed', type=Path, required=True)
p.add_argument('--jobs', nargs='*', default=[])
a=p.parse_args()
result={'timestamp':datetime.now(timezone.utc).isoformat(), 'cases':{}}
for arm in ['w64','s64','r128']:
    out=a.root/arm
    entry={'markers':[name for name in ['MAP_DONE','MAP_DIAGNOSTICS_DONE','FIT_DONE'] if (out/name).exists()]}
    for stage in ['map','refine','viz_map','viz_ref']:
        f=out/(stage+'.log')
        if f.exists():entry[stage+'_tail']=f.read_text(errors='replace').splitlines()[-3:]
    f=a.processed/'model_results'/f'refined_float64__age_map_float32_shared_clean_{arm}_quick90/refine_report.json'
    if f.exists():entry['refine_report']=json.loads(f.read_text())
    for stage in ['map','ref']:
        for name in ['metrics.json','west_sink_audit.json']:
            f=out/('viz_'+stage)/name
            if f.exists():entry[stage+'/'+name]=json.loads(f.read_text())
    result['cases'][arm]=entry
if a.jobs:
    job_ids=','.join(a.jobs)
    cmd=['sacct','-j',job_ids,'-n','-X','-P','-o','JobID,State,Elapsed,Timelimit']
    r=subprocess.run(cmd,text=True,capture_output=True,timeout=20)
    result['jobs']=r.stdout.strip().splitlines()
    if r.returncode:result['scheduler_error']=r.stderr.strip()
print(json.dumps(result,allow_nan=False))
