"""Publish validated batch, prioritize long HMC on small, request 3 development GPUs."""
import hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)

def main():
    validated=json.loads((R/'monitor/overflow_early_VALIDATED.json').read_text())
    if not all(s['finite'] for s in validated['full_model_checks']):raise ValueError('invalid full-model checks')
    actual=hashlib.sha256((R/'inputs/cbc_early_20261102/cbc_early.npz').read_bytes()).hexdigest()
    if actual!=validated['cbc_input_sha256']:raise ValueError('CBC input changed after validation')
    # Keep current running fits intact; unstarted redundant all-years CBC variants are reserves.
    held=R/'queue/held_new_priority';held.mkdir(exist_ok=True)
    moved=[]
    for p in (R/'queue/pending').glob('*_step38_*.sh'):
        if 'circle2_free_s41' in p.name:continue
        p.rename(held/p.name);moved.append(p.name)
    # Only the already authorized diagonal HMC timing screen is promoted.
    path=R/'monitor/hmc_diagnostic_specs.json';specs=json.loads(path.read_text())
    for s in specs:
        if s['mode']=='nuts' and s['step']=='step33_nestcbc_abs_s41':
            old=R/'queue/pending/0640_hmc_nuts_step33_nestcbc_abs_s41.sh'
            if old.exists():old.rename(old.with_name('0604_hmc_nuts_step33_nestcbc_abs_s41.sh'))
            s['queue_priority']='0604'
    write(path,specs)
    subprocess.run([str(R.parents[1]/'venv/bin/python'),str(R/'monitor/stage_overflow_early.py'),'--publish'],check=True)
    env=dict(os.environ)
    import re
    reply=subprocess.check_output(['sbatch','--parsable',str(R/'monitor/overflow_dev.slurm')],cwd=R,env=env,text=True)
    result=re.findall(r'^([0-9]+)(?:;[^\n]+)?$',reply,re.MULTILINE)[-1]
    write(R/'monitor/overflow_early_DISPATCH.json',dict(epoch=time.time(),dev_job=result,
        existing_small_job='3544518',development_gpu_lanes=3,development_walltime_hours=2,
        small_HMC_priority='0604',held_redundant_CBC=moved))
    print('DEVELOPMENT JOB',result,flush=True)

if __name__=='__main__':main()
