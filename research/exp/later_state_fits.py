"""Stage/execute the requested later established-cell initialization comparison.

Reuses frozen current model, identical observations/features and existing workers.
No release-cell Allee exemption, founder prior learning or CBC backcast likelihood.
"""
import argparse,fcntl,hashlib,json,math,os,pickle,shutil,subprocess,time
from pathlib import Path
import numpy as np


def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def stage(root):
    base=root/'runs/step33_nestcbc_abs_s41'
    cfg=json.loads((base/'code/config/overlays/grid54_step33_nestcbc_abs_s41.json').read_text())
    source=Path(cfg['input_dir'].replace('${HOUFIN_PROCESSED}',str(root.parents[1]/'processed')))
    derived=root/'inputs/established1951'
    if derived.exists():raise FileExistsError('preserve existing initialization inputs')
    m=pickle.load(open(source/'metadata.pkl','rb'));new=dict(m)
    years=np.asarray(m['years']);idx=np.flatnonzero(years==1951)
    if len(idx)!=1:raise ValueError('1951 absent from timeline')
    new['inv_timestep']=int(idx[0])
    if int(years[int(m['inv_timestep'])])!=1940:raise ValueError('unexpected original release year')
    derived.mkdir(parents=True)
    for p in source.iterdir():
        if p.name not in ['metadata.pkl','INPUT_CONTRACT.json']:(derived/p.name).symlink_to(p,target_is_directory=p.is_dir())
    with (derived/'metadata.pkl').open('wb') as f:pickle.dump(new,f)
    contract={'parent':str(source),'derived':str(derived),'changed_metadata_keys':['inv_timestep'],
        'old_year':1940,'new_year':1951,'old_index':int(m['inv_timestep']),'new_index':int(new['inv_timestep']),
        'old_metadata_sha256':digest(source/'metadata.pkl'),'new_metadata_sha256':digest(derived/'metadata.pkl'),
        'observations_features_grid_years_unchanged':True,
        'interpretation':'Initialize already established within-cell population before documented outside-cell expansion; not a reconstruction of the cage release. Spatial within-cell occupancy remains unresolved at this grid scale.'}
    write(derived/'INPUT_CONTRACT.json',contract)
    specs=[]
    for native,parent in [(False,'step33_nestcbc_abs_s41'),(True,'step34_abs_natdep_s41')]:
        for label,bird_equiv in [('lo',280.),('mid',560.),('hi',1120.)]:
            name=f"step35_est1951_{'nat' if native else 'ctl'}_{label}_s41"
            source_run=root/'runs'/parent;dest=root/'runs'/name
            if dest.exists():raise FileExistsError(dest)
            for line in (source_run/'CODE_MANIFEST.sha256').read_text().splitlines():
                sha,rel=line.split('  ',1)
                if digest(source_run/'code'/rel)!=sha:raise ValueError('parent source changed '+rel)
            dest.mkdir();shutil.copytree(source_run/'code',dest/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            code=dest/'code';conf=json.loads((code/f'config/overlays/grid54_{parent}.json').read_text())
            pop=conf['population_model'];units=bird_equiv/102.8
            pop['pinned_sites']['log_inv_total']=math.log(units)
            pop['release_allee_exemption_years']=0
            # Pinned parameter: this broad reference has no demographic gradient.
            # Remove the historical backcast median/precision from the experiment.
            pop['invasion_pulse_prior']['total_median']=1.
            pop['invasion_pulse_prior']['total_log_sd']=10.
            conf['input_dir']=str(derived)
            conf.pop('_second_wave',None)
            spec={'step':name,'parent':parent,'seed':41,'initialization_year':1951,
                'total_injection_units':units,'conditional_bird_equivalent':bird_equiv,
                'conversion_birds_per_unit':102.8,'conversion_status':'conditional gauge, not observed census or confidence interval',
                'state_brackets':'fixed broad sensitivity, not founder-count priors',
                'native_departure':native,'release_allee_exemption_years':0,'cbc_backcast_used':False,
                'input_contract':contract,'age_split':'inherited 50/50 adult and young initialization; annual maturation and breeding unchanged'}
            conf['_later_established_state']=spec
            for key in ['map','vis_map']:conf['run_names'][key]='age_map_{precision}_grid54_'+name
            write(code/f'config/overlays/grid54_{name}.json',conf)
            dc=json.loads((code/'config/overlays/grid54_data.json').read_text())
            dc.setdefault('timeline',{})['invasion_year']=1951
            write(code/'config/overlays/grid54_data.json',dc)
            shutil.copy2(source_run/'grid54_fit.sh',dest/'grid54_fit.sh')
            script=(source_run/'run_candidate.sh').read_text().replace(parent,name)
            script+=f'''\n# Exact contribution and budget audit on the allocated node's CPU.
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export PYTHONPATH=$CODE:$CODE/src:{root}/monitor
"$PY" {root}/monitor/run_contribution_audit.py --run "$REF" --out "$R/out/contribution" > "$R/out/contribution.log" 2>&1
'''
            (dest/'run_candidate.sh').write_text(script)
            write(dest/'RUN_SPEC.json',spec)
            manifest=[digest(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file()]
            (dest/'CODE_MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
            # Publish only after immutable config, source and metadata are complete.
            q=root/'queue/pending'/f'0610_{name}.sh';temp=q.with_suffix('.tmp')
            temp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {root.parents[1]}/venv/bin/python {root}/monitor/later_state_fits.py --root {root} --execute {name}\n')
            os.rename(temp,q);specs.append(spec)
            print('QUEUED',name,flush=True)
    write(root/'monitor/later_state_specs.json',specs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--execute');a=p.parse_args()
    if a.root.resolve()!=Path('/work/07980/bwa386/ls6/houfin/research/grid54'):raise ValueError('wrong experiment root')
    if a.execute:
        if not a.execute.startswith('step35_est1951_') or '/' in a.execute:raise ValueError('invalid run')
        run=a.root/'runs'/a.execute
        with (run/'attempt.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (run/'ATTEMPT_STARTED.json').exists():raise RuntimeError('existing attempt preserved')
            start=time.time();write(run/'ATTEMPT_STARTED.json',{'epoch':start})
            rc=subprocess.run(['timeout','--kill-after=30','7200','bash',str(run/'run_candidate.sh')]).returncode
            write(run/'RUN_RECEIPT.json',{'exit_code':rc,'elapsed_seconds':time.time()-start,'started':start})
        raise SystemExit(rc)
    stage(a.root)


if __name__=='__main__':main()
