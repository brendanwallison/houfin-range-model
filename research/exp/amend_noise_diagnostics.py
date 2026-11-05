"""Version and validate a diagnostic-only amendment; retain model/checkpoint identity."""
import ast,hashlib,json,sys,time,types,subprocess,fcntl,textwrap
from pathlib import Path
import numpy as np
from noise_diagnostic_patch import patch,NEW_ACTIVE
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 tmp=p.with_name(p.name+'.next');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)

with (R/'monitor/noise_diagnostic_amendment.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 specs=[s for s in json.loads((R/'monitor/minor_flexibility_specs.json').read_text()) if s['step'].startswith('step45_')]
 validpath=R/'monitor/minor_flexibility_VALIDATED.json';valid=json.loads(validpath.read_text());plans=[]
 for s in specs:
  run=R/'runs'/s['step'];code=run/'code';file=code/'scripts/viz/map_diagnostics.py';amend=run/'diagnostic_amendments/noise_v1'
  if (amend/'receipt.json').exists():continue
  assert sha(run/'CODE_MANIFEST.sha256')==valid['manifests'][s['step']]
  manifest=(run/'CODE_MANIFEST.sha256').read_text()
  for line in manifest.splitlines():
   h,rel=line.split('  ',1);assert sha(code/rel)==h,(s['step'],rel)
  amended=patch(file.read_text());plans.append((s,run,file,amend,manifest,amended))
 assert len(specs)==18
 # Test patched diagnostic against independently assembled arrays and frozen model helpers.
 for family in ['cell_iid','regional']:
  spec=next(s for s in specs if s['family']==family);code=R/'runs'/spec['step']/'code'
  source=patch((code/'scripts/viz/map_diagnostics.py').read_text()) if not (R/'runs'/spec['step']/'diagnostic_amendments/noise_v1/receipt.json').exists() else (code/'scripts/viz/map_diagnostics.py').read_text()
  fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_process_noise_eps')
  helper=types.ModuleType('src.model.minor_flexibility');exec((code/'src/model/minor_flexibility.py').read_text(),helper.__dict__)
  prior=types.ModuleType('src.model.age_priors');sys.modules['src.model.age_priors']=prior;sys.modules['src.model.minor_flexibility']=helper
  namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<patched diagnostic>','exec'),namespace)
  for sigma in [.005,.01,.02]:
   prior._PN=dict(form=family,sigma=sigma,unit_cell_variance=True)
   active=np.array([0.,0.,1.,1.,1.,0.,0.,0.]);indices=np.flatnonzero(active)
   basis=np.array([[1.,0.],[0.,1.],[.6,.8],[-.8,.6]])
   raw=np.arange(12,dtype=float).reshape(3,4)/10 if family=='cell_iid' else np.arange(16,dtype=float).reshape(8,2)/10
   data=dict(pn_active=active,pn_time_indices=indices,pn_S=np.zeros((4,0)) if family=='cell_iid' else basis)
   actual=namespace['_process_noise_eps'](data,dict(process_noise_u=raw))
   expected=np.zeros((8,4))
   if family=='cell_iid':expected[indices]=sigma*raw-.5*sigma**2
   else:
    for t in indices:
     for cell in range(4):expected[t,cell]=sigma*sum(raw[t,j]*basis[cell,j] for j in range(2))-.5*sigma**2
   np.testing.assert_allclose(actual,expected,atol=1e-15)
   env=dict(np=np,sim=dict(process_noise_u=raw),data=data);exec(textwrap.dedent(NEW_ACTIVE),env)
   assert env['u'].shape==(3,4) if family=='cell_iid' else env['u'].shape==(3,2)
 # No visualization process may be reading an amended file at publication.
 ps=subprocess.check_output(['ssh','c301-002','ps -u bwa386 -o args='],text=True)
 for s,_,_,_,_,_ in plans:
  assert not any('scripts/viz/map_diagnostics.py' in line and s['step'] in line for line in ps.splitlines()),'diagnostic currently executing '+s['step']
 receipts=[]
 for s,run,file,amend,manifest,amended in plans:
  amend.mkdir(parents=True);(amend/'map_diagnostics.original.py').write_bytes(file.read_bytes());(amend/'CODE_MANIFEST.original.sha256').write_text(manifest)
  oldhash=sha(file);tmp=file.with_suffix('.next');tmp.write_text(amended);tmp.replace(file);newhash=sha(file)
  relative='./scripts/viz/map_diagnostics.py';oldline=oldhash+'  '+relative
  assert manifest.count(oldline)==1
  newmanifest=manifest.replace(oldline,newhash+'  '+relative)
  tmp=run/'CODE_MANIFEST.sha256.next';tmp.write_text(newmanifest);tmp.replace(run/'CODE_MANIFEST.sha256')
  receipt=dict(epoch=time.time(),step=s['step'],scope='diagnostic only: compact active-year arrays and mean-one log multiplier',
   old_diagnostic_sha256=oldhash,new_diagnostic_sha256=newhash,old_manifest_sha256=valid['manifests'][s['step']],new_manifest_sha256=sha(run/'CODE_MANIFEST.sha256'),
   model_sources_unchanged=True,config_unchanged=True,checkpoint_unchanged=True,validated_six_numeric_cases=True)
  write(amend/'receipt.json',receipt);receipts.append(receipt);valid['manifests'][s['step']]=receipt['new_manifest_sha256']
 valid.setdefault('diagnostic_amendments',[]).append(dict(epoch=time.time(),receipts=receipts,validation='compact-time and normalized-regional diagnostics match model innovations at all three sigma values'))
 write(validpath,valid);write(R/'monitor/NOISE_DIAGNOSTIC_AMENDMENT_20261103.json',dict(passed=True,receipts=receipts))
 failed=R/'queue/failed/0609_step45_cell_iid_s010_s41.sh';pending=R/'queue/pending'/failed.name
 if failed.exists():
  assert not pending.exists();log=R/'queue/logs/0609_step45_cell_iid_s010_s41.log'
  if log.exists():(R/'runs/step45_cell_iid_s010_s41/diagnostic_amendments/noise_v1/failed_queue_log.txt').write_bytes(log.read_bytes())
  failed.rename(pending)
 print('AMENDED_DIAGNOSTICS',len(receipts),'VALIDATED_CASES',6,'INDEPENDENT_01_REQUEUED',pending.exists(),flush=True)