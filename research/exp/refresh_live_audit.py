"""Refresh the full archive audit when launched fits/diagnostics change state."""
import argparse,json,subprocess,time,fcntl
from pathlib import Path
REPO=Path('/mnt/c/Dev/houfin-range-model');REMOTE='/work/07980/bwa386/ls6/houfin/research/grid54'
PY='/home/breallis/houfin/venv/bin/python';HOST='bwa386@ls6.tacc.utexas.edu';SOCKET=str(Path.home()/'.ssh/cm-tacc')
SSH=['ssh','-S',SOCKET,'-o','BatchMode=yes',HOST]
OUT=REPO/'research/experiments/FULL_PATHOLOGY_AUDIT_LIVE_20261101.json'


def refresh():
    subprocess.run(SSH+[f'/work/07980/bwa386/ls6/houfin/venv/bin/python {REMOTE}/monitor/full_pathology_audit.py --root {REMOTE} --out {REMOTE}/monitor/reports/FULL_PATHOLOGY_AUDIT_LIVE_20261101.json'],check=True,timeout=240)
    raw=subprocess.check_output(SSH+[f'cat {REMOTE}/monitor/reports/FULL_PATHOLOGY_AUDIT_LIVE_20261101.json'],timeout=60)
    json.loads(raw);tmp=OUT.with_suffix('.tmp.json');tmp.write_bytes(raw);tmp.replace(OUT)
    subprocess.run(SSH+[f'/work/07980/bwa386/ls6/houfin/venv/bin/python {REMOTE}/monitor/regional_establishment_audit.py --root {REMOTE} --audit {REMOTE}/monitor/reports/FULL_PATHOLOGY_AUDIT_LIVE_20261101.json --out {REMOTE}/monitor/reports/REGIONAL_ESTABLISHMENT_LIVE_20261101.json'],check=True,timeout=90,stdout=subprocess.DEVNULL)
    regional=subprocess.check_output(SSH+[f'cat {REMOTE}/monitor/reports/REGIONAL_ESTABLISHMENT_LIVE_20261101.json'],timeout=45)
    json.loads(regional);(OUT.parent/'REGIONAL_ESTABLISHMENT_LIVE_20261101.json').write_bytes(regional)
    subprocess.run(['bash',str(REPO/'.research_runtime/contribution_context.sh')],check=True,timeout=240)
    subprocess.run([PY,str(REPO/'research/exp/nest_site_fecundity_audit.py'),'--audit',str(REPO/'research/experiments/FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json'),'--out',str(REPO/'research/experiments/NEST_SITE_FECUNDITY_LIVE_20261101.json')],check=True,timeout=60,stdout=subprocess.DEVNULL)
    subprocess.run([PY,str(REPO/'research/exp/report_movement_followups.py'),'--root','/mnt/e/Datasets/houfin/movement_followup_20261101','--out',str(REPO/'research/experiments/MOVEMENT_ESTABLISHMENT_RESULTS_20261101.md')],check=True,timeout=60)
    subprocess.run([PY,str(REPO/'research/exp/review_initialization_results.py'),'--repo',str(REPO)],check=True,timeout=60)
    subprocess.run([PY,str(REPO/'research/exp/functional_stability_audit.py'),'--repo',str(REPO)],check=True,timeout=120)
    subprocess.run([PY,str(REPO/'research/exp/temporal_functional_stability.py')],check=True,timeout=60)
    subprocess.run([PY,str(REPO/'research/exp/eastern_regime_audit.py')],check=True,timeout=60)
    subprocess.run([PY,str(REPO/'research/exp/cross_model_process_audit.py')],check=True,timeout=240)
    subprocess.run([PY,str(REPO/'research/exp/plains_source_connectivity.py')],check=True,timeout=240)
    subprocess.run([PY,str(REPO/'research/exp/temporal_compensation_audit.py')],check=True,timeout=120)
    subprocess.run([PY,str(REPO/'research/exp/report_outstanding_work.py')],check=True,timeout=60)
    print('LIVE ARCHIVE AUDIT UPDATED',flush=True)
    error=REPO/'.research_runtime/live_audit_error.json'
    if error.exists():error.replace(error.with_name(f'live_audit_resolved_{time.time_ns()}.json'))


def main():
    lock=(REPO/'.research_runtime/full_audit_watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');a=p.parse_args();previous=None
    while not (REPO/'.research_runtime/STOP_LIVE_AUDIT').exists():
        try:
            records=[]
            for name in ['later_state_status.json','hmc_diagnostic_status.json']:
                path=REPO/'.research_runtime'/name
                if path.exists():
                    data=json.loads(path.read_text())
                    records.extend((name,s,b.get('queue_state'),b.get('fit_done'),b.get('contribution_done'),bool(b.get('report'))) for s,b in data['runs'].items())
            signature=tuple(records)
            signature+=tuple(('movement',p.parent.name,p.stat().st_mtime_ns) for p in sorted(Path('/mnt/e/Datasets/houfin/movement_followup_20261101').glob('*/READY.json')))
            if signature!=previous or a.once:refresh();previous=signature
        except Exception as e:
            (REPO/'.research_runtime/live_audit_error.json').write_text(json.dumps({'epoch':time.time(),'error':str(e)}))
            if a.once:raise
        if a.once:break
        time.sleep(60)


if __name__=='__main__':main()
