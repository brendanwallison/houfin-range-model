"""Release one timing calibration, then only cost-supported HMC comparisons."""
import json,os,time
from pathlib import Path
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')


def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def saturated_seconds(chains,warmup,draws,depth,seconds_per_step):
    return chains*(warmup+draws)*(2**depth-1)*seconds_per_step


def main():
    held=R/'queue/held_geometry';held.mkdir(exist_ok=True)
    costheld=R/'queue/held_cost';costheld.mkdir(exist_ok=True)
    specs=read(R/'monitor/hmc_diagnostic_specs.json') or []
    samplers=[s for s in specs if s['mode']!='geometry']
    choicefile=R/'monitor/hmc_calibration_choice.json';choice=read(choicefile)
    if choice is None:
        for spec in samplers:
            if spec['mode']!='nuts':continue
            report=read(R/'runs'/spec['step']/'out/hmc_geometry/READY.json')
            if report and report.get('screen_kind')=='comparative_geometry_screen':
                choice={'step':spec['step'],'mode':'nuts','epoch':time.time()};choicefile.write_text(json.dumps(choice)+'\n');break
    calibration=read(R/'runs'/choice['step']/'out/hmc_screen/chain0/PROGRESS.json') if choice else None
    calibrated=bool(calibration and calibration['completed_iterations']>=8 and calibration.get('seconds_per_leapfrog_median'))
    forecasts=[]
    for spec in samplers:
        mode,step=spec['mode'],spec['step'];name=f"{spec.get('queue_priority','0640')}_hmc_{mode}_{step}.sh"
        pending=R/'queue/pending'/name;geometry_held=held/name;cost_held=costheld/name
        report=read(R/'runs'/step/'out/hmc_geometry/READY.json')
        own_calibration=bool(choice and choice['step']==step and choice['mode']==mode)
        available=bool(report and report.get('screen_kind')=='comparative_geometry_screen')
        rate=report['grad_seconds_median'] if available else None
        if calibrated and rate is not None:rate=max(rate,calibration['seconds_per_leapfrog_median'])
        s=spec['sampling'];forecast=saturated_seconds(s['chains'],s['warmup_per_chain'],s['draws_per_chain'],s['max_depth'],rate) if rate is not None else None
        permitted=available and (own_calibration or (calibrated and forecast is not None and forecast<=spec['time_limit_seconds']))
        target=geometry_held if not available else pending if permitted else cost_held
        for source in [pending,geometry_held,cost_held]:
            if source!=target and source.exists():
                try:os.rename(source,target)
                except FileNotFoundError:pass
        forecasts.append({'step':step,'mode':mode,'gradient_or_calibrated_step_seconds':rate,
            'saturated_trajectory_hours':forecast/3600 if forecast is not None else None,
            'attempt_budget_hours':spec['time_limit_seconds']/3600,'timing_calibration':own_calibration,
            'calibration_available':calibrated,'release_permitted':permitted,
            'interpretation':'Compute-cost forecast at depth saturation, not expected completion time or a model-quality verdict'})
    (R/'monitor/hmc_cost_forecasts.json').write_text(json.dumps(forecasts,indent=2)+'\n')


if __name__=='__main__':main()
