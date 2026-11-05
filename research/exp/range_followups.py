"""Automatically queue the pre-registered founder held-out comparisons.

Reads the local monitor snapshot, never changes the TACC queue. GPU dispatch and
scientific adoption remain separate. Every action has a durable task label.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

try:
    from .range_watch import command, atomic_json, read_json
except ImportError:
    from range_watch import command, atomic_json, read_json

COMPARISONS={"step32_cbc_yb":"step29_yb_native","step32_cbc_native":"step3_native",
             "step32_cbclo_yb":"step29_yb_native","step32_cbchi_yb":"step29_yb_native",
             "step31_nest":"step29_yb_native","step31_nest_bloose":"step29_yb_native",
             "step31_nestonly":"step29_yb_native",
             "step33_nestcbc_s41":"step31_nestonly","step33_nestcbc_s42":"step31_nestonly",
             "step33_nestcbc_natdep_s41":"step31_nestonly","step33_nestcbc_natdep_s42":"step31_nestonly",
             "step33_nestcbc_abs_s41":"step31_nestonly"}
try:
    from .range_second_wave import VARIANTS
except ImportError:
    from range_second_wave import VARIANTS
COMPARISONS.update({'step34_'+v: spec[0] for v,spec in VARIANTS.items()})
COMPARISONS.update({f"step35_est1951_{mode}_{level}_s41":parent
    for mode,parent in [('ctl','step33_nestcbc_abs_s41'),('nat','step34_abs_natdep_s41')]
    for level in ['lo','mid','hi']})
COMPARISONS.update({f"step36_nonest_{mode}_s{seed}":
    f'step34_abs_natdep_s{seed}' if mode=='nat' else 'step33_nestcbc_abs_s41' if seed==41 else 'step34_abs_s42'
    for mode in ['ctl','nat'] for seed in [41,42]})
COMPARISONS.update({'step35_est1951_ctl_mid_s42':'step34_abs_s42',
    'step35_est1951_nat_mid_s42':'step34_abs_natdep_s42',
    'step36_est1951_nonest_ctl_s41':'step35_est1951_ctl_mid_s41',
    'step36_est1951_nonest_nat_s41':'step35_est1951_nat_mid_s41'})
_cbc_specs=Path('/mnt/c/Dev/houfin-range-model/research/experiments/CBC_ESTABLISHED_RUNS_20261101.json')
if _cbc_specs.exists():
    COMPARISONS.update({s['step']:s['reference'] for s in json.loads(_cbc_specs.read_text())})
_overflow_specs=_cbc_specs.with_name('OVERFLOW_EARLY_RUNS_20261102.json')
if _overflow_specs.exists():
    COMPARISONS.update({s['step']:s['reference'] for s in json.loads(_overflow_specs.read_text())})
_range_specs=_cbc_specs.with_name('HISTORICAL_RANGE_RUNS_20261102.json')
if _range_specs.exists():
    COMPARISONS.update({s['step']:s['reference'] for s in json.loads(_range_specs.read_text())})
for _registered in ('GRADUAL_DEPARTURE_RUNS_20261102.json',
                    'CROWDING_CURVATURE_RUNS_20261102.json',
                    'KFLOOR_COMBINATION_RUNS_20261102.json',
                    'MINOR_FLEXIBILITY_RUNS_20261103.json','OUTSTANDING_TRANSFER_RUNS_20261103.json'):
    _path = _cbc_specs.with_name(_registered)
    if _path.exists():
        COMPARISONS.update({s['step']: s['reference'] for s in json.loads(_path.read_text())})
HOST="bwa386@ls6.tacc.utexas.edu"
REMOTE="/work/07980/bwa386/ls6/houfin"
REPO=Path("/mnt/c/Dev/houfin-range-model")
STATE=REPO/".research_runtime/range_monitor"
PUEUE=str(Path.home()/".local/bin/pueue")
PRECISION_STEPS=('step31_nestonly','step33_nestcbc_s41','step33_nestcbc_natdep_s41','step33_nestcbc_abs_s41')


def queue_precision(step):
    if step not in PRECISION_STEPS:raise ValueError('unregistered precision probe')
    socket=str(Path.home()/'.ssh/cm-tacc')
    name=f'0580_precision_{step}.sh'
    # All interpolated tokens come from the fixed registration above.
    script=f'''set -eu
R={REMOTE}/research/grid54
for state in pending running done failed; do if [ -f "$R/queue/$state/{name}" ]; then exit 0; fi; done
printf '%s\\n' '#!/bin/bash' 'set -euo pipefail' 'timeout 1200 bash {REMOTE}/research/grid54/monitor/range_precision_probe.sh {step}' > "$R/queue/pending/.{name}.tmp"
mv "$R/queue/pending/.{name}.tmp" "$R/queue/pending/{name}"
'''
    subprocess.run(['ssh','-S',socket,'-o','BatchMode=yes',HOST,'bash','-s'],input=script,text=True,check=True,timeout=30)
    atomic_json(STATE/f'precision_{step}.queued.json',{'step':step,'epoch':time.time()})


def digest(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def rescore(step):
    if step not in COMPARISONS:
        raise ValueError("unregistered follow-up")
    socket=str(Path.home()/".ssh/cm-tacc")
    name=f"refined_float64__age_map_float32_grid54_{step}_quick90"
    parent=Path(REMOTE)/"processed/model_results"
    temporary=parent/f".range_download_{step}_{uuid.uuid4().hex}"
    temporary.mkdir()
    subprocess.run(["scp","-r","-o",f"ControlPath={socket}",f"{HOST}:{REMOTE}/processed/model_results/{name}/.",str(temporary)+"/"],check=True,timeout=180)
    report=read_json(temporary/"refine_report.json")
    if not report or not report.get("valid"):
        raise ValueError("refinement is not accepted")
    destination=parent/name
    if destination.exists():
        if digest(destination/"map_checkpoint.pkl")!=digest(temporary/"map_checkpoint.pkl"):
            raise ValueError("existing local refined point differs; retained download for review")
    else:
        os.rename(temporary,destination)
    # The existing scorer reads run-local status markers and diagnostics as well
    # as the refined checkpoint. Preserve their provenance instead of inventing
    # completion markers from a successful download.
    local_run=Path("/mnt/e/Datasets/houfin/grid54/runs")/step
    if step.startswith(tuple(f'step{n}_' for n in range(34,48))) and not (local_run/'code').exists():
        staged_source=local_run.with_name('.download_source_'+step)
        staged_source.mkdir(exist_ok=True)
        for relative in ('code','CODE_MANIFEST.sha256','RUN_SPEC.json'):
            subprocess.run(['scp','-r','-o',f'ControlPath={socket}',
                f'{HOST}:{REMOTE}/research/grid54/runs/{step}/{relative}',str(staged_source)+'/'],check=True,timeout=180)
        for line in (staged_source/'CODE_MANIFEST.sha256').read_text().splitlines():
            sha,relative=line.split('  ',1)
            if digest(staged_source/'code'/relative)!=sha:
                raise ValueError('downloaded source manifest mismatch')
        if local_run.exists():raise ValueError('partial local source retained for inspection')
        os.rename(staged_source,local_run)
    local_out=local_run/"out"
    (local_out/"viz_ref").mkdir(parents=True,exist_ok=True)
    for relative in ("FIT_DONE","REFINEMENT_DONE","viz_ref/metrics.json"):
        staged=STATE/f"download_{step}_{Path(relative).name}"
        subprocess.run(["scp","-o",f"ControlPath={socket}",f"{HOST}:{REMOTE}/research/grid54/runs/{step}/out/{relative}",str(staged)],check=True,timeout=30)
        target=local_out/relative
        if target.exists() and digest(target)!=digest(staged):
            raise ValueError(f"existing local diagnostic differs: {target}")
        if not target.exists():
            target.write_bytes(staged.read_bytes())
    script=(REPO/"research/exp/rescore_run.sh").read_text()
    # Same scorer/configuration; move only CPU affinity away from local MAP cores.
    wrapper=STATE/f"rescore_{step}.sh"
    wrapper.write_text(script.replace("taskset -c 0-3","taskset -c 8-11"))
    subprocess.run(["bash",str(wrapper),step,COMPARISONS[step]],check=True)
    source=Path("/mnt/e/Datasets/houfin/grid54/baselines/rescore")/step/"rescore.json"
    result=read_json(source)
    if result is None:
        raise ValueError("scorer did not produce rescore.json")
    if not result.get("status",{}).get("refinement_done") or not result.get("status",{}).get("refine_valid"):
        raise ValueError("scorer did not verify accepted refinement status")
    if result.get("parity_pass") is False:
        raise ValueError("prediction parity check failed")
    atomic_json(STATE/f"rescore_{step}.json",result)
    atomic_json(STATE/f"rescore_{step}.done.json",{"step":step,"reference":COMPARISONS[step],"epoch":time.time(),"checkpoint_sha256":digest(destination/"map_checkpoint.pkl")})


def watch():
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/"followups.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (STATE/"followups.pid").write_text(str(os.getpid())+"\n")
        while not (STATE/"STOP_FOLLOWUPS").exists():
            try:
                latest=read_json(STATE/"latest.json") or {}
                remote=latest.get("remote",{})
                if time.time()-remote.get("epoch",0)<180:
                    tasks=json.loads(command([PUEUE,"status","--json"]))["tasks"]
                    labels={v.get("label"):k for k,v in tasks.items()}
                    for step in COMPARISONS:
                        label=f"range-autonomy/rescore/{step}"
                        run=remote.get("experiments",{}).get(step,{})
                        if run.get("fit_done") and (run.get("report") or {}).get("valid") and label not in labels:
                            response=command([PUEUE,"add","--group","range-cpu","--label",label,
                                f"env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/breallis/houfin/venv/bin/python {REPO}/research/exp/range_followups.py rescore {step}"])
                            with (STATE/"followup_events.jsonl").open("a") as stream:
                                stream.write(json.dumps({"epoch":time.time(),"step":step,"response":response.strip()})+"\n")
                    for step in PRECISION_STEPS:
                        run=remote.get('experiments',{}).get(step,{})
                        if run.get('fit_done') and (run.get('report') or {}).get('valid') and not (STATE/f'precision_{step}.queued.json').exists():
                            queue_precision(step)
                (STATE/"followups.heartbeat").write_text(str(time.time())+"\n")
                overnight_report()
            except Exception as exc:
                with (STATE/"followup_errors.jsonl").open("a") as stream:
                    stream.write(json.dumps({"epoch":time.time(),"error":str(exc)})+"\n")
            time.sleep(60)


def overnight_report():
    """Descriptive receipt of completed work, not an automatic adoption rule."""
    import math
    latest=read_json(STATE/"latest.json") or {}
    remote=latest.get("remote",{})
    experiments=remote.get("experiments",{})
    lines=["# Overnight results — automatically updated", "",
           f"Snapshot UTC: {remote.get('checked_utc','unavailable')}", "",
           "These are descriptive readouts. Intercepts are not spatial mean vital rates; point-score gains are not equivalence tests.", "",
           "| Fit | State | Added distance cost h | Fecundity intercept | Juvenile offset | Paired spatial log-score change |",
           "|---|---|---:|---:|---:|---|"]
    records={}
    for step in COMPARISONS:
        run=experiments.get(step,{})
        report=run.get("report") or {}
        status="complete" if run.get("fit_done") and report.get("valid") else "running/pending"
        if report and not report.get("valid"): status="refinement rejected"
        params=run.get("scalars",{})
        value=lambda key: "—" if key not in params else f"{params[key]:.4f}"
        f=params.get("alpha_f_auto_loc")
        fec="—" if f is None else f"{max(f,0)+math.log1p(math.exp(-abs(f))):.3f}"
        score=read_json(STATE/f"rescore_{step}.json") or {}
        paired=(score.get("paired_vs_ref") or {}).get("diff",{}).get("S_real_marg")
        text="pending" if not paired else f"{paired['mean']:+.4f} [{paired['lo90']:+.4f}, {paired['hi90']:+.4f}] (90%)"
        lines.append(f"| {step} | {status} | {value('journey_distance_hazard_auto_loc')} | {fec} | {value('alpha_j_auto_loc')} | {text} |")
        records[step]={"state":status,"scalars":params,"metrics":run.get("metrics"),"paired":paired,"pathology":run.get("pathology"),"front_speed":run.get("front_speed")}
    lines += ["", "## Whole-configuration biological scorecard", "",
              "Population-weighted modern rates; early/modern eastern lambda uses land means. Sj here is stayer survival. Missing readouts remain pending.", "",
              "| Fit | Adult survival median | Juvenile/adult ratio median | West population in sinks | Plains K land q10 (relative units) | East lambda early → modern | Realized fecundity at nests |",
              "|---|---:|---:|---:|---:|---|---:|"]
    for step in ["step29_yb_native","step31_nestonly","step31_nest","step31_nest_bloose","step32_cbc_yb"]+[n for n in COMPARISONS if n.startswith(("step33_","step34_"))]:
        p=experiments.get(step,{}).get("pathology") or remote.get("audits",{}).get(f"{step}_pathology.json")
        if not p or p.get("status")!="complete":
            lines.append(f"| {step} | pending | | | | | |")
            continue
        modern=p['regions']['all_land']['modern']
        sa=modern['Sa']['population']; ratio=modern['Sj_over_Sa']['population']
        west=p['regions']['west']['modern']['sink_population_share']
        plains=p['regions']['plains']['modern']['K_relative']['land']['q10_50_90'][0]
        east=p['regions']['east']; e=east['early']['lambda0']['land']['mean']; m=east['modern']['lambda0']['land']['mean']
        f=p.get('nests',{}).get('nest_F_real'); ftext='—' if not f else f"{f['mean']:.3f}"
        lines.append(f"| {step} | {sa['q10_50_90'][1]:.3f} | {ratio['q10_50_90'][1]:.3f} | {west:.1%} | {plains:.4f} | {e:.3f} → {m:.3f} | {ftext} |")
    lines += ["", "## Synthetic distance-cost checks", "",
              "Known planted distance cost: h = .03. The following are estimates from simulated observations, not real-data movement measurements.", ""]
    for step in ("step31_j1rec_nonest","step31_nestrec","step31_nestrec_bloose","step31_nestrec_bwrong"):
        run=experiments.get(step,{})
        h=run.get("scalars",{}).get("journey_distance_hazard_auto_loc")
        if h is not None:
            lines.append(f"- {step}: h = {h:.5f}; absolute estimation error {abs(.03-h):.5f}.")
    lines += ["", "Nest data improved estimation in the existing pinned-attempts simulation. Remaining compensator changes and sensitivity to attempts per season qualify precision; they do not erase that improvement.", "",
              "## Audit results", ""]
    controller=remote.get('second_wave') or {}
    lines += [f"Second-wave controller: {controller.get('state','awaiting snapshot')}; staged: {len(controller.get('staged',{}))}; charged/reserved GPU-hours: {controller.get('charged_gpu_hours',0):.2f} of {controller.get('budget_gpu_hours',36)}.",
              f"Controller error: {controller.get('error','none')}; budget deferrals: {controller.get('deferred',{})}; blocked: {controller.get('blocked',{})}", ""]
    for name,audit in remote.get("audits",{}).items():
        lines.append(f"- {name}: collected ({audit.get('status','see JSON')}).")
        points=audit.get("points",{})
        if "generating" in points and "fitted" in points:
            delta=points["generating"]["refinement_potential"]-points["fitted"]["refinement_potential"]
            lines.append(f"  Generating-minus-fitted refinement objective: {delta:+.3f}; saved-objective agreement: {audit.get('stored_objective_agreement')}.")
    if not remote.get("audits"): lines.append("Fixed-point objective audits pending.")
    atomic_json(STATE/"overnight_results.json",{"checked_utc":remote.get("checked_utc"),"fits":records,"audits":remote.get("audits",{})})
    path=REPO/"research/experiments/OVERNIGHT_RESULTS_20261031.md"
    tmp=path.with_suffix(".tmp.md"); tmp.write_text("\n".join(lines)+"\n"); os.replace(tmp,path)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["watch","rescore","report"])
    parser.add_argument("step",nargs="?")
    args=parser.parse_args()
    if args.mode=="watch": watch()
    elif args.mode=="report": overnight_report()
    else: rescore(args.step)
