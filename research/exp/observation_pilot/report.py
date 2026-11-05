"""Readable receipts for the isolated observation pilots; no model adoption."""
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
from pathlib import Path
import time


def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}


def report(root,target):
    lines=['# Joint-observation implementation — first results','',
      datetime.now(ZoneInfo('America/Chicago')).isoformat(),'',
      'This is a data/observation-model pilot, not a completed joint population refit. Existing population fits and TACC queue are unchanged. Methods and amendments are registered in JOINT_OBSERVATION_PILOT_20261101.md.','',
      '| Source | Preparation | Calibration attempt |','|---|---|---|']
    for label,folder in [('CBC','cbc'),('eBird Canada','ebird_CA'),('eBird US','ebird_US')]:
        p=root/folder;ready=read(p/'READY.json');progress=read(p/'progress.json');receipt=read(p/'CALIBRATION_RECEIPT.json')
        state='complete' if ready else progress.get('phase','queued')
        if progress and not ready:state+=f"; {progress.get('rows_read',0):,} rows scanned"
        cal='finished; review required' if receipt and receipt['exit_code']==0 else f"failed/timeout ({receipt['exit_code']})" if receipt else 'waiting/running'
        lines.append(f'| {label} | {state} | {cal} |')
    cbc=read(root/'cbc/READY.json');cal=read(root/'cbc/calibration.json')
    if cbc:
        lines+=['',f"CBC focal export: {cbc['circle_years']:,} circle-years at {cbc['circles']:,} circles, {cbc['year_min']}–{cbc['year_max']}. Raw date/year conflicts: {cbc['date_year_mismatches']}; excluded from primary calibration. Missing historical effort is retained as missing.",'']
    if cal:
        lines+=['Standard NB2 pilots use survey effort, geography/time and an optional circle panel. Counts remain winter observations. These are source-calibration baselines, not the final shared-population likelihood.','',
                '| Model | Converged | Temporal-test n | Temporal log score | Mean observed / predicted |','|---|---|---:|---:|---|']
        for name,value in cal['models'].items():
            s=value['scores'].get('temporal_test',{})
            lines.append(f"| {name} | {value['converged']} | {s.get('n')} | {s.get('mean_log_score',0):.4f} | {s.get('mean_observed',0):.2f} / {s.get('mean_predicted',0):.2f} |")
        lines+=['','The global polynomial baselines overpredict average counts, and the circle panel underpredicts the temporal-test mean. These simplified geographic/trend baselines are not adequate production calibrations. The circle-panel prediction is conditional on previously observed circles, selected using survey availability and geography, and is not a spatial-transfer test. The next likelihood comparison must use the ecological/population readout and retain blocked evaluation.']
    nest=read(root/'demography/nest_conversion_audit.json');bbl=read(root/'demography/bbl_integration_inventory.json')
    if nest:
        lines+=['',f"NestWatch audit reconciles {nest['likelihood_attempts']:,} training attempts ({nest['exposure_rows']:,} exposure rows; {nest['count_rows']:,} successful-count rows). {nest['share_training_since_2000']:.1%} date from 2000 onward.",
          f"There are {nest['location_years_with_multiple_reported_attempts']:,} location-years with multiple reported attempts, but no individual-female or complete-season denominator in the inspected schema. Annual attempts and breeding participation are not identified by those reporting counts."]
    if bbl:
        lines+=['',f"BBL inputs retained: {bbl['all_bands']:,} adult bands and {bbl['all_dead_recoveries']:,} dead recoveries. A differentiable port of the existing coarse regional recovery likelihood now exists and matches the independent implementation/finite-difference gradients in tests. It retains never-recovered birds, reporting eras and partial exposure. It is not yet connected to the spatial population simulator; the full movement/survival adapter and approximation checks remain required."]
    for country in ['CA','US']:
        cal=read(root/f'ebird_{country}/calibration.json')
        if not cal:continue
        lines+=['',f'eBird {country}: effort-aware hurdle calibration results','',
          '| Season | n after thinning | Encounter converged | Positive-count converged | Temporal Brier | Temporal observed / predicted detection |','|---|---:|---|---|---:|---|']
        for season,value in cal.get('seasons',{}).items():
            if 'scores' not in value:
                lines.append(f"| {season} | {value.get('n')} | {value.get('status')} | | | |")
                continue
            scores=value['scores']
            t=scores.get('temporal_test',{})
            lines.append(f"| {season} | {value['n']} | {value['encounter']['converged']} | {value['conditional_count']['converged']} | {t.get('encounter_brier',0):.4f} | {t.get('observed_detection_rate',0):.4f} / {t.get('predicted_detection_rate',0):.4f} |")
        lines+=['','Convergence is not acceptance. The Canadian baselines underpredict temporal-test encounter rates. A low Brier score for a rarely detected species does not by itself establish calibration; inspect observed versus predicted rates and compare the shared population/ecological readout before integration.']
    lines+=['','The likelihood tests cover checklist denominator selection, group deduplication, X detections, winter-year alignment, zero-truncated NB2 conditioning, and the differentiable recovery kernel. A successful source fit is not a biological identification gate.','',
            f'Artifacts: `{root}`. Preparation and calibration jobs run persistently in the observation-io, observation-cpu and observation-watch pueue groups. Full-country scans are required before zero-filled focal products become ready.','',
            'Next: completed eBird exports automatically trigger the registered effort-aware encounter/positive-count calibration, using the pinned 216-km spatial holdout and 2021–25 temporal holdout. Then evaluate the shared annual population readout and modern seasonal mapping. No founder prior or demographic parameter has been changed by this pilot.']
    target.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.watch:
        import fcntl
        with (a.root/'report.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while not (a.root/'STOP_REPORT').exists():
                report(a.root,a.out)
                if all((a.root/f/'CALIBRATION_RECEIPT.json').exists() for f in ['cbc','ebird_CA','ebird_US']):break
                time.sleep(30)
    else:report(a.root,a.out)
