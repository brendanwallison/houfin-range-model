"""Dispatch the isolated conditioning pilot only after valid curvature inputs exist."""
import json
import math
from pathlib import Path

root = Path('/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025')
queue = root.parent / 'age_model/queue2'
result = json.loads((root / 'gradient_audit/fd_control64_scale.json').read_text())
scales = result['coordinate_scale_by_site']
assert scales and all(math.isfinite(value) and value > 0 for value in scales.values())
name = '9970_preconditioned_control64.sh'
if not any((queue / state / name).exists() for state in ('pending', 'running', 'done', 'failed')):
    temporary = queue / 'pending' / (name + '.tmp')
    temporary.write_text((root / 'diagnostics/9970_preconditioned_control64.deferred.sh').read_text())
    temporary.replace(queue / 'pending' / name)
    print('queued', name)
