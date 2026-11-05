"""Local reproduction-only numerical audit; no fitting or change to model biology."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from src.model import age_forward as af

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
af.ALLEE_AGG_K = 3.0


def trajectory(log_seed, fecundity, dtype):
    n = jnp.exp(jnp.asarray(log_seed, dtype=dtype))
    def step(state, _):
        adults, juveniles = state
        return af.reproduction_age_structured(
            adults, juveniles, jnp.zeros_like(adults),
            jnp.asarray(.65, dtype), jnp.asarray(.35, dtype),
            jnp.asarray(fecundity, dtype), jnp.asarray(100., dtype),
            jnp.asarray(2., dtype), jnp.asarray(.5, dtype),
            allee_relative=False), None
    final, _ = jax.lax.scan(step, (n*.5, n*.5), None, length=80)
    return jnp.log1p(final[0]+final[1])


records = []
for dtype, label in [(jnp.float32, 'float32'), (jnp.float64, 'float64')]:
    vg = jax.jit(jax.value_and_grad(lambda seed, f: trajectory(seed, f, dtype)))
    fn = jax.jit(lambda seed, f: trajectory(seed, f, dtype))
    for fecundity in [4., 8., 16.]:
        for log_seed in np.linspace(-90., 7., 80):
            point = jnp.asarray(log_seed, dtype)
            value, grad = vg(point, fecundity)
            differences = []
            for h in [1e-2, 1e-4, 1e-6, 1e-8]:
                plus, minus = float(fn(point+h, fecundity)), float(fn(point-h, fecundity))
                actual_step = float(jnp.asarray(point+h, dtype)-jnp.asarray(point-h, dtype))
                fd = (plus-minus)/actual_step if actual_step else None
                differences.append({'requested_step': h, 'actual_span': actual_step,
                                    'finite_difference': fd,
                                    'relative_error': None if fd is None else abs(fd-float(grad))/max(abs(fd),abs(float(grad)),1e-12)})
            records.append({'dtype': label, 'log_seed': float(log_seed), 'fecundity': fecundity,
                            'terminal_log1p_N': float(value), 'derivative_log_seed': float(grad),
                            'finite_difference': differences})
    # Exact zero must remain extinct in this no-immigration reproduction subsystem.
    zero = af.reproduction_age_structured(*[jnp.asarray(0., dtype)]*3,
        jnp.asarray(.65, dtype), jnp.asarray(.35, dtype), jnp.asarray(8., dtype),
        jnp.asarray(100., dtype), jnp.asarray(2., dtype), jnp.asarray(.5, dtype), allee_relative=False)
    assert all(float(v)==0 for v in zero)
if not all(np.isfinite(row['terminal_log1p_N']) and np.isfinite(row['derivative_log_seed']) for row in records):
    raise RuntimeError('Nonfinite reproductive-subsystem audit result')
# Coarse log spacing misses a sharp establishment boundary. Locate that boundary
# independently in float64, then inspect progressively smaller neighborhoods.
boundaries = []
fn64 = jax.jit(lambda seed, f: trajectory(seed, f, jnp.float64))
for fecundity in [4., 8., 16.]:
    low, high = -90., 7.
    high_value = float(fn64(high, fecundity))
    if high_value < .01:
        continue
    target = high_value/2
    for _ in range(55):
        middle = (low+high)/2
        if float(fn64(middle, fecundity)) < target:
            low = middle
        else:
            high = middle
    center = (low+high)/2
    boundary = {'fecundity': fecundity, 'log_seed_center_float64': center,
                'target_terminal_log1p_N': target, 'probes': []}
    for dtype, label in [(jnp.float32, 'float32'), (jnp.float64, 'float64')]:
        vg = jax.jit(jax.value_and_grad(lambda seed: trajectory(seed, fecundity, dtype)))
        fn = jax.jit(lambda seed: trajectory(seed, fecundity, dtype))
        for delta in [0., -1e-2, 1e-2, -1e-4, 1e-4, -1e-6, 1e-6, -1e-8, 1e-8, -1e-10, 1e-10]:
            x = jnp.asarray(center+delta, dtype)
            value, grad = vg(x)
            fd_checks = []
            for h in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
                xp, xm = jnp.asarray(x+h, dtype), jnp.asarray(x-h, dtype)
                span = float(xp-xm)
                fd = (float(fn(xp))-float(fn(xm)))/span if span else None
                fd_checks.append({'requested_step': h, 'actual_span': span, 'finite_difference': fd,
                                  'relative_error': None if fd is None else abs(fd-float(grad))/max(abs(fd),abs(float(grad)),1e-12)})
            boundary['probes'].append({'dtype': label, 'offset': delta, 'actual_log_seed': float(x),
                                       'terminal_log1p_N': float(value), 'gradient': float(grad),
                                       'finite_difference': fd_checks})
    boundaries.append(boundary)
result = {'interpretation': 'Actual absolute-density k3 reproduction,80 years, no dispersal/immigration. Diagnostic subsystem only; does not validate full simulator thresholds or fitted launch.',
          'source_sha256': hashlib.sha256(Path(af.__file__).read_bytes()).hexdigest(),
          'audit_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'jax': jax.__version__, 'backend': jax.default_backend(), 'exact_zero_preserved': True,
          'records': records, 'establishment_boundaries': boundaries}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=2, allow_nan=False))
print('Wrote',a.out,len(records),'finite cases; exact zero preserved',flush=True)
