"""Audit the exact existing density at retained points; never change or optimize it."""
import argparse
import json
import pickle
from pathlib import Path

from src.model import hmc_common as hc
import jax
import jax.numpy as jnp
import numpy as np
from src.config_utils import load_age_model_config
from src.model.age_priors import build_model_2d
from src.model.data_loading import load_data
from src.model.runtime_diagnostics import require_gpu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True)
p.add_argument('--curvature-probes', type=int, default=0)
p.add_argument('--steps', type=float, nargs='+', default=[1e-4, 1e-5, 1e-6, 1e-7])
a = p.parse_args()
if any(not np.isfinite(step) or step <= 0 for step in a.steps):
    raise ValueError('finite-difference steps must be finite and positive')
device = require_gpu('retained-point gradient audit')
latents, checkpoint = hc.load_map()
hc.check_model_drift(checkpoint['fingerprint_payload'])
data = load_data(load_age_model_config()['input_dir'], target_device=device, precision='float64', verbose=False)
arrays, static = hc.split_data(data)
kw = {'arrays': arrays}
model = hc.hide_deterministics(hc.array_model(build_model_2d, static))
z, potential, _ = hc.potential_with_args(model, kw, latents)
names, x, unravel = hc.flatten_sorted(z)
roundtrip = np.asarray(x)
state_path = Path(hc.map_dir()) / 'refine_state.pkl'
state_info = None
if state_path.exists():
    with state_path.open('rb') as stream:
        state = pickle.load(stream)
    exact = np.asarray(state['x'])
    assert exact.shape == roundtrip.shape and np.isfinite(exact).all()
    state_info = {'path': str(state_path), 'last_U': state['history'][-1][1],
                  'roundtrip_max_abs_difference': float(np.max(np.abs(exact-roundtrip)))}
    x = jnp.asarray(exact)
f = jax.jit(lambda v, k: potential(unravel(v), k))
vg = jax.jit(jax.value_and_grad(lambda v, k: potential(unravel(v), k)))
value, gradient = vg(x, kw)
g = np.asarray(gradient)
offset = np.cumsum([0] + [int(np.size(z[n])) for n in names])
blocks = {name: {'norm': float(np.linalg.norm(g[offset[i]:offset[i+1]])),
                 'max_abs': float(np.max(np.abs(g[offset[i]:offset[i+1]]))),
                 'size': int(offset[i+1]-offset[i])} for i, name in enumerate(names)}
directions = [('gradient_unit', g / max(np.linalg.norm(g), 1e-300))]
for index in np.argsort(np.abs(g))[-3:][::-1]:
    direction = np.zeros_like(g); direction[index] = 1
    block = names[int(np.searchsorted(offset[1:], index, side='right'))]
    directions.append((f'coordinate_{index}_{block}', direction))
checks = []
for name, direction in directions:
    predicted = float(np.dot(g, direction))
    for h in a.steps:
        up = float(f(x + h * jnp.asarray(direction), kw))
        down = float(f(x - h * jnp.asarray(direction), kw))
        finite = np.isfinite(up) and np.isfinite(down)
        fd = (up-down)/(2*h) if finite else None
        checks.append({'direction': name, 'step': h, 'autodiff': predicted,
                       'central_difference': fd, 'U_plus': up if np.isfinite(up) else None,
                       'U_minus': down if np.isfinite(down) else None,
                       'relative_error': abs(fd-predicted)/max(1., abs(fd), abs(predicted)) if finite else None})
        print(json.dumps(checks[-1]), flush=True)
result = {'source_point': hc.map_dir(), 'exact_state': state_info,
          'roundtrip_U': float(f(jnp.asarray(roundtrip), kw)),
          'U': float(value), 'grad_norm': float(np.linalg.norm(g)),
          'repeat_U': [float(f(x, kw)) for _ in range(3)], 'blocks': blocks,
          'finite_difference': checks, 'interpretation': 'Diagnostic only; no convergence gate or model changed.'}
if a.curvature_probes:
    if a.curvature_probes < 4:
        raise ValueError('use at least four curvature probes')
    grad_f = jax.grad(lambda v, k: potential(unravel(v), k))
    hvp = jax.jit(lambda point, direction, k: jax.jvp(lambda v: grad_f(v, k), (point,), (direction,))[1])
    rng = np.random.default_rng(20261025)
    diagonal = np.zeros_like(g)
    for index in range(a.curvature_probes):
        direction = rng.choice([-1., 1.], size=g.shape)
        diagonal += direction * np.asarray(hvp(x, jnp.asarray(direction), kw)) / a.curvature_probes
        print('curvature probe', index+1, flush=True)
    curvature = {name: float(np.mean(diagonal[offset[i]:offset[i+1]])) for i, name in enumerate(names)}
    scales = {name: float(1 / np.sqrt(np.clip(abs(value), 1., 1e12))) for name, value in curvature.items()}
    result.update({'curvature_probes': a.curvature_probes, 'curvature_seed': 20261025,
                   'block_mean_diagonal_curvature': curvature, 'coordinate_scale_by_site': scales})
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
temporary = out.with_suffix('.tmp.json')
temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
temporary.replace(out)
print('saved', out, flush=True)
