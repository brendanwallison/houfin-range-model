"""Run against the complete frozen age environment, with JAX on CPU."""
import importlib.util
import pickle
from pathlib import Path
import tempfile
import sys

spec = importlib.util.spec_from_file_location('scaled_refiner', sys.argv[1])
refiner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refiner)
import jax.numpy as jnp
import numpy as np

settings = dict(max_iter=100, memory=10, tol_du=1e-14, tol_window=5,
                tol_grad=1e-5, plateau_grad_max=1e-3, ckpt_every=10, log_every=100)
f = lambda x, k: .5 * jnp.sum(k['precision'] * (x-k['optimum'])**2)
kw = {'precision': jnp.array([1., 1e10]), 'optimum': jnp.array([3., -2.])}
with tempfile.TemporaryDirectory() as directory:
    x, history, reason = refiner.lbfgs_minimize(f, jnp.zeros(2), kw, settings,
        str(Path(directory)/'scaled.pkl'), 'toy_scaled', log=lambda _: None,
        coordinate_scale=jnp.array([1., 1e-5]))
    np.testing.assert_allclose(x, kw['optimum'], atol=1e-12)
    with (Path(directory)/'scaled.pkl').open('rb') as stream:
        np.testing.assert_array_equal(pickle.load(stream)['x'], np.asarray(x))
    assert history[-1][2] < settings['tol_grad'], (history, reason)
    np.testing.assert_allclose(history[0][2], np.linalg.norm([3., -2e10]), rtol=1e-12)
    # A tiny transformed gradient must not masquerade as a small RAW gradient.
    _, history, _ = refiner.lbfgs_minimize(f, jnp.zeros(2), kw, {**settings,'max_iter':1},
        str(Path(directory)/'tiny.pkl'), 'toy_tiny', log=lambda _: None,
        coordinate_scale=jnp.array([1e-16, 1e-16]))
    assert history[0][2] > 1e10
    path = str(Path(directory)/'resume.pkl')
    refiner.lbfgs_minimize(f, jnp.zeros(2), kw, {**settings,'max_iter':1}, path,
        'toy_resume', log=lambda _: None, coordinate_scale=jnp.array([1.,1e-5]))
    resumed, _, _ = refiner.lbfgs_minimize(f, jnp.zeros(2), kw, settings, path,
        'toy_resume', log=lambda _: None, coordinate_scale=jnp.array([1.,1e-5]))
    np.testing.assert_allclose(resumed, kw['optimum'], atol=1e-12)
print('Passed: ill-conditioned optimum, original-gradient gate, original-coordinate checkpoint convention')
