"""Check a scan-slice optimization barrier without changing projection arithmetic."""
import json
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


def projection(beta, z, barrier):
    @jax.checkpoint
    def step(carry, t):
        x = jnp.take(z, t, axis=0)
        if barrier:
            x = jax.lax.optimization_barrier(x)
        y = jnp.einsum('nkm,m->nk', x, beta)
        return carry, jnp.sum(jnp.sin(y))
    return jax.lax.scan(step, None, jnp.arange(z.shape[0]))[1].sum()


rng = np.random.default_rng(12)
z = jnp.asarray(rng.normal(size=(8, 19, 6, 7)), dtype=jnp.float32)
b = jnp.asarray(rng.normal(size=7), dtype=jnp.float64)
results = []
for barrier in [False, True]:
    f = jax.jit(jax.value_and_grad(lambda beta, data: projection(beta, data, barrier)))
    v, g = f(b, z)
    h = jax.jit(jax.hessian(lambda beta: projection(beta, z, barrier)))(b)
    compiled = f.lower(b, z).compile()
    analysis = compiled.memory_analysis()
    results.append((np.asarray(v), np.asarray(g), np.asarray(h)))
    print(json.dumps({'barrier': barrier, 'value': float(v),
                      'gradient_norm': float(jnp.linalg.norm(g)),
                      'argument_bytes': analysis.argument_size_in_bytes,
                      'temporary_bytes': analysis.temp_size_in_bytes}))
for x, y in zip(*results):
    np.testing.assert_allclose(x, y, rtol=2e-13, atol=2e-12)
print('Value, gradient and Hessian agree; no derivative through the data barrier is required.')
