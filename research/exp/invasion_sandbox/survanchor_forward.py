"""C2 forked scan: low-density SURVIVAL anchored high (survival-side density dependence).

Sandbox-only copy of the frozen step's age_forward.forward_sim_age_structured (asserted source edit). Just before the
(non-subcell) reproduction call, adult survival becomes
    Sa' = hi' - (hi' - Sa) * min(N/K, 1)^q,     hi' = max(SA['hi'], Sa)
so a near-empty cell has survival hi' (habitat-independent), falling to the habitat field's Sa as the cell fills
(N = K gives Sa exactly, so K stays the equilibrium with the original c). Juvenile survival keeps its ratio to adult
survival. The field can then no longer brake a FRONT through survival; survival only sets settled abundance.
Settings are read at TRACE time from the module dict SA.
"""
import inspect, textwrap
import jax
import jax.numpy as jnp
from src.model import age_forward as af

SA = dict(active=False, hi=.6, q=1.)

_EDITS = [
    ("        if strata_u is None:\n",
     "        if SA['active']:\n"
     "            _r = jnp.clip(jnp.maximum(N_a_post + juvenile_stayers + juvenile_arriving, 0.0)\n"
     "                          / jnp.maximum(K_g, 1e-12), 0.0, 1.0) ** SA['q']\n"
     "            _hi = jnp.maximum(SA['hi'], Sa_g)\n"
     "            _sa = _hi - (_hi - Sa_g) * _r\n"
     "            Sj_g = Sj_g * _sa / jnp.maximum(Sa_g, 1e-6)\n"
     "            Sa_g = _sa\n"
     "        if strata_u is None:\n"),
]


def build():
    src = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    for old, new in _EDITS:
        if src.count(old) != 1:
            raise ValueError(f'survanchor fork: anchor not found exactly once: {old!r}')
        src = src.replace(old, new)
    ns = dict(af.__dict__); ns.update(SA=SA, jax=jax, jnp=jnp)
    exec(compile(src, '<survanchor_forward>', 'exec'), ns)
    return ns['forward_sim_age_structured']
