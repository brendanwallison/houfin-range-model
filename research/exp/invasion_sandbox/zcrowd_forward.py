"""Z-readout forked scan: habitat-dependent crowding CURVATURE theta(cell, t) (middle ground built on Z).

Sandbox-only copy of the frozen step's age_forward.forward_sim_age_structured, made by source edits (asserted). Just
before the (non-subcell) reproduction call, the step stores that year's theta grid in THETA_CUR; a patched
age_forward.crowding_argument then returns (N/K)^theta instead of the fixed power. At N = K the argument is 1 for any
theta, so K stays the equilibrium (c unchanged). theta > 1: little crowding until near K (territorial / ceiling-
like); theta < 1: strongly compensatory at low density.
Settings are read at TRACE time from the module dict THETA (set them inside the traced objective). install() patches
age_forward.crowding_argument (resolved by name inside reproduction_age_structured); restore() undoes it.
"""
import inspect, textwrap
import jax
import jax.numpy as jnp
from src.model import age_forward as af

THETA = dict(active=False)
THETA_CUR = [None]
_ORIG = af.crowding_argument

_EDITS = [
    ("        if strata_u is None:\n",
     "        THETA_CUR[0] = THETA['grid'][t] if THETA['active'] else None\n"
     "        if strata_u is None:\n"),
]


def _crowding_argument(ratio, power=None):
    th = THETA_CUR[0] if THETA['active'] else None
    if th is None or power is not None:
        return _ORIG(ratio, power)
    return jnp.exp(th * jnp.log(jnp.maximum(ratio, 0.0) + 1e-30))


def install():
    af.crowding_argument = _crowding_argument


def restore():
    af.crowding_argument = _ORIG; THETA_CUR[0] = None


def build():
    src = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    for old, new in _EDITS:
        if src.count(old) != 1:
            raise ValueError(f'zcrowd fork: anchor not found exactly once: {old!r}')
        src = src.replace(old, new)
    ns = dict(af.__dict__); ns.update(THETA=THETA, THETA_CUR=THETA_CUR, jax=jax, jnp=jnp)
    exec(compile(src, '<zcrowd_forward>', 'exec'), ns)
    return ns['forward_sim_age_structured']
