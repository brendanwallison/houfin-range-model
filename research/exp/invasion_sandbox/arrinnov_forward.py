"""R24 forked scan: innovations on ARRIVALS' settlement success, gated by the ABSOLUTE Allee deficit.

Sandbox-only copy of the frozen step's age_forward.forward_sim_age_structured (asserted source edit). Just before
reproduction, the juvenile arrivals in each cell are multiplied by
    exp( FA['sigma'] * u * D ),   D = 1 - allee_mated_fraction(allee_gamma * N_local)
where N_local is the cell's breeding-season density (residents + arrivals; absolute, BBS density units -- the same
quantity and form the base's absolute mate-finding Allee factor uses, incl. any aggregation k). D is the share of
breeders that fail to find a mate: ~1 in an empty or sparse cell (the knife's edge an arriving trickle sits on), -> 0
once the cell is dense enough that mates are not limiting. So the correction acts where, and only while, arrivals
land below the Allee knee -- no N/K ratio anywhere. u is FA['grid'] (Ny, Nx) per cell, or FA['grid_t'] (T, Ny, Nx)
per cell-year. u = 0 reproduces the original forward exactly (identity check in the caller).
Settings are read at TRACE time from the module dict FA.
"""
import inspect, textwrap
import jax
import jax.numpy as jnp
from src.model import age_forward as af

FA = dict(active=False)

_EDITS = [
    ("        F_nest = None\n",
     "        if FA['active']:\n"
     "            _D = 1.0 - allee_mated_fraction(allee_gamma * jnp.maximum(N_a_post + juvenile_stayers + juvenile_arriving, 0.0))\n"
     "            _u = FA['grid_t'][t] if FA.get('grid_t') is not None else FA['grid']\n"
     "            juvenile_arriving = juvenile_arriving * jnp.exp(FA['sigma'] * _u * _D)\n"
     "        F_nest = None\n"),
]


def build():
    src = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    for old, new in _EDITS:
        if src.count(old) != 1:
            raise ValueError(f'arrinnov fork: anchor not found exactly once: {old!r}')
        src = src.replace(old, new)
    ns = dict(af.__dict__); ns.update(FA=FA, jax=jax, jnp=jnp)
    exec(compile(src, '<arrinnov_forward>', 'exec'), ns)
    return ns['forward_sim_age_structured']
