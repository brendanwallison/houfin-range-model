"""B7 forked scan: a time-varying, per-cell INCREASE in departure (e.g. eastern partial-migration evolution).

Sandbox-only copies of the frozen step's age_forward.dispersal_step_age_structured and forward_sim_age_structured,
made by source edits (asserted). In the dispersal step the departure multiplier becomes a capped boost,
    p' = 1 - (1 - p)^s        (s = 1 leaves p unchanged; s > 1 raises departure, never above 1),
and in the forward step s is read from a (T, Ny, Nx) grid DEP['grid'][t] (when DEP['active']) instead of the static
departure_scale. Both ages are affected (as the original departure_scale). Settings are read at TRACE time.
"""
import inspect, textwrap
import jax
import jax.numpy as jnp
from src.model import age_forward as af

DEP = dict(active=False)

_DISP_EDITS = [
    ("        p_total, p_adult = p_total * departure_scale, p_adult * departure_scale\n",
     "        p_total, p_adult = (1.0 - jnp.power(jnp.clip(1.0 - p_total, 1e-9, 1.0), departure_scale),\n"
     "                            1.0 - jnp.power(jnp.clip(1.0 - p_adult, 1e-9, 1.0), departure_scale))\n"
     "        _fl = DEP.get('floor', 0.0) if DEP['active'] else 0.0   # B9: departure floor (birds leave near-empty cells too)\n"
     "        p_total, p_adult = _fl + (1.0 - _fl) * p_total, _fl + (1.0 - _fl) * p_adult\n"),
]
_FWD_EDITS = [
    ("            departure_scale=dep_scale_g,\n",
     "            departure_scale=(DEP['grid'][t] if DEP['active'] else dep_scale_g),\n"),
]


def _apply(src, edits, what):
    for old, new in edits:
        if src.count(old) != 1:
            raise ValueError(f'dep fork ({what}): anchor not found exactly once: {old!r}')
        src = src.replace(old, new)
    return src


def build(extra_fwd_edits=(), extra_ns=None):
    """extra_fwd_edits / extra_ns: another fork's forward edits and settings dict (e.g. lowdens_forward._EDITS and
    dict(LD=lowdens_forward.LD)), so both forks run in one forward."""
    ns = dict(af.__dict__); ns.update(DEP=DEP, jax=jax, jnp=jnp, **(extra_ns or {}))
    dsrc = _apply(textwrap.dedent(inspect.getsource(af.dispersal_step_age_structured)), _DISP_EDITS, 'dispersal')
    exec(compile(dsrc, '<dep_dispersal>', 'exec'), ns)            # ns now holds the capped-boost dispersal step
    fsrc = _apply(textwrap.dedent(inspect.getsource(af.forward_sim_age_structured)),
                  list(_FWD_EDITS) + list(extra_fwd_edits), 'forward')
    exec(compile(fsrc, '<dep_forward>', 'exec'), ns)              # the forward resolves dispersal_step from ns
    return ns['forward_sim_age_structured']
