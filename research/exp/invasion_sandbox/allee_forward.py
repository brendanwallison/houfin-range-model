"""R13 forked scan: a per-cell, per-year multiplier on the mate-finding (Allee) encounter rate allee_gamma.

Sandbox-only copy of the frozen step's age_forward.forward_sim_age_structured, made by source edits (asserted, so a
code mismatch fails loudly). In the (non-subcell) reproduction call, allee_gamma becomes allee_gamma * G[t], where G
is a (T, Ny, Nx) grid. Since the mated fraction is a function of gamma * N (or gamma * N/K), G > 1 lowers the mate-
finding threshold n50 = ln 2 / gamma by the factor G: easier mating at low density (e.g. birds concentrated at
feeders / towns), leaving the settled level (K) and the rates at density untouched.
Settings are read at TRACE time from the module dict ALLEE (set them inside the traced objective).
"""
import inspect, textwrap
import jax
import jax.numpy as jnp
from src.model import age_forward as af

ALLEE = dict(active=False)

_EDITS = [
    ("                Sa_g, Sj_g, Fmax_g, K_g, c_g, allee_gamma, allee_off=allee_off,\n",
     "                Sa_g, Sj_g, Fmax_g, K_g, c_g,\n"
     "                (allee_gamma * ALLEE['grid'][t]) if ALLEE['active'] else allee_gamma, allee_off=allee_off,\n"),
]


def build():
    src = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    for old, new in _EDITS:
        if src.count(old) != 1:
            raise ValueError(f'allee fork: anchor not found exactly once: {old!r}')
        src = src.replace(old, new)
    ns = dict(af.__dict__); ns.update(ALLEE=ALLEE, jax=jax, jnp=jnp)
    exec(compile(src, '<allee_forward>', 'exec'), ns)
    return ns['forward_sim_age_structured']
