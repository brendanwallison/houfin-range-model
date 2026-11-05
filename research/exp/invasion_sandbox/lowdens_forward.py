"""R14 forked scan: a flexible LOW-DENSITY growth boost that fades as a cell fills.

Sandbox-only copy of the frozen step's age_forward.forward_sim_age_structured, made by source edits (asserted). Just
before the (non-subcell) reproduction call, that year's fecundity becomes
    Fmax_g * (1 + (G[t] - 1) * exp(-(N / K) / rho)),
with N the post-dispersal local density. So at N -> 0 fecundity is Fmax G (establishment / low-density growth), and
by N about a few rho K it is back to Fmax; the crowding brake c is NOT recomputed, so the settled level stays near K.
G is a (T, Ny, Nx) grid (any covariate or time shape); rho > 0 sets how fast the boost fades.
Settings are read at TRACE time from the module dict LD (set them inside the traced objective).
"""
import inspect, textwrap
import jax
import jax.numpy as jnp
from src.model import age_forward as af

LD = dict(active=False)

_EDITS = [
    ("        if strata_u is None:\n",
     "        if LD['active']:\n"
     # clamp: post-dispersal FFT ringing leaves tiny negative densities where K ~ 0 (off land), which made
     # exp(-rel / rho) overflow and (G - 1) * inf = NaN even at G = 1 (first R14 attempt)
     "            _rel = jnp.maximum(N_a_post + juvenile_stayers + juvenile_arriving, 0.0) / jnp.maximum(K_g, 1e-12)\n"
     # LD['gate'] = 'allee' (R37, J19 fix): fade with the ABSOLUTE Allee mate-finding deficit instead of N/K
     "            _fade = ((1.0 - allee_mated_fraction(allee_gamma * jnp.maximum(N_a_post + juvenile_stayers + juvenile_arriving, 0.0)))\n"
     "                     if LD.get('gate') == 'allee' else jnp.exp(-_rel / LD['rho']))\n"
     "            Fmax_g = Fmax_g * (1.0 + (LD['grid'][t] - 1.0) * _fade)\n"
     "        if strata_u is None:\n"),
]


def build():
    src = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    for old, new in _EDITS:
        if src.count(old) != 1:
            raise ValueError(f'lowdens fork: anchor not found exactly once: {old!r}')
        src = src.replace(old, new)
    ns = dict(af.__dict__); ns.update(LD=LD, jax=jax, jnp=jnp)
    exec(compile(src, '<lowdens_forward>', 'exec'), ns)
    return ns['forward_sim_age_structured']
