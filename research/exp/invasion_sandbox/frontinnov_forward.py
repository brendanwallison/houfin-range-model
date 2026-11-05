"""R23 forked scan: ONE innovation per cell, active only while that cell is on the wavefront.

Sandbox-only copy of the frozen step's age_forward.forward_sim_age_structured (asserted source edits). It carries a
soft colonisation clock E (years occupied; as clock_forward.py, in the psi slot when occupancy is off) and, just before
reproduction, multiplies that year's fecundity by
    exp( FI['sigma'] * u_cell * exp(-E / FI['tau']) * exp(-(N/K) / FI['rho']) ),
(FI['gate'] = 'allee' replaces the N/K factor by the absolute Allee mate-finding deficit 1 - mated_fraction(gamma N),
the right low-density measure on absolute-Allee bases; R23b),
so each cell's single innovation u_cell acts only while the cell is newly colonised (E small) AND sparse (N/K small):
an establishment correction that cannot touch settled abundance and cannot be re-applied to a location year after year
(the prior charges it once per cell). u = 0 reproduces the original forward exactly (identity check in the caller).
Settings are read at TRACE time from the module dict FI (grid = (Ny, Nx) per-cell u).
"""
import inspect, textwrap
import jax
import jax.numpy as jnp
from src.model import age_forward as af

FI = dict(active=False)

_EDITS = [
    ("        N_a, N_j, psi = pools\n",
     "        N_a, N_j, psi = pools\n"
     "        _fi = FI['active'] and not occ\n"
     "        E_clk = psi\n"),
    ("        if strata_u is None:\n",
     "        if _fi:\n"
     "            _rel = jnp.maximum(N_a_post + juvenile_stayers + juvenile_arriving, 0.0) / jnp.maximum(K_g, 1e-12)\n"
     "            _gate = ((1.0 - allee_mated_fraction(allee_gamma * jnp.maximum(N_a_post + juvenile_stayers + juvenile_arriving, 0.0)))\n"
     "                     if FI.get('gate') == 'allee' else (1.0 if FI.get('gate') == 'none' else jnp.exp(-_rel / FI['rho'])))\n"
     "            Fmax_g = Fmax_g * jnp.exp(FI['sigma'] * FI['grid'] * jnp.exp(-E_clk / FI['tau']) * _gate)\n"
     "        if strata_u is None:\n"),
    ("        N_total_new = N_a_new + N_j_new\n",
     "        N_total_new = N_a_new + N_j_new\n"
     "        if _fi:\n"
     "            psi = (E_clk + jax.nn.sigmoid((jnp.log(N_total_new + 1e-30) - FI['log_thr']) / FI['scale'])) * land_mask\n"),
]


def build():
    src = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    for old, new in _EDITS:
        if src.count(old) != 1:
            raise ValueError(f'frontinnov fork: anchor not found exactly once: {old!r}')
        src = src.replace(old, new)
    ns = dict(af.__dict__); ns.update(FI=FI, jax=jax, jnp=jnp)
    exec(compile(src, '<frontinnov_forward>', 'exec'), ns)
    return ns['forward_sim_age_structured']
