"""R6b forked scan: a per-cell LOCAL COLONISATION CLOCK carried inside the forward lax.scan.

Sandbox-only copy of the frozen step's age_forward.forward_sim_age_structured, made by source edits (asserted, so a
code mismatch fails loudly). Without occupancy (occupancy_b None) the scan carries an unused psi slot (zeros); the
fork stores the clock E there:
  E_t+1 = E_t + sigmoid((log N_t - log thr) / scale)      (soft "years occupied" since colonisation)
and multiplies that year's Fmax (East cells only) by g(E) = interp(E, knots, values) before dispersal, recomputing the
crowding brake c from the edited Fmax as harness.c_of does (so K stays the equilibrium). Everything is differentiable,
so the clock multiplier is fitted with exact gradients in one pass -- no fixed-point iteration (R6's failed: the
colonisation years kept jumping between rounds).
Settings are read at TRACE time from the module dict CLOCK (set it inside the traced objective).
"""
import inspect, textwrap
import jax
import jax.numpy as jnp
from src.model import age_forward as af

CLOCK = dict(active=False)

_EDITS = [
    ("        N_a, N_j, psi = pools\n",
     "        N_a, N_j, psi = pools\n"
     "        _clk = CLOCK['active'] and not occ\n"
     "        E_clk = psi\n"),
    ("        # 3. Dispersal\n",
     "        if _clk:\n"
     "            _g = jnp.interp(E_clk, CLOCK['knots'], CLOCK['values'])\n"
     "            Fmax_g = Fmax_g * jnp.where(CLOCK['mask'], _g, 1.0)\n"
     "            _num = Fmax_g * Sj_g if yearling_breeding else Fmax_g * Sa_g * Sj_g\n"
     "            c_g = jnp.maximum(_num / (1.0 - Sa_g + 1e-6) - 1.0, 0.0) * land_mask\n"
     "        # 3. Dispersal\n"),
    ("        N_total_new = N_a_new + N_j_new\n",
     "        N_total_new = N_a_new + N_j_new\n"
     "        if _clk:\n"
     "            psi = (E_clk + jax.nn.sigmoid((jnp.log(N_total_new + 1e-30) - CLOCK['log_thr'])\n"
     "                                          / CLOCK['scale'])) * land_mask\n"),
]


def build():
    src = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    for old, new in _EDITS:
        if src.count(old) != 1:
            raise ValueError(f'clock fork: anchor not found exactly once: {old!r}')
        src = src.replace(old, new)
    ns = dict(af.__dict__); ns.update(CLOCK=CLOCK, jax=jax, jnp=jnp)
    exec(compile(src, '<clock_forward>', 'exec'), ns)
    return ns['forward_sim_age_structured']
