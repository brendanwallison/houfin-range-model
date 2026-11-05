"""Plan L A5 / S2 forked scan: DENSITY-DEPENDENT JUVENILE SURVIVAL (recruitment) alongside fecundity crowding.

Sandbox-only copy of the frozen step's age_forward.forward_sim_age_structured (asserted source edit). Just before
reproduction, this year's juvenile stayers and arrivers (whose survival the reproduction step applies / has applied) are
scaled by f = 1 / (1 + DS['cs'] * (N / K)^DS['p']), N the local post-dispersal density. The caller re-solves the fecundity
crowding term c with Sj f_K (f_K = 1 / (1 + cs)) so the equilibrium stays at K (build(); see wrapped()). cs = 0 reproduces
the original forward exactly. Disease mortality (target survival) is applied as before, BEFORE this step, so disease deaths
lower N and thereby RELAX the density-dependent juvenile mortality (compensation) -- the mechanism the fecundity-only
crowding lacks.
"""
import inspect, textwrap
import jax
import jax.numpy as jnp
from src.model import age_forward as af

DS = dict(active=False, cs=0., p=1., resolve=True)

_EDITS = [
    ("        F_nest = None\n",
     "        if DS['active']:\n"
     "            _rel = jnp.maximum(N_a_post + juvenile_stayers + juvenile_arriving, 0.0) / jnp.maximum(K_g, 1e-12)\n"
     "            _f = 1.0 / (1.0 + DS['cs'] * jnp.power(_rel + 1e-12, DS['p']))\n"
     "            juvenile_stayers = juvenile_stayers * _f\n"
     "            juvenile_arriving = juvenile_arriving * _f\n"
     "        F_nest = None\n"),
]


def build():
    src = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    for old, new in _EDITS:
        if src.count(old) != 1:
            raise ValueError(f'ddsurv fork: anchor not found exactly once: {old!r}')
        src = src.replace(old, new)
    ns = dict(af.__dict__); ns.update(DS=DS, jax=jax, jnp=jnp)
    exec(compile(src, '<ddsurv_forward>', 'exec'), ns)
    fork = ns['forward_sim_age_structured']
    from harness import c_of
    sig = inspect.signature(fork)

    def wrapped(*aa, **kk):   # re-solve c with the density-dependent juvenile survival at N = K
        b = sig.bind(*aa, **kk); b.apply_defaults()
        if DS['active'] and DS.get('resolve', True):   # resolve=False: EMERGENT equilibrium (K is a resource scale)
            fK = 1.0 / (1.0 + DS['cs'])
            b.arguments['c_flat'] = c_of(b.arguments['Fmax_flat'], b.arguments['Sa_flat'], b.arguments['Sj_flat'] * fK,
                                         {'yearling_breeding': b.arguments.get('yearling_breeding')})
        if DS['active'] and DS.get('dd_only'):   # ALL density dependence through juvenile survival: no fecundity crowding
            b.arguments['c_flat'] = jnp.zeros_like(jnp.asarray(b.arguments['c_flat']))
        return fork(*b.args, **b.kwargs)
    wrapped.__signature__ = sig
    return wrapped
