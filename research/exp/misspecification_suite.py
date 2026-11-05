"""Isolated weak-constraint/state-versus-establishment experiments (TACC clones).

This module is installed only in immutable experiment copies. It leaves the
production forward interface and output tuple unchanged. Zero innovations are
the exact identity, except genuine shooting nodes, whose continuity is soft.
"""
import inspect
import textwrap
from pathlib import Path
import numpy as np


def bounded_log(raw, sigma, xp=np, limit=2.):
    return limit * xp.tanh(float(sigma) * raw / limit)


def ar1(raw, rho, xp=np):
    """Stationary unit-marginal-variance AR(1), including the first year."""
    out = [raw[0]]
    for t in range(1, raw.shape[0]):
        out.append(rho * out[-1] + (1-rho*rho)**.5 * raw[t])
    return xp.stack(out)


def build_forward(af, mm):
    source = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    def edit(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('ambiguous misspecification fork: '+repr(old))
        source = source.replace(old, new)
    edit('        N_a, N_j, psi = pools\n', '''        N_a, N_j, psi = pools
        _mm_node_delta = jnp.zeros_like(N_a)
        if MM.get('nodes') is not None:
            _ix = MM['node_index'][t]
            _na = jnp.exp(MM['nodes'][jnp.maximum(_ix, 0), 0])
            _nj = jnp.exp(MM['nodes'][jnp.maximum(_ix, 0), 1])
            _ag = zero_grid.at[land_rows, land_cols].set(_na)
            _jg = zero_grid.at[land_rows, land_cols].set(_nj)
            _mm_node_delta = jnp.where(_ix >= 0, _ag+_jg-N_a-N_j, 0.)
            N_a = jnp.where(_ix >= 0, _ag, N_a)
            N_j = jnp.where(_ix >= 0, _jg, N_j)
        _mm_input_a, _mm_input_j = N_a, N_j
        _mm_logf = jnp.zeros_like(N_a)
        _mm_arrival = jnp.ones_like(N_a)
        _mm_eps = jnp.zeros_like(N_a)
''')
    edit('        F_nest = None\n', '''        _mm_n = jnp.maximum(N_a_post+juvenile_stayers+juvenile_arriving, 0.)
        _mm_mated = allee_mated_fraction(allee_gamma*_mm_n)
        if MM.get('field') is not None and MM['kind'] in ('establishment', 'arrival'):
            _uf = zero_grid.at[land_rows, land_cols].set(MM['field'][t])
            _gate = 1.-_mm_mated
            if MM.get('clock', False):
                _gate = _gate*jnp.exp(-psi/10.)
            _corr = _uf*_gate
            if MM['kind']=='establishment':
                _mm_logf = _corr
                Fmax_g = Fmax_g*jnp.exp(_corr)
            else:
                _w = jnp.exp(_corr)*land_mask
                _num = jnp.sum(juvenile_arriving)
                _den = jnp.sum(juvenile_arriving*_w)
                _mm_arrival = jnp.where(_num>1e-30, _w*_num/jnp.maximum(_den,1e-30), 1.)
                juvenile_arriving = juvenile_arriving*_mm_arrival
        F_nest = None
''')
    edit('        N_total_new = N_a_new + N_j_new\n', '''        if MM.get('field') is not None and MM['kind']=='abundance':
            _uf = zero_grid.at[land_rows, land_cols].set(MM['field'][t])
            _m = allee_mated_fraction(allee_gamma*jnp.maximum(N_a_new+N_j_new,0.))
            _mm_eps = _uf*(_m*_m if MM.get('established_only',True) else 1.)
            N_a_new, N_j_new = N_a_new*jnp.exp(_mm_eps), N_j_new*jnp.exp(_mm_eps)
        if MM.get('clock',False):
            _n = jnp.maximum(N_a_new+N_j_new,0.)
            psi = (psi+_n/(_n+1./jnp.maximum(allee_gamma,1e-12)))*land_mask
        N_total_new = N_a_new + N_j_new
        _mm_diag = jnp.stack([_mm_input_a, _mm_input_j, _mm_eps,
            _mm_logf, _mm_arrival, _mm_node_delta, psi])[:,land_rows,land_cols]
''')
    edit('(N_total_new, N_a_new, N_j_new, psi, F_nest)',
         '(N_total_new, N_a_new, N_j_new, psi, F_nest, _mm_diag)')
    edit('(N_total_new, N_a_new, N_j_new, psi)\n',
         '(N_total_new, N_a_new, N_j_new, psi, _mm_diag)\n')
    edit('    total_densities, Na_densities, Nj_densities, psi_t = _outs[:4]',
         "    MM['diagnostics'] = _outs[-1]\n    total_densities, Na_densities, Nj_densities, psi_t = _outs[:4]")
    ns = dict(af.__dict__, MM=mm)
    exec(compile(source, '<misspecification_forward>', 'exec'), ns)
    return ns['forward_sim_age_structured']


def install(base_model, namespace, spec, asset_path):
    """Wrap the original NumPyro model; retain all original observation sources."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro import handlers
    from src.model import age_forward as af
    with np.load(asset_path) as z:
        assets = {k: np.asarray(z[k]) for k in z.files}
    mm = {}; fork = build_forward(af, mm)
    original_forward = namespace['forward_sim_age_structured']
    capture = []

    def corrected(*args, **kw):
        bound = inspect.signature(original_forward).bind(*args, **kw)
        bound.apply_defaults()
        if mm.get('disabled'):
            # Skeleton replay removes every discrepancy, including shooting nodes.
            out = original_forward(*args, **kw)
        else:
            out = fork(*args, **kw)
        capture.append((bound.arguments, out))
        return out
    corrected.__signature__ = inspect.signature(original_forward)

    def model(data, prior_scale=1.):
        mm.clear(); capture.clear(); mm.update(spec)
        mm['disabled'] = bool(namespace.get('_MM_DISABLE', False))
        kind = spec['kind']; T, N = assets['years'].size, assets['basis'].shape[0]
        active = jnp.asarray(assets['active'])
        if kind in ('abundance','establishment','arrival'):
            mode = spec['mode']
            if mode=='cell_static':
                u = numpyro.sample('mm_u', dist.Normal(0.,1.).expand([N]))
                raw = jnp.broadcast_to(u,(T,N))
            elif mode=='cell_blocks':
                u = numpyro.sample('mm_u',dist.Normal(0.,1.).expand([int(spec['n_blocks']),N]))
                raw = u[jnp.asarray(assets['block_index'])]
            else:
                u = numpyro.sample('mm_u',dist.Normal(0.,1.).expand([T,assets['basis'].shape[1]]))
                raw = ar1(u,float(spec.get('rho',0.)),jnp)@jnp.asarray(assets['basis']).T
            mm['field'] = bounded_log(raw,spec['sigma'],jnp)*active[:,None]
        elif kind=='shooting':
            center = jnp.asarray(assets['node_center'])
            # This broad proposal density is cancelled exactly below. Its center
            # supplies initialization only, never an empirical prior/second likelihood.
            proposal = dist.Normal(center,3.)
            mm['nodes'] = numpyro.sample('mm_nodes_log',proposal)
            mm['node_index'] = jnp.asarray(assets['node_index'])
        # Install the forward hook once below. External diagnostic capture or
        # counterfactual wrappers must remain able to wrap that same hook.
        tr = handlers.trace(base_model).get_trace(data=data,prior_scale=prior_scale)
        if len(capture)!=1:raise ValueError('unexpected multiple forward calls')
        x, states = capture[0]
        if not mm['disabled']:
            numpyro.deterministic('mm_diagnostics', mm['diagnostics'])
        if kind=='shooting':
            idx = jnp.asarray(assets['node_times'])
            previous_state = jnp.stack([states[1][idx-1][:,x['land_rows'],x['land_cols']],
                                        states[2][idx-1][:,x['land_rows'],x['land_cols']]],axis=1)
            target = jnp.log(jnp.maximum(previous_state,1e-20))
            logcont = dist.Normal(target,float(spec['sigma'])).log_prob(mm['nodes'])
            numpyro.factor('mm_node_continuity', jnp.sum(logcont-proposal.log_prob(mm['nodes'])))
            numpyro.deterministic('mm_node_log_residual',mm['nodes']-target)
        count_weight = float(spec.get('positive_count_weight',1.))
        if count_weight!=1.:
            # Exact NB2 hurdle decomposition: occurrence + conditional positive
            # count. Replace the count-part weight; do not add a second likelihood.
            mu = tr['expected_obs']['value']; conc = tr['obs_concentration']['value']
            nb = dist.NegativeBinomial2(mean=mu,concentration=conc)
            y = data['observed_results']; lp0 = nb.log_prob(jnp.zeros_like(y))
            occurrence = jnp.where(y>0,jnp.log(-jnp.expm1(jnp.minimum(lp0,-1e-15))),lp0)
            count = jnp.where(y>0,nb.log_prob(y)-occurrence,0.)
            weight = data.get('obs_weight',jnp.ones_like(y))
            real = data['obs_obsroute_idx']>=0
            numpyro.factor('mm_positive_count_reweight',
                (count_weight-1.)*jnp.sum(jnp.where(real,weight*count,0.)))
        # Expose the exact frozen ambient corrections for post-fit accounting.
        namespace['_MM_LAST'] = dict(args=x,states=states,diagnostics=mm.get('diagnostics'),spec=spec)
    model.__name__ = 'build_model_2d'
    namespace['forward_sim_age_structured'] = corrected
    return model


def audit_arguments(last):
    """Frozen ambient linearization; discrepancies remain separately recorded."""
    import jax.numpy as jnp
    x = dict(last['args']); d = last['diagnostics']
    if d is None:return x
    x['mm_input_adult'], x['mm_input_juvenile'] = d[:,0], d[:,1]
    x['process_noise'] = d[:,2]
    x['Fmax_flat'] = x['Fmax_flat']*jnp.exp(d[:,3])
    x['mm_arrival_weight'] = d[:,4]
    x['mm_node_adjustment'] = d[:,5]
    return x
