"""Full observed-data smoke with nonzero private fields or cell-year innovations."""
from pathlib import Path
script=Path(__file__).with_name('smoke_overflow_early.py').read_text()
script=script.replace("keys=['alpha_a','alpha_j','alpha_f','alpha_k','log_inv_total']", '''
_family=cfg['_minor_flexibility']['family']
_private=None
if _family.startswith('kprivate'):
    _private='capacity_private';_p=.2*jnp.cos(jnp.arange(lat['manifold_factor'].shape[0]))[:,None]
elif _family=='allrho99':
    _private='manifold_idio';_p=.2*jnp.cos(jnp.arange(lat['manifold_factor'].shape[0])[:,None]+jnp.arange(4)[None,:])
if _private:lat[_private]=_p
_pn=cfg['population_model'].get('process_noise',{})
if _pn.get('enabled'):
    _rows=data['pn_time_indices'].shape[0] if _family=='cell_iid' else data['pn_active'].shape[0]
    _cols=data['land_rows'].shape[0] if _family=='cell_iid' else data['pn_S'].shape[1]
    _pattern=.35*jnp.sin(jnp.arange(_rows)[:,None]/5.+jnp.arange(_cols)[None,:])
    lat['process_noise_u']=_pattern
keys=['alpha_a','alpha_j','alpha_f','alpha_k','log_inv_total']
if _private:lat['_private_probe']=jnp.asarray(1.);keys.append('_private_probe')
if _pn.get('enabled'):lat['_process_probe']=jnp.asarray(1.);keys.append('_process_probe')
''')
script=script.replace('point=dict(lat);point.update(zip(keys,v))', '''point=dict(lat);point.update(zip(keys,v))
    if _private:point[_private]=_p*point.pop('_private_probe')
    if _pn.get('enabled'):point['process_noise_u']=_pattern*point.pop('_process_probe')''')
exec(compile(script,str(Path(__file__).with_name('smoke_overflow_early.py')),'exec'))
