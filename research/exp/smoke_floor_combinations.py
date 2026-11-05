"""Use full-model conservation smoke with explicit nonzero new latent effects."""
from pathlib import Path
script=Path(__file__).with_name('smoke_overflow_early.py').read_text()
script=script.replace("keys=['alpha_a','alpha_j','alpha_f','alpha_k','log_inv_total']", """
_pop=cfg['population_model']
if _pop.get('native_departure',{}).get('enabled'):
    lat['native_departure_r']=jnp.asarray(.5)
_pn_active=_pop.get('process_noise',{}).get('enabled',False)
if _pn_active:
    _tt=jnp.arange(data['pn_active'].shape[0])[:,None]
    _bb=jnp.arange(data['pn_S'].shape[-1])[None,:]
    _pattern=.35*jnp.sin(_tt/5.+_bb)
    lat['process_noise_u']=_pattern
keys=['alpha_a','alpha_j','alpha_f','alpha_k','log_inv_total']
if 'native_departure_r' in lat:keys.append('native_departure_r')
if _pn_active:
    lat['_process_probe']=jnp.asarray(1.)
    keys.append('_process_probe')
""")
script=script.replace('point=dict(lat);point.update(zip(keys,v))',"point=dict(lat);point.update(zip(keys,v))\n    if _pn_active:point['process_noise_u']=_pattern*point.pop('_process_probe')")
exec(compile(script,str(Path(__file__).with_name('smoke_overflow_early.py')),'exec'))
