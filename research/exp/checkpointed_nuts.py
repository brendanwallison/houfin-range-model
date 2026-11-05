"""Checkpoint NumPyro NUTS transitions without resetting warmup adaptation."""
import json,os,pickle,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np


def atomic_pickle(path,value):
    tmp=path.with_suffix('.tmp.pkl')
    with tmp.open('wb') as f:pickle.dump(value,f)
    tmp.replace(path)


def read_pickle(path):
    with path.open('rb') as f:return pickle.load(f)


def run_checkpointed(kernel,initial,kwargs,rng,warmup,samples,out,fingerprint,
                     postprocess,chunk=4,session_steps=None):
    """Use the public init/sample kernel API; HMCState.i keeps the full schedule.

    Preserve the complete state including RNG and adaptation through every chunk.
    A saved chunk precedes state advancement; repeating an interrupted chunk does
    not duplicate draws. No shorter-warmup sampler is substituted on resume.
    """
    out=Path(out);(out/'chunks').mkdir(parents=True,exist_ok=True)
    init_start=time.time()
    state=jax.block_until_ready(kernel.init(rng,num_warmup=warmup,init_params=initial,model_args=(),model_kwargs=kwargs))
    init_seconds=time.time()-init_start
    cp=out/'checkpoint.pkl';elapsed=init_seconds
    if cp.exists():
        old=read_pickle(cp)
        if old['fingerprint']!=fingerprint:raise ValueError('incompatible sampler checkpoint')
        state=jax.tree.map(jnp.asarray,old['state']);elapsed=old['elapsed_seconds']+init_seconds
    total=warmup+samples;start_iteration=int(state.i);session_start=time.time()
    def advance(s,kw):
        def step(st,_):
            nxt=kernel.sample(st,model_args=(),model_kwargs=kw)
            return nxt,nxt
        return jax.lax.scan(step,s,None,length=chunk)
    advance=jax.jit(advance)
    while int(state.i)<total:
        before=int(state.i)
        if total-before<chunk:raise ValueError('transition count must be divisible by chunk size')
        t=time.time();last,batch=jax.block_until_ready(advance(state,kwargs));seconds=time.time()-t
        batch=jax.device_get(batch);keep=np.asarray(batch.i)>warmup
        extra={name:np.asarray(getattr(batch,name))[keep] for name in ['num_steps','accept_prob','energy','potential_energy','diverging']}
        extra['adapt_state.step_size']=np.asarray(batch.adapt_state.step_size)[keep]
        part={'start_iteration':before,'end_iteration':int(last.i),'seconds':seconds,
              'leapfrog_steps_all':int(np.asarray(batch.num_steps).sum()),
              'compile_included':before==start_iteration,
              'z':{k:np.asarray(v)[keep] for k,v in batch.z.items()},'extra':extra}
        atomic_pickle(out/'chunks'/f'chunk_{before:06d}.pkl',part)
        elapsed+=seconds
        atomic_pickle(cp,{'fingerprint':fingerprint,'state':jax.device_get(last),'elapsed_seconds':elapsed})
        state=last
        parts=[read_pickle(p) for p in sorted((out/'chunks').glob('chunk_*.pkl'))]
        timings=[p['seconds']/p['leapfrog_steps_all'] for p in parts if not p['compile_included'] and p['leapfrog_steps_all']>0]
        progress={'completed_iterations':int(state.i),'planned_iterations':total,'warmup_iterations':warmup,
            'retained_draws':max(int(state.i)-warmup,0),'elapsed_transition_seconds':elapsed,
            'seconds_per_leapfrog_median':float(np.median(timings)) if timings else None,
            'checkpoint_chunk':chunk,'adaptation_schedule_preserved':True}
        tmp=out/'PROGRESS.tmp.json';tmp.write_text(json.dumps(progress,indent=2)+'\n');tmp.replace(out/'PROGRESS.json')
        print('NUTS CHECKPOINT',progress,flush=True)
        if session_steps is not None and int(state.i)-start_iteration>=session_steps and int(state.i)<total:return None
    parts=[read_pickle(p) for p in sorted((out/'chunks').glob('chunk_*.pkl'))]
    z={k:np.concatenate([p['z'][k] for p in parts]) for k in parts[0]['z']}
    extra={k:np.concatenate([p['extra'][k] for p in parts]) for k in parts[0]['extra']}
    processed=jax.jit(lambda z,kw:jax.vmap(lambda q:postprocess(q,kw))(z))(jax.tree.map(jnp.asarray,z),kwargs)
    return jax.tree.map(np.asarray,jax.block_until_ready(processed)),extra,elapsed
