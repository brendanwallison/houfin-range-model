"""Independent spatial impulse check; optionally against the frozen live model.

HOUFIN_BBL_FORWARD points to the age_forward.py under audit. No data files or
production inputs are needed, and this test changes no model configuration.
"""
import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from research.exp.bbl_survival import marked_transition


def test_marked_juvenile_transition_matches_forward_fft_impulses():
    jax.config.update("jax_enable_x64",True)
    path=Path(os.environ.get("HOUFIN_BBL_FORWARD","src/model/age_forward.py"))
    spec=importlib.util.spec_from_file_location("bbl_audited_forward",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    land=np.array([[True,True,False],[True,True,True],[False,True,True]])
    cells=np.argwhere(land); n=len(cells)
    kernel=np.zeros((5,5)); kernel[0,0]=.3; kernel[0,1]=.3; kernel[0,-1]=.2; kernel[1,0]=.2
    # Build the destination transition by explicit finite-grid enumeration,
    # independently of the FFT/cross-correlation implementations.
    raw=np.zeros((n,n))
    for i,(y,x) in enumerate(cells):
        for j,(v,u) in enumerate(cells):
            raw[i,j]=kernel[(v-y)%5,(u-x)%5]
    edge=raw.sum(1)
    eps=1e-6
    move=raw/(edge[:,None]+eps)
    edge_grid=np.ones(land.shape); edge_grid[land]=edge
    p=np.linspace(.05,.95,n); sj=np.linspace(.2,.7,n); q=np.linspace(.3,.8,n)
    q_grid=np.zeros(land.shape); q_grid[land]=q
    fft=jnp.asarray(np.fft.fft2(kernel)[None])
    def arriving(source):
        return np.asarray(module.juvenile_dispersal_vectorized(
            jnp.asarray(source),fft,jnp.asarray(q_grid[None]),jnp.asarray(edge_grid[None]),eps))
    zero=arriving(np.zeros(land.shape))
    actual=[]
    for i,(y,x) in enumerate(cells):
        impulse=np.zeros(land.shape); impulse[y,x]=p[i]
        row=(arriving(impulse)-zero)[land]
        row[i]+=(1-p[i])*sj[i]
        actual.append(row)
    np.testing.assert_allclose(actual,marked_transition(p,sj,move,q),atol=1e-12,rtol=1e-10)
