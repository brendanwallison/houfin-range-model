"""Explicit adapter for the frozen BBS breeding-readout convention.

This reproduces the existing observation mapping, not an independent validation
of its biological census interpretation. No BBS observer offsets transfer to
eBird. Source-specific young availability remains an explicit choice.
"""
import numpy as np


def breeding_grid(total,young,fledgling_weight,pop_scalar=20.,initial=None,floor=1e-6):
    total=np.asarray(total);young=np.asarray(young)
    if total.shape!=young.shape or total.ndim<2:raise ValueError('matching time/spatial arrays required')
    if not 0<=fledgling_weight<=1 or pop_scalar<=0:raise ValueError('invalid readout weight/scale')
    prior=np.empty_like(total,dtype=np.result_type(total.dtype,np.float64))
    prior[1:]=total[:-1]
    # Initial-year counts require the actual seeded initial population, not the
    # post-step state. Modern eBird and BBS observations do not use that year.
    prior[0]=np.nan if initial is None else initial
    return np.maximum((prior+fledgling_weight*young)*pop_scalar,floor)
