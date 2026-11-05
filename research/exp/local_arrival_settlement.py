"""Mass-conserving partial local placement of arrivals; residents are untouched."""
import jax.numpy as jnp
from jax.scipy.signal import convolve2d


def shift(field,dy,dx):
    padded=jnp.pad(field,((1,1),(1,1)))
    return padded[1-dy:1-dy+field.shape[0],1-dx:1-dx+field.shape[1]]


def denominator(capacity):
    return convolve2d(jnp.maximum(capacity,0.),jnp.ones((3,3),capacity.dtype),mode='same')


def redistribute(arrivals,capacity,fraction):
    w=jnp.maximum(capacity,0.);land=w>0
    arrivals=jnp.where(land,arrivals,0.)
    if fraction==0.:return arrivals
    norm=denominator(w)
    source=arrivals/jnp.where(norm>0,norm,1.)
    return (1-fraction)*arrivals+fraction*w*convolve2d(source,jnp.ones((3,3),source.dtype),mode='same')


def offset_fractions(capacity,fraction):
    w=jnp.maximum(capacity,0.);norm=denominator(w)
    return [(dy,dx,jnp.where(w>0,fraction*shift(w,-dy,-dx)/jnp.where(norm>0,norm,1.)
              +((1-fraction) if dy==0 and dx==0 else 0.),0.))
            for dy in (-1,0,1) for dx in (-1,0,1)]


def composite_diagonal(kernel_fft,q,capacity,fraction):
    """Diagonal of local-placement after convolution and destination journey Q.

    Excludes departure and origin edge normalization. Works on one or stacked
    kernels/Q grids; the last two dimensions are spatial.
    """
    kernel=jnp.fft.ifft2(kernel_fft,axes=(-2,-1)).real
    w=jnp.maximum(capacity,0.);norm=denominator(w)
    result=(1-fraction)*kernel[...,0,0][...,None,None]*q
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            q_shift=jnp.stack([shift(v,dy,dx) for v in (q*(w>0)/jnp.where(norm>0,norm,1.)).reshape((-1,)+w.shape)])
            q_shift=q_shift.reshape(q.shape)
            result=result+fraction*w*kernel[...,-dy % kernel.shape[-2],-dx % kernel.shape[-1]][...,None,None]*q_shift
    return result*(w>0)
