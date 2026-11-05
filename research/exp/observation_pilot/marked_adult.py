"""Full-exposure marked-adult step using the frozen movement primitive.

Validated scope: current plain model, no settlement attraction/subcell allocation.
Ambient density determines departure; marked density never determines crowding.
No reproduction and no juvenile survival/Q factor is applied to marked adults.
Partial-year, age-switch and reporting-location adapters remain separate work.
"""
import jax.numpy as jnp


def adult_step(alive,ambient_density,capacity,adult_survival,land_mask,
               movement_fn,movement_kwargs):
    if movement_kwargs.get('attraction') is not None:
        raise NotImplementedError('attraction requires ambient rather than marked settlement weights')
    if 'departure_density' in movement_kwargs:
        raise ValueError('ambient departure density is supplied by the adapter')
    # Reuse departure curves, edge corrections, FFT kernels and native-retention
    # scale from the actual movement implementation, rather than duplicate them.
    moved=movement_fn(alive,jnp.zeros_like(alive),capacity,
                      departure_density=ambient_density,**movement_kwargs)[0]
    moved=jnp.where(land_mask,moved,0.)
    next_alive=moved*adult_survival
    deaths=moved*(1-adult_survival)
    movement_loss=jnp.sum(alive)-jnp.sum(moved)
    return {'alive':next_alive,'deaths_by_destination':deaths,
            'movement_mass_loss':movement_loss,
            'mass_balance_error':jnp.sum(alive)-jnp.sum(next_alive)-jnp.sum(deaths)-movement_loss}
