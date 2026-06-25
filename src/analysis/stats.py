import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

def calculate_demographics(Sa_flat, Sj_flat, Fmax_flat, K_flat=None, pop_scalar=1.0):
    """
    Standardizes the calculation of demographic rates.
    Handles shapes: (Time, Space) or (Samples, Time, Space).
    """
    # R0 is the Biotic Potential (Replacement Rate)
    # R0 = (Fmax * Sj) / (1 - Sa)
    R0 = (Fmax_flat * Sj_flat) / (1.0 - Sa_flat + 1e-6)
    
    stats = {
        "Sa": Sa_flat,
        "Sj": Sj_flat,
        "Fmax": Fmax_flat,
        "R0": R0
    }
    
    if K_flat is not None:
        stats["K"] = K_flat * pop_scalar
        
    return stats

def calculate_pioneer_fitness(R0_grid, Q_grid):
    """
    Calculates the fitness of a migrant arriving at a destination.
    Fitness = Destination Quality (R0) * Survival of the Journey (Q).
    
    Q_grid shape: (..., Ny, Nx, K_kernels)
    R0_grid shape: (..., Ny, Nx)
    """
    # Expand R0 to match kernel dimension if necessary
    if Q_grid.ndim > R0_grid.ndim:
        R0_expanded = jnp.expand_dims(R0_grid, axis=-1)
    else:
        R0_expanded = R0_grid
        
    return R0_expanded * Q_grid

def get_uncertainty_summaries(data_array):
    """
    Calculates mean and 90% Credibility Intervals across the first axis.
    Assumes axis 0 is the 'Sample' axis.
    """
    if data_array.ndim < 2:
        return data_array, None, None # Case for MAP (no samples)
        
    mean = jnp.mean(data_array, axis=0)
    lower = jnp.percentile(data_array, 5, axis=0)
    upper = jnp.percentile(data_array, 95, axis=0)
    
    return mean, lower, upper

def calculate_environmental_contributions(Z_gathered, weights):
    """
    Calculates the linear contribution of each Z feature.
    Z: (Time, Space, M)
    Weights: (M,) or (Samples, M)
    """
    # Logic to handle broadcasting samples over space/time
    if weights.ndim == 2: # SVI Samples
        # Result: (Samples, Time, Space, M)
        return Z_gathered[None, ...] * weights[:, None, None, :]
    else: # MAP Weights
        # Result: (Time, Space, M)
        return Z_gathered * weights[None, None, :]