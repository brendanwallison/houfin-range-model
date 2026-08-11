"""Path integration for the dispersal features (Z_disp): the FFT operator.

For each year, applies the directional/radial dispersal cohorts to latent Z. At each
fractional displacement, normalized convolution excludes ocean/nodata and conditions on
the remaining land support. Averaging those fractions yields a land-conditioned
neighborhood/path summary, not a literal water-crossing hazard.

This module is the operator only. The driver that loads the cube, resolves the kernels,
writes ``Z_disp_{year}.npz`` and renders the diagnostics is
``src/processing/generate_all_path_features.py``; every importer here takes
``integrate_paths`` and nothing else.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import fft2, ifft2
from tqdm import tqdm

def resize_kernel_stack(kernel_stack, scale):
    """
    Resizes a stack of kernels by a scale factor 's'.
    
    CRITICAL FIX: 
    Kernels are in FFT layout (mass at corners).
    We must fftshift (mass to center) BEFORE resizing/padding, 
    then ifftshift (mass back to corners) AFTER.
    Otherwise, padding moves the mass from index 0 to N/2 (massive displacement).
    """
    K, Ly, Lx = kernel_stack.shape
    new_H = int(Ly * scale)
    new_W = int(Lx * scale)
    
    if new_H < 1 or new_W < 1:
        return jnp.zeros_like(kernel_stack)

    # 1. Shift Mass to Center (Spatial Layout)
    stack_centered = jnp.fft.fftshift(kernel_stack, axes=(-2, -1))

    # 2. Resize (Shrink the centered blob)
    shrunk = jax.image.resize(stack_centered, (K, new_H, new_W), method='bilinear')
    
    # 3. Pad (Restores original frame size, keeping blob in center)
    # Defensive max(0, ...) protects against floating point issues when scale=1.0
    pad_y = max(0, (Ly - new_H) // 2)
    pad_x = max(0, (Lx - new_W) // 2)
    pad_y_end = max(0, Ly - new_H - pad_y)
    pad_x_end = max(0, Lx - new_W - pad_x)
    padded = jnp.pad(shrunk, ((0,0), (pad_y, pad_y_end), (pad_x, pad_x_end)))
    
    # 4. Unshift Mass to Corners (FFT Layout)
    result = jnp.fft.ifftshift(padded, axes=(-2, -1))
    
    # 5. Restore Sum (Mass Conservation)
    current_sum = jnp.sum(result, axis=(1,2), keepdims=True)
    target_sum = jnp.sum(kernel_stack, axis=(1,2), keepdims=True)
    scale_factor = jnp.where(current_sum > 1e-9, target_sum / current_sum, 0.0)
    
    return result * scale_factor

@jax.jit
def convolve_step(Z_t, kernel_stack_fft):
    """Efficiently convolves a batch of Z features."""
    Ny, Nx, _ = Z_t.shape
    K, Ly, Lx = kernel_stack_fft.shape
    
    pad_y = max(0, Ly - Ny)
    pad_x = max(0, Lx - Nx)
    Z_padded = jnp.pad(Z_t, ((0, pad_y), (0, pad_x), (0, 0)))
    
    Z_fft = jnp.fft.fft2(Z_padded, axes=(0, 1)).transpose(2, 0, 1)
    
    # (Batch, 1, Ly, Lx) * (1, K, Ly, Lx)
    # Standard convolution here correctly "looks back" at the history
    conv_fft = Z_fft[:, None, :, :] * kernel_stack_fft[None, :, :, :]
    conv_spatial = jnp.real(jnp.fft.ifft2(conv_fft, axes=(-2, -1)))
    return conv_spatial[:, :, :Ny, :Nx]

@jax.jit
def convolve_mask_step(mask, kernel_stack_fft):
    """Convolves binary land mask to find normalization weights."""
    Ny, Nx = mask.shape
    K, Ly, Lx = kernel_stack_fft.shape
    
    pad_y = max(0, Ly - Ny)
    pad_x = max(0, Lx - Nx)
    mask_padded = jnp.pad(mask, ((0, pad_y), (0, pad_x)))
    
    mask_fft = fft2(mask_padded)
    conv_fft = mask_fft[None, :, :] * kernel_stack_fft
    return jnp.real(ifft2(conv_fft, axes=(-2, -1)))[:, :Ny, :Nx]

def integrate_paths(Z, kernel_stack, land_mask, steps=10, feature_batch_size=4):
    """Compute land-conditioned cohort features by normalized convolution."""
    print(f"Path Integration: Z={Z.shape}, Kernels={kernel_stack.shape}, Steps={steps}")
    
    Time, Ny, Nx, M = Z.shape
    K_kernels = kernel_stack.shape[0]
    
    # 1. Sanitize Z (NaN -> 0.0)
    Z_safe = jnp.nan_to_num(Z, nan=0.0)
    
    # 2. Mask Z (Water -> 0.0)
    # This ensures water pixels don't contribute to the numerator
    mask_bc = land_mask[None, :, :, None] 
    Z_masked = jnp.where(mask_bc > 0.5, Z_safe, 0.0)
    
    Z_disp_acc = jnp.zeros((Time, M, K_kernels, Ny, Nx))
    s_vals = np.linspace(1.0 / steps, 1.0, steps)
    
    for s in tqdm(s_vals, desc="Integrating Paths"):
        # A. Resize Kernels
        scaled_kernels = resize_kernel_stack(kernel_stack, s)
        scaled_kernels_fft = fft2(scaled_kernels)
        
        # B. Compute Normalizer (Denominator: How much land was traversed?)
        land_weight = convolve_mask_step(land_mask, scaled_kernels_fft)
        has_land_support = land_weight > 1e-10
        safe_land_weight = jnp.where(has_land_support, land_weight, 1.0)
        
        for t in range(Time):
            for i in range(0, M, feature_batch_size):
                z_slice = Z_masked[t, :, :, i : i + feature_batch_size]
                
                # Convolution (Numerator: Sum of Z on Land)
                num_slice = convolve_step(z_slice, scaled_kernels_fft)
                
                # Normalization (Numerator / Denominator)
                avg_feat_slice = jnp.where(
                    has_land_support[None, :, :, :],
                    num_slice / safe_land_weight[None, :, :, :],
                    0.0,
                )
                
                Z_disp_acc = Z_disp_acc.at[t, i : i + feature_batch_size].add(avg_feat_slice)
            
    Z_disp_final = jnp.transpose(Z_disp_acc / steps, (0, 3, 4, 1, 2))
    return Z_disp_final
