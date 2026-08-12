"""FFT-based dispersal kernels for the age-structured forward simulation.

Dispersal each timestep is a convolution of the population field with a
distance-decay kernel. Convolutions are done in the Fourier domain (``fft2`` /
``ifft2``) because the kernel spans the whole grid and a direct convolution
would be O(N^2) per step. Kernels are the 2-D radial **generalized Gaussian**
``exp(-(r/scale)^shape)`` (shape<2 = fat-tailed, longer-distance dispersal);
``scale`` is set from the mean dispersal distance via gamma-function moments
(:func:`get_gamma_scale`).

Two populations disperse differently: adults isotropically (one kernel), and
juveniles anisotropically -- the juvenile master kernel is split into directional
x radial **wedges** (:func:`make_radial_directional_kernels`) via a partition of
unity, so the wedges sum back to the master kernel (mass-conserving; no
per-wedge renormalization). Because the grid is finite, each kernel also gets an
**edge correction**: the fraction of its mass that lands on valid habitat
(:func:`edge_correction_from_fft`), which the forward step divides by so mass
isn't lost off-grid or into water. All grids use odd padded dimensions and a
toroidal (wrap-around) distance convention required by the FFT.
"""
import warnings

import jax.numpy as jnp
import jax.nn
from jax.numpy.fft import fft2, ifft2

# Re-exported so callers can resolve the dispersal spec from here. These live in a
# deliberately JAX-FREE module, so tooling that only needs the spec -- notably the sweep
# driver's login-node preflight -- can import it without JAX, whose CPU-backend init
# aborts on TACC login nodes.
from src.model.dispersal_spec import (  # noqa: F401
    dispersal_spec,
    get_dispersal_quantiles,
    get_gamma_scale,
    resolve_radial_splits,
)

# 1. CORE GEOMETRY & MATH

def toroidal_distance_grid(Lx: int, Ly: int, cell_size: float) -> jnp.ndarray:
    """Distance (in km) from the origin to every cell, wrapped toroidally.

    Uses the FFT wrap-around convention: index 0 is the origin and distances
    increase then mirror back toward the far edge, so a kernel built on this grid
    convolves correctly under ``fft2``. ``Lx``/``Ly`` must be odd (symmetric
    wrap); ``cell_size`` scales grid steps to kilometres.
    """
    if (Lx % 2 == 0) or (Ly % 2 == 0):
        raise ValueError("Lx and Ly must be odd")

    x = jnp.concatenate([jnp.arange(Lx//2 + 1), jnp.arange(Lx//2, 0, -1)])
    y = jnp.concatenate([jnp.arange(Ly//2 + 1), jnp.arange(Ly//2, 0, -1)])

    x_steps = jnp.tile(x, (Ly, 1))
    y_steps = jnp.tile(y[:, None], (1, Lx))

    return jnp.sqrt(x_steps**2 + y_steps**2) * cell_size

def angular_weights_toroidal(Lx: int, Ly: int):
    """Per-cell directional weights (N/S/E/W) for splitting a kernel into wedges.

    Returns a dict of four smooth angular tapers (raised-cosine over +/-pi/2 about
    each cardinal direction) on the toroidal grid; together they form a partition
    of unity over direction, so multiplying a radial kernel by each and summing
    recovers the original. Direction is undefined at the origin, so its mass is
    divided equally among the four cohorts.
    """
    y_idx, x_idx = jnp.meshgrid(jnp.arange(Ly), jnp.arange(Lx), indexing="ij")
    
    # Centered coordinates for toroidal FFT conventions
    dx = jnp.where(x_idx <= Lx // 2, x_idx, x_idx - Lx)
    dy = jnp.where(y_idx <= Ly // 2, y_idx, y_idx - Ly)
    
    angles = jnp.arctan2(dy, dx) 

    directions = {
        'to_NORTH': -jnp.pi/2, 
        'to_SOUTH':  jnp.pi/2, 
        'to_EAST':   0.0, 
        'to_WEST':   jnp.pi
    }
    width = jnp.pi 

    w_dict = {}
    for d, target_angle in directions.items():
        diff = jnp.mod(angles - target_angle + jnp.pi, 2*jnp.pi) - jnp.pi
        taper = jnp.clip(diff, -width/2, width/2) 
        weight = 0.5 * (1 + jnp.cos(jnp.pi * taper / (width/2)))
        weight = jnp.where((dx == 0) & (dy == 0), 0.25, weight)
        w_dict[d] = weight

    return w_dict

def edge_correction_from_fft(fft_land, fft_kernel, land_mask, Ny, Nx, eps=1e-12):
    """
    Calculates the Fraction of the kernel that lands on valid habitat.
    
    FIX: Returns the FRACTION (denominator), not the reciprocal.
    age_forward.py divides by this value: Result = Conv / Fraction.
    """
    # Cross-Correlation in Fourier Domain = F(A) * conj(F(B))
    fraction_land = jnp.real(ifft2(fft_land * jnp.conj(fft_kernel)))[:Ny, :Nx]
    fraction_land = jnp.maximum(fraction_land, eps)
    
    # On water pixels, we don't care (mask later), but set to 1.0 to avoid NaNs
    fraction_land = jnp.where(land_mask, fraction_land, 1.0)
    
    return fraction_land

# 2. KERNEL BUILDERS (The Factories)

def make_radial_directional_kernels(
    Lx, Ly, 
    cell_size, 
    base_kernel_grid,   # [NEW] The Master PDF
    radii_splits, 
    smoothness_km=None
):
    """
    Splits the base_kernel_grid into 4 x (len(radii_splits)-1) wedges using soft masking
    (12 at the pinned 3-band config; tests/test_dispersal_physics.py exercises a 4-band, K=16 case).
    Does NOT re-normalize. The sum of all kernels equals base_kernel_grid.
    """
    if smoothness_km is None:
        smoothness_km = 2.0 * cell_size

    r_dist = toroidal_distance_grid(Lx, Ly, cell_size)
    angular_w = angular_weights_toroidal(Lx, Ly)
    
    kernels = []
    labels = []
    
    slope = 4.0 / smoothness_km
    
    radial_cdfs = []
    for r_boundary in radii_splits:
        if r_boundary <= 1e-6:
            cdf = jnp.ones_like(r_dist)
        elif r_boundary >= 1e9:
            cdf = jnp.zeros_like(r_dist)
        else:
            cdf = jax.nn.sigmoid(slope * (r_dist - r_boundary))
        radial_cdfs.append(cdf)
        
    direction_order = ['to_NORTH', 'to_SOUTH', 'to_EAST', 'to_WEST']
    
    for d in direction_order:
        w_dir = angular_w[d]
        
        for i in range(len(radii_splits) - 1):
            r_min_val = radii_splits[i]
            r_max_val = radii_splits[i+1]

            # 1. Calculate the Partition of Unity Mask
            # (Which fraction of space belongs to this bin?)
            mask_radial = radial_cdfs[i] - radial_cdfs[i+1]
            mask_combined = w_dir * mask_radial

            # 2. Apply Mask to Base PDF (Scenario B)
            # This preserves the probability mass of the donut.
            k = base_kernel_grid * mask_combined

            # Deliberately NOT renormalized per wedge: the wedges partition the master
            # kernel's mass, and rescaling each to sum 1 would destroy that partition.
            # make_juvenile_kernel_stack asserts the total is conserved.

            kernels.append(k)
            labels.append(f"{d}_{r_min_val:.0f}-{r_max_val:.0f}")

    return jnp.stack(kernels, axis=0), labels


# --- Juvenile dispersal kernel: SINGLE SOURCE OF TRUTH -----------------------------------
# The forward simulation (build_simulation_struct) MOVES juveniles with this kernel stack,
# and the path-feature builder (generate_all_path_features) gathers origin/path habitat with
# the SAME stack -> the per-kernel journey survival Q[p,k] is only meaningful if both use an
# identical kernel family (base PDF + radii_splits + cell_size). Build both through this one
# function so they cannot drift. Mean dispersal distance / shape live here as the sole source.


def make_juvenile_kernel_stack(Lx, Ly, cell_size, radii_splits,
                               mean_dist, shape):
    """Directional x radial juvenile dispersal kernels: the master PDF split into wedges.

    Builds the normalized 2-D radial generalized-Gaussian master ``exp(-(r/scale)^shape)``
    (``scale`` set from ``mean_dist`` via gamma moments) and splits it via
    :func:`make_radial_directional_kernels`. Returns ``(stack (K,Ly,Lx), labels)``.

    ``mean_dist``/``shape`` are required rather than defaulting to the module
    constants: a silent 330 km default would mask a config or sweep overlay that
    meant to set something else.
    """
    # Resolution guard for small mdd. The radial boundaries are soft sigmoids of
    # width ~2*cell_size (see make_radial_directional_kernels), so a first split
    # comparable to that width means the innermost cohort's mass leaks heavily
    # into the next band and the "0-r" label is nominal. Warn rather than raise:
    # a dispersal sweep deliberately visits short distances, and the caller may
    # legitimately accept a partly-resolved inner cohort.
    smoothness_km = 2.0 * cell_size
    if len(radii_splits) > 2 and 0.0 < radii_splits[1] < smoothness_km:
        warnings.warn(
            f"innermost radial split {radii_splits[1]:.1f} km is below the "
            f"boundary smoothing width {smoothness_km:.1f} km (2 x {cell_size:.1f} km "
            f"cells) at mean_dist={mean_dist:.0f} km: the inner cohort is only "
            f"partly resolved and overlaps the next band",
            RuntimeWarning, stacklevel=2,
        )

    r_dist = toroidal_distance_grid(Lx, Ly, cell_size)
    scale = get_gamma_scale(mean_dist, shape)
    master = jnp.exp(-(r_dist / scale) ** shape)
    master = master / jnp.sum(master)
    stack, labels = make_radial_directional_kernels(
        Lx, Ly, cell_size, master, radii_splits)
    # This is a physical probability decomposition. Fail immediately if a future
    # angular/radial refactor stops being a partition of unity.
    total = jnp.sum(stack)
    if not bool(jnp.isclose(total, 1.0, rtol=2e-5, atol=2e-6)):
        raise ValueError(f"juvenile kernel stack is not mass-conserving (sum={float(total):.8f})")
    return stack, labels


def build_simulation_struct(
    land: jnp.ndarray,
    cell_size: float,
    adult_mdd: float,
    juvenile_mdd: float,
    adult_shape: float,
    juvenile_shape: float,
    radii_splits
):
    """
    Builds simulation structure with mass-conservative weighted kernels.

    ``radii_splits`` is REQUIRED and must come from
    ``dispersal_spec(config)["juvenile_radial_splits_km"]``. This used to accept
    None and derive terciles itself, but a second derivation site is exactly the
    drift this module's juvenile-kernel note warns about: the path-feature
    builder and the forward model must use an identical kernel family, and that
    is only guaranteed if both take their splits from the one resolver.
    """
    if radii_splits is None:
        raise ValueError(
            "radii_splits is required; pass "
            "dispersal_spec(config)['juvenile_radial_splits_km'] (which resolves "
            'an explicit list or the "derive" sentinel) rather than relying on a '
            "second, independent derivation here"
        )
    Ny, Nx = land.shape
    Lx, Ly = 2 * Nx - 1, 2 * Ny - 1
    land_mask = land.astype(bool)
    
    padded_land = jnp.zeros((Ly, Lx)).at[:Ny, :Nx].set(land)
    fft_land = fft2(padded_land)
    
    # 1. Grid (in km)
    r_dist = toroidal_distance_grid(Lx, Ly, cell_size)
    
    # 2. Adult (Isotropic)
    adult_scale = get_gamma_scale(adult_mdd, adult_shape)
    adult_kernel = jnp.exp(-(r_dist / adult_scale) ** adult_shape)
    adult_kernel /= jnp.sum(adult_kernel) # Normalize Master
    adult_fft_kernel = fft2(adult_kernel)
    
    # Adult Edge Correction (Standard)
    adult_edge_correction = edge_correction_from_fft(fft_land, adult_fft_kernel, land_mask, Ny, Nx)

    # 3. Juvenile cohorts (K = 4 directions x radial bands)
    # A+B. Master juvenile PDF split into directional x radial wedges, via the shared builder
    # so the Z_disp path features use the IDENTICAL kernel family (same base PDF + splits).
    juv_kernels, labels = make_juvenile_kernel_stack(
        Lx, Ly, cell_size, radii_splits, mean_dist=juvenile_mdd, shape=juvenile_shape)
    
    fft_list = []
    edge_list = []
    
    # C. Calculate Edge Corrections for Weighted Kernels
    for i in range(juv_kernels.shape[0]):
        k_weighted = juv_kernels[i]
        
        # 1. Store Weighted Kernel for Simulation
        fft_k = fft2(k_weighted)
        fft_list.append(fft_k)
        
        # 2. Calculate Edge Correction using NORMALIZED shape
        # Edge correction asks: "If I throw this SHAPE at land, what fraction hits?"
        # We must temporarily normalize to answer that question correctly.
        weight = jnp.sum(k_weighted)
        
        # Safety for empty kernels (remote possibility)
        k_normalized = jnp.where(weight > 1e-12, k_weighted / weight, k_weighted)
        fft_k_norm = fft2(k_normalized)
        
        # Returns FRACTION (0.0 to 1.0)
        edge_c = edge_correction_from_fft(fft_land, fft_k_norm, land_mask, Ny, Nx)
        edge_list.append(edge_c)

    return {
        "fft_land": fft_land,
        "adult_fft_kernel": adult_fft_kernel,
        "adult_edge_correction": adult_edge_correction,
        
        "juvenile_fft_kernel_stack": jnp.stack(fft_list, axis=0),        # (K, Ly, Lx)
        "juvenile_edge_correction_stack": jnp.stack(edge_list, axis=0),  # (12, Ny, Nx)
        
        "labels": labels,
        "radii_splits": radii_splits
    }
