import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import jax.nn as jnn
from src.analysis.stats import get_uncertainty_summaries, calculate_demographics

# --- INTERNAL UTILITIES ---

def _to_grid(flat_array, data):
    """Utility to scatter 1D spatial arrays back to the 2D land mask."""
    Ny, Nx = data['Ny'], data['Nx']
    grid = np.full((Ny, Nx), np.nan)
    grid[data['land_rows'], data['land_cols']] = flat_array
    return grid

def _get_modern_avg(array, sample_axis=True):
    """Extracts the mean of the last 10 years of data."""
    # Input shape: (Samples, Time, Space) or (Time, Space)
    if sample_axis and array.ndim == 3:
        return np.mean(array[:, -10:, :], axis=(0, 1))
    elif array.ndim == 2:
        return np.mean(array[-10:, :], axis=0)
    return array

# --- 1. POSTERIOR WEIGHTS ---

def plot_posterior_weights(samples, z_names, output_dir):
    """Plots weights with 90% Credibility Intervals for SVI or points for MAP."""
    # Ensure we are looking at the right shapes
    # w_env shape is (Samples, M, 2)
    ws = samples['w_env'][..., 0] # Survival weights
    wr = samples['w_env'][..., 1] # Reproduction weights
    
    M = len(z_names)
    y_pos = np.arange(M)
    fig, ax = plt.subplots(figsize=(10, 8))

    for label, data, color, marker, offset in [
        (r'Survival ($\beta_s$)', ws, 'dodgerblue', 'o', 0.15),
        (r'Reproduction ($\beta_r$)', wr, 'darkorange', 's', -0.15)
    ]:
        mean, low, high = get_uncertainty_summaries(data)
        
        # Squeeze to ensure mean is 1D (M,) and matches y_pos (M,)
        mean = np.squeeze(mean)
        
        if mean.shape[0] != M:
            print(f"Warning: Shape mismatch! Mean shape {mean.shape} vs z_names {M}")
            # Dynamic fix if shapes are transposed
            if mean.shape[-1] == M: mean = mean.T

        ax.scatter(mean, y_pos + offset, color=color, label=label, marker=marker, s=64, zorder=3)
        
        if low is not None:
            low, high = np.squeeze(low), np.squeeze(high)
            # errorbar expects (2, M) for asymmetric errors
            errs = np.array([mean - low, high - mean])
            ax.errorbar(mean, y_pos + offset, xerr=errs, 
                        fmt='none', ecolor=color, alpha=0.5, capsize=3, zorder=2)
            
        for i in range(M):
            ax.plot([0, mean[i]], [y_pos[i] + offset, y_pos[i] + offset], 
                    color=color, linestyle=':', alpha=0.2, zorder=1)

    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(z_names)
    ax.set_xlabel("Learned Weight (Latent Scale)")
    ax.set_title("Environmental Profile: Survival vs. Reproduction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_posterior_weights.png"), dpi=300)
    plt.close()
    
# --- 2. DEMOGRAPHIC RESPONSE CURVES ---

def plot_demographic_response_curves(samples, data, z_names, target_idx, output_dir):
    """Sweeps a single environmental variable to show non-linear biological response."""
    z_sweep = np.linspace(-3, 3, 100)
    M = len(z_names)
    
    # We use the mean weights for the response curve
    ws = np.mean(samples['w_env'][..., 0], axis=0) if samples['w_env'].ndim == 3 else samples['w_env'][..., 0]
    wr = np.mean(samples['w_env'][..., 1], axis=0) if samples['w_env'].ndim == 3 else samples['w_env'][..., 1]
    
    # Sweep matrix
    Z_matrix = np.zeros((100, M))
    Z_matrix[:, target_idx] = z_sweep
    
    H_s = Z_matrix @ ws
    H_r = Z_matrix @ wr
    
    # Extract intercepts (using mean if SVI)
    def _m(k): return np.mean(samples[k]) if samples[k].ndim > 0 else samples[k]
    
    gamma_a = jnn.softplus(_m('gamma_a_raw'))
    gamma_j = gamma_a + _m('gamma_j_diff')
    gamma_f = jnn.softplus(_m('gamma_f_raw'))
    
    S_a = jnn.sigmoid(_m('alpha_a') + gamma_a * H_s)
    S_j = jnn.sigmoid(_m('alpha_j') + gamma_j * H_s)
    # Must match the model's link function (age_fields.py uses softplus, not exp).
    F_max = jnn.softplus(_m('alpha_f') + gamma_f * H_r)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(z_sweep, S_a, color='navy', lw=2, label='$S_a$')
    ax1.plot(z_sweep, S_j, color='royalblue', lw=2, linestyle='--', label='$S_j$')
    ax1.set_xlabel(f"{z_names[target_idx]} (Standardized)")
    ax1.set_ylabel("Survival Probability", color='navy')
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(z_sweep, F_max, color='darkorange', lw=2, label='$F_{max}$')
    ax2.set_ylabel("Max Fecundity", color='darkorange')
    
    plt.title(f"Biological Response to {z_names[target_idx]}")
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"2_response_curve_{z_names[target_idx]}.png"), dpi=300)
    plt.close()

# --- 3. TEMPORAL EPOCHS (DRIVERS AND LIMITS) ---

def plot_temporal_epochs(samples, data, z_names, output_dir):
    """Maps limiting and driving factors across early vs late eras."""
    ws = np.mean(samples['w_env'][..., 0], axis=0) if samples['w_env'].ndim == 3 else samples['w_env'][..., 0]
    wr = np.mean(samples['w_env'][..., 1], axis=0) if samples['w_env'].ndim == 3 else samples['w_env'][..., 1]
    
    Z = data['Z_gathered']
    M = len(z_names)
    
    # Define eras
    early_idx = np.where((data['years'] >= 1970) & (data['years'] <= 1975))[0]
    late_idx = np.where((data['years'] >= 2013) & (data['years'] <= 2023))[0]
    
    Z_early = np.mean(Z[early_idx], axis=0)
    Z_late = np.mean(Z[late_idx], axis=0)
    
    cmap = plt.get_cmap('tab20', M)
    patches = [mpatches.Patch(color=cmap(i), label=z_names[i]) for i in range(M)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, (Z_era, label) in enumerate([(Z_early, "Early (1970s)"), (Z_late, "Modern (2020s)")]):
        # Calculate local linear contribution
        cont_s = Z_era * ws
        cont_r = Z_era * wr
        
        # Limiting factor is the most negative contribution
        lim_s = np.argmin(cont_s, axis=-1)
        lim_r = np.argmin(cont_r, axis=-1)
        
        axes[0, i].imshow(_to_grid(lim_s, data), cmap=cmap, vmin=-0.5, vmax=M-0.5)
        axes[0, i].set_title(f"Survival Limitation | {label}")
        axes[1, i].imshow(_to_grid(lim_r, data), cmap=cmap, vmin=-0.5, vmax=M-0.5)
        axes[1, i].set_title(f"Reproduction Limitation | {label}")
        
    for ax in axes.flatten(): ax.axis('off')
    
    fig.legend(handles=patches, loc='lower center', ncol=min(M, 5), bbox_to_anchor=(0.5, 0.05))
    plt.suptitle("Spatiotemporal Limitation Shifts", fontsize=18)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "3_limitation_epochs.png"), dpi=300)
    plt.close()

# --- 4. CONTINENTAL VIOLINS ---

def plot_continental_violins(samples, data, z_names, output_dir):
    """Shows the spatial distribution of environmental impacts across the continent."""
    ws = np.mean(samples['w_env'][..., 0], axis=0) if samples['w_env'].ndim == 3 else samples['w_env'][..., 0]
    wr = np.mean(samples['w_env'][..., 1], axis=0) if samples['w_env'].ndim == 3 else samples['w_env'][..., 1]
    
    Z_modern = np.mean(data['Z_gathered'][-10:], axis=0)
    M = len(z_names)
    
    cont_s = Z_modern * ws
    cont_r = Z_modern * wr
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    y_pos = np.arange(M)
    
    for i, (cont, title, color) in enumerate([(cont_s, "Survival Impact", "dodgerblue"), (cont_r, "Reproduction Impact", "darkorange")]):
        dataset = [cont[:, j] for j in range(M)]
        parts = axes[i].violinplot(dataset, positions=y_pos, vert=False, showmeans=True)
        for pc in parts['bodies']: pc.set_facecolor(color)
        axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[i].set_title(title)
        
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(z_names)
    plt.suptitle("Continental Environmental Impact Distribution", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_continental_violins.png"), dpi=300)
    plt.close()

# --- 5. KEYSTONE R0 MAP ---

def plot_keystone_r0(samples, data, z_names, output_dir):
    """Maps the 'Keystone' feature—the variable whose removal causes the largest absolute change in R0."""
    ws = np.mean(samples['w_env'][..., 0], axis=0) if samples['w_env'].ndim == 3 else samples['w_env'][..., 0]
    wr = np.mean(samples['w_env'][..., 1], axis=0) if samples['w_env'].ndim == 3 else samples['w_env'][..., 1]
    
    def _m(k): return np.mean(samples[k]) if samples[k].ndim > 0 else samples[k]
    
    Z_modern = np.mean(data['Z_gathered'][-10:], axis=0)
    M = len(z_names)
    
    def calc_R0_local(Z_in):
        h_s = Z_in @ ws
        h_r = Z_in @ wr
        ga, gj, gf = jnn.softplus(_m('gamma_a_raw')), jnn.softplus(_m('gamma_a_raw')) + _m('gamma_j_diff'), jnn.softplus(_m('gamma_f_raw'))
        sa = jnn.sigmoid(_m('alpha_a') + ga * h_s)
        sj = jnn.sigmoid(_m('alpha_j') + gj * h_s)
        fm = np.exp(_m('alpha_f') + gf * h_r)
        return (fm * sj) / (1.0 - sa + 1e-6)

    baseline = calc_R0_local(Z_modern)
    impacts = []
    for j in range(M):
        Z_ablated = Z_modern.copy(); Z_ablated[:, j] = 0.0
        impacts.append(np.abs(baseline - calc_R0_local(Z_ablated)))
    
    keystone_idx = np.argmax(impacts, axis=0)
    grid = _to_grid(keystone_idx, data)
    
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab20', M)
    plt.imshow(grid, cmap=cmap, vmin=-0.5, vmax=M-0.5)
    plt.colorbar(label="Keystone Feature Index")
    plt.title("Keystone Environmental Drivers\n(Feature with largest impact on Local R0)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "5_keystone_map.png"), dpi=300)
    plt.close()

# --- 6. ANIMATION ---

def create_animation(samples, data, output_dir, filename="population_evolution.mp4"):
    """Creates a side-by-side animation of Predicted Density vs Observations."""
    density = samples['simulated_density'] * data['pop_scalar']
    mean_dens = np.mean(density, axis=0) if density.ndim == 3 else density
    
    obs_grid = np.full((data['time'], data['Ny'], data['Nx']), np.nan)
    obs_grid[data['obs_time_indices'], data['obs_rows'], data['obs_cols']] = data['observed_results']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    vmax = np.nanpercentile(mean_dens, 99)
    im1 = ax1.imshow(mean_dens[0], cmap='magma', vmin=0, vmax=vmax)
    im2 = ax2.imshow(obs_grid[0], cmap='magma', vmin=0, vmax=vmax)
    
    ax1.set_title("Predicted Density"); ax2.set_title("Observations")
    for ax in [ax1, ax2]: ax.axis('off')

    def update(t):
        im1.set_data(mean_dens[t])
        im2.set_data(obs_grid[t])
        return im1, im2

    ani = animation.FuncAnimation(fig, update, frames=data['time'], interval=100)
    ani.save(os.path.join(output_dir, filename), writer='ffmpeg', fps=10)
    plt.close()