"""Shared plotting helpers for the sample-based age-model visualizers
(``visualize_advi_model`` and ``visualize_hmc_model``).

Both backends summarize the posterior as per-site means/variances plus raw
global samples, so their diagnostic plots are identical apart from a few
title/filename labels. Those are parameterized here; the MAP visualizer
(``visualize_age_model``) works on point-estimate ``sim`` objects instead and
keeps its own richer, point-estimate-specific plots.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


def plot_posterior_weights(raw_samples, M, z_names, output_dir, label="SVI", fname="1_posterior_weights.png"):
    """Beta_s / beta_r estimates with 90% credible intervals."""
    fig, ax = plt.subplots(figsize=(10, 8))

    bs_samples = raw_samples['w_env'][..., 0]
    br_samples = raw_samples['w_env'][..., 1]

    bs_mean, br_mean = np.mean(bs_samples, axis=0), np.mean(br_samples, axis=0)

    bs_err = np.vstack([bs_mean - np.percentile(bs_samples, 5, axis=0),
                        np.percentile(bs_samples, 95, axis=0) - bs_mean])
    br_err = np.vstack([br_mean - np.percentile(br_samples, 5, axis=0),
                        np.percentile(br_samples, 95, axis=0) - br_mean])

    y_pos = np.arange(M)
    for i in range(M):
        ax.plot([bs_mean[i], br_mean[i]], [y_pos[i], y_pos[i]], color='gray', linestyle=':', alpha=0.7, zorder=1)

    ax.errorbar(bs_mean, y_pos, xerr=bs_err, fmt='o', color='dodgerblue', label=r'Survival ($\beta_s$)', markersize=8, capsize=3, zorder=3)
    ax.errorbar(br_mean, y_pos, xerr=br_err, fmt='s', color='darkorange', label=r'Reproduction ($\beta_r$)', markersize=8, capsize=3, zorder=2)

    ax.axvline(0, color='black', linestyle='-', alpha=0.3, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(z_names)
    ax.set_xlabel("Learned Weight (90% CI)")
    ax.set_title(f"Environmental Profile: Survival vs. Reproduction ({label} Posterior)")
    ax.set_ylim(-0.5, M - 0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()


def plot_demographic_response_curves(raw_samples, M, z_names, target_z_idx, output_dir):
    """Sweeps a target Z axis to show the non-linear biological response with 90% HDI."""
    z_sweep = np.linspace(-3, 3, 100)
    Z_matrix = np.zeros((100, M))
    Z_matrix[:, target_z_idx] = z_sweep

    def sigmoid(x): return 1 / (1 + np.exp(-x))
    def softplus(x): return np.log1p(np.exp(x))

    alpha_a = raw_samples['alpha_a'][:, None]
    alpha_j = raw_samples['alpha_j'][:, None]
    alpha_f = raw_samples['alpha_f'][:, None]

    gamma_a = softplus(raw_samples['gamma_a_raw'])[:, None]
    gamma_j = gamma_a + raw_samples['gamma_j_diff'][:, None]
    gamma_f = softplus(raw_samples['gamma_f_raw'])[:, None]

    H_s = np.dot(Z_matrix, raw_samples['w_env'][..., 0].T).T
    H_r = np.dot(Z_matrix, raw_samples['w_env'][..., 1].T).T

    S_a_curves = sigmoid(alpha_a + gamma_a * H_s)
    S_j_curves = sigmoid(alpha_j + gamma_j * H_s)
    # Must match the model's link function (age_fields.py uses softplus, not exp).
    F_max_curves = softplus(alpha_f + gamma_f * H_r)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    def plot_band(ax, x, curves, color, label, linestyle='-'):
        mean_c = np.mean(curves, axis=0)
        low_c = np.percentile(curves, 5, axis=0)
        high_c = np.percentile(curves, 95, axis=0)
        ax.plot(x, mean_c, color=color, linewidth=2, linestyle=linestyle, label=label)
        ax.fill_between(x, low_c, high_c, color=color, alpha=0.2, edgecolor='none')

    plot_band(ax1, z_sweep, S_a_curves, 'navy', r'Adult Survival ($S_a$)')
    plot_band(ax1, z_sweep, S_j_curves, 'royalblue', r'Juvenile Survival ($S_j$)', '--')

    ax1.set_xlabel(f"{z_names[target_z_idx]} Gradient (Standardized)")
    ax1.set_ylabel("Annual Survival Probability", color='navy')
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='y', labelcolor='navy')

    ax2 = ax1.twinx()
    plot_band(ax2, z_sweep, F_max_curves, 'darkorange', r'Max Fecundity ($F_{max}$)')
    ax2.set_ylabel("Maximum Fecundity", color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    plt.title(f"Demographic Response to {z_names[target_z_idx]} (90% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"2_response_curve_{z_names[target_z_idx]}.png"), dpi=300)
    plt.close()


def plot_spatial_uncertainty(Sa_grid_var, Sj_grid_var, Fmax_grid_var, K_grid_var, output_dir, land_mask):
    print("Generating Spatial Uncertainty Maps (Standard Deviations)...")

    Sa_std = np.sqrt(np.mean(Sa_grid_var, axis=0))
    Sj_std = np.sqrt(np.mean(Sj_grid_var, axis=0))
    Fmax_std = np.sqrt(np.mean(Fmax_grid_var, axis=0))
    K_std = np.sqrt(np.mean(K_grid_var, axis=0))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plots = [
        (Sa_std, r"Uncertainty: Adult Survival ($\sigma_{S_a}$)", "magma", axes[0, 0]),
        (Sj_std, r"Uncertainty: Juvenile Survival ($\sigma_{S_j}$)", "magma", axes[0, 1]),
        (Fmax_std, r"Uncertainty: Max Fecundity ($\sigma_{F_{max}}$)", "inferno", axes[1, 0]),
        (K_std, r"Uncertainty: Carrying Capacity ($\sigma_K$)", "cividis", axes[1, 1])
    ]

    for grid, title, cmap, ax in plots:
        masked_grid = np.ma.masked_where(land_mask == 0, grid)
        im = ax.imshow(masked_grid, cmap=cmap, origin='upper')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label="Standard Deviation", fraction=0.046, pad=0.04)

    plt.suptitle("Spatiotemporal Environmental Uncertainty (Model Confidence)", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "analysis_spatial_uncertainty.png"), dpi=200)
    plt.close()


def plot_demographic_baselines(Sa_grid_mean, Sj_grid_mean, Fmax_grid_mean, K_grid_mean, output_dir, land_mask, scalar):
    print("Generating Age-Structured Demographic Baselines (Means)...")
    Sa_avg = np.mean(Sa_grid_mean, axis=0)
    Sj_avg = np.mean(Sj_grid_mean, axis=0)
    Fmax_avg = np.mean(Fmax_grid_mean, axis=0)
    K_avg = np.mean(K_grid_mean, axis=0) * scalar

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plots = [
        (Sa_avg, r"Adult Survival ($S_a$)", "plasma", axes[0, 0], (0, 1)),
        (Sj_avg, r"Juvenile Survival ($S_j$)", "plasma", axes[0, 1], (0, 1)),
        (Fmax_avg, r"Maximum Fecundity ($F_{\max}$)", "viridis", axes[1, 0], None),
        (K_avg, r"Carrying Capacity ($K$)", "cividis", axes[1, 1], 'log')
    ]

    for grid, title, cmap, ax, scale in plots:
        masked_grid = np.ma.masked_where(land_mask == 0, grid)
        if scale == 'log':
            im = ax.imshow(np.log1p(masked_grid), cmap=cmap, origin='upper')
            cbar_label = "Log(1 + Expected Birds)"
        elif isinstance(scale, tuple):
            im = ax.imshow(masked_grid, cmap=cmap, origin='upper', vmin=scale[0], vmax=scale[1])
            cbar_label = "Probability"
        else:
            im = ax.imshow(masked_grid, cmap=cmap, origin='upper')
            cbar_label = "Offspring per Adult"

        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    plt.suptitle("Age-Structured Demographic Manifold (Expected Mean)", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "analysis_demographic_baselines.png"), dpi=200)
    plt.close()


def create_animation(density, obs_grid, years, output_dir, land_mask, filename="evolution_history.mp4", logscale=False):
    print(f"Generating Expected Density Animation: {filename}...")
    fig, (ax_sim, ax_obs) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(top=0.85)

    vmax_val = np.nanpercentile(density, 99)
    norm = mcolors.LogNorm(vmin=1e-6, vmax=vmax_val) if logscale else mcolors.Normalize(vmin=0, vmax=vmax_val)
    im_sim = ax_sim.imshow(density[0], cmap='magma', origin='upper', norm=norm)

    land_bg = np.zeros_like(land_mask, dtype=float)
    land_bg[land_mask == 1] = 0.2
    ax_obs.imshow(land_bg, cmap='gray_r', vmin=0, vmax=1, origin='upper', alpha=0.3)
    im_obs = ax_obs.imshow(obs_grid[0], cmap='magma', origin='upper', norm=norm, interpolation='none')

    title = fig.suptitle(f"Year: {years[0]}", fontsize=16, fontweight='bold')

    def update(frame):
        title.set_text(f"Year: {years[frame]}")
        sim_data = np.clip(density[frame], 1e-8, None) if logscale else density[frame]
        obs_data = np.clip(obs_grid[frame], 1e-8, None) if logscale else obs_grid[frame]
        im_sim.set_data(sim_data)
        im_obs.set_data(obs_data)
        return im_sim, im_obs, title

    ani = animation.FuncAnimation(fig, update, frames=len(years), interval=100, blit=False)
    save_path = os.path.join(output_dir, filename)
    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        ani.save(save_path, writer=writer)
    except Exception:
        print(f"FFMpegWriter failed. Saving as GIF instead: {save_path.replace('.mp4', '.gif')}")
        ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=10)
    plt.close()


def scatter_to_grid_robust(flat_array, rows, cols, shape):
    Ny, Nx = shape
    if flat_array is None: return None
    if flat_array.ndim == 1:
        grid = np.zeros((Ny, Nx)); grid[rows, cols] = flat_array; return grid
    elif flat_array.ndim == 2:
        if flat_array.shape[0] == len(rows):
            K = flat_array.shape[1]; grid = np.zeros((Ny, Nx, K)); grid[rows, cols, :] = flat_array
        else:
            T = flat_array.shape[0]; grid = np.zeros((T, Ny, Nx)); grid[:, rows, cols] = flat_array
        return grid
    elif flat_array.ndim == 3:
        T, _, K = flat_array.shape; grid = np.zeros((T, Ny, Nx, K)); grid[:, rows, cols, :] = flat_array
        return grid
    return None


def scatter_observations_to_grid(obs, t_idx, rows, cols, shape, time_steps):
    grid = np.full((time_steps, shape[0], shape[1]), np.nan)
    grid[t_idx, rows, cols] = obs
    return grid


def analyze_source_sink_mortality(data, Sa_grid, Sj_grid, Fmax_grid, source_prob_mean, output_dir):
    print("Generating Probabilistic Source-Sink & Mortality Maps...")
    land_mask = data['land_mask']
    os.makedirs(output_dir, exist_ok=True)

    prob_source_avg = np.mean(source_prob_mean, axis=0)
    prob_masked = np.ma.masked_where(land_mask == 0, prob_source_avg)

    # 1. Binary Consensus Source-Sink Map
    plt.figure(figsize=(10, 8))
    consensus_map = (prob_masked > 0.5).astype(float)

    cmap_binary = mcolors.ListedColormap(['#d73027', '#4575b4'])  # Red=Sink, Blue=Source
    plt.imshow(consensus_map, cmap=cmap_binary, origin='upper')

    legend_elements = [
        mpatches.Patch(color='#4575b4', label=r'Consensus Source ($P(R_0 > 1) > 0.5$)'),
        mpatches.Patch(color='#d73027', label=r'Consensus Sink ($P(R_0 > 1) \leq 0.5$)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True)
    plt.title("Probabilistic Source-Sink Landscape (Bayesian Consensus)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_source_sink_probabilistic.png"), dpi=200)
    plt.close()

    # 1B. Continuous Gradient Source-Sink Map
    plt.figure(figsize=(11, 8))
    im_continuous = plt.imshow(prob_masked, cmap='RdYlBu_r', origin='upper', vmin=0.0, vmax=1.0)

    cbar_cont = plt.colorbar(im_continuous, fraction=0.036, pad=0.04)
    cbar_cont.set_label(r"Probability of Functioning as a Demographic Source $P(R_0 > 1.0)$")

    plt.title("Probabilistic Source-Sink Landscape (Continuous Bayesian Gradient)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_source_sink_continuous.png"), dpi=200)
    plt.close()

    # 2. Uncertainty/Fuzziness Map (The "Range Edge")
    fuzziness = 1.0 - 2.0 * np.abs(prob_masked - 0.5)
    plt.figure(figsize=(11, 8))
    plt.imshow(fuzziness, cmap='magma', origin='upper', vmin=0, vmax=1)

    cbar_fuzz = plt.colorbar(im_continuous, fraction=0.036, pad=0.04)
    cbar_fuzz.set_label("Model Uncertainty (Distance from 0.5 Probability)")

    plt.title("Range Limit Fuzziness (Stochastic Edge)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_range_edge_fuzziness.png"), dpi=200)
    plt.close()
