"""Small process discrepancies with explicit variance and active-year semantics."""
import numpy as np


def active_indices(years, start, end, holdout_years=()):
    years = np.asarray(years, int)
    active = (years >= start) & (years <= end)
    if holdout_years:
        active &= years < min(holdout_years)
    return np.flatnonzero(active).astype(np.int32), active.astype(float)


def normalize_cell_variance(basis):
    basis = np.asarray(basis)
    norm = np.sqrt(np.sum(basis**2, axis=1))
    if np.any(norm <= 0):
        raise ValueError('zero-variance spatial-basis row')
    return basis / norm[:, None]


def iid_log_innovations(raw, indices, ntime, sigma, xp=np):
    """No dense identity basis; inactive cell-years have exactly zero discrepancy."""
    values = float(sigma) * raw - .5 * float(sigma)**2
    out = xp.zeros((ntime, raw.shape[1]), dtype=raw.dtype)
    if xp is np:
        out[indices] = values
        return out
    return out.at[indices].set(values)


def regional_log_innovations(raw, active, basis, sigma):
    """Basis rows have unit variance; all active-cell prior multipliers mean one."""
    return (float(sigma) * (raw * active[:, None])) @ basis.T - .5 * float(sigma)**2 * active[:, None]
