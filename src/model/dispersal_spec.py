"""The config-owned dispersal specification -- resolved here, and ONLY here.

This module is deliberately **JAX-free** (scipy + stdlib only). ``dispersal_spec``
is needed by things that must not pay for, or cannot survive, importing JAX:

* ``scripts/tacc/submit_juv_mdd_sweep.sh`` resolves each sweep point's splits on a
  login node to validate overlays before submitting. TACC login nodes abort when
  JAX initializes its CPU backend (``make_cpu_client``), so importing
  ``build_kernels`` there hangs or core-dumps.
* provenance checks and summary scripts that only need to know what the spec is.

``build_kernels`` re-exports these names, so existing imports keep working and
there is still exactly one resolver. Splits must become numbers in one place
because ``generate_all_path_features`` and ``model_inputs`` run as separate jobs
and ingest compares the two resolved dicts for EXACT equality (see
``model_inputs.py``'s "path-feature dispersal specification differs" guard).
"""
import math

from scipy.special import gamma, gammaincinv

# Juvenile dispersal kernel defaults. Kept here beside the resolver so the
# config, the path-feature builder, and the forward model all trace back to one
# definition of the kernel family.
JUVENILE_MDD_KM = 330.0
JUVENILE_SHAPE = 0.468


def get_gamma_scale(mean_dist, shape):
    """Kernel ``scale`` giving a 2-D radial generalized Gaussian the target mean.

    For ``exp(-(r/scale)^shape)`` weighted by area (2-D radial), the mean radial
    distance is ``scale * Gamma(3/shape)/Gamma(2/shape)``; invert that so the
    kernel's mean dispersal distance equals ``mean_dist``.
    """
    return mean_dist * gamma(2.0 / shape) / gamma(3.0 / shape)


def get_dispersal_quantiles(mean_dist, shape_param, quantiles=(0.33, 0.66)):
    """Radii enclosing given fractions of the 2-D radial kernel's dispersal mass.

    Inverts the radial CDF (regularized incomplete gamma) of the generalized
    Gaussian with the given mean/shape. Used to split the juvenile kernel into
    roughly equal-mass radial bins (default terciles) for the directional wedges.
    """
    g2 = gamma(2.0 / shape_param)
    g3 = gamma(3.0 / shape_param)
    scale = mean_dist / (g3 / g2)

    radii = []
    for q in quantiles:
        val = gammaincinv(2.0 / shape_param, q)
        radii.append(scale * (val ** (1.0 / shape_param)))
    return radii


def _validate_splits(splits, origin):
    """Shared validator for pinned and derived radial splits."""
    if len(splits) < 2 or splits[0] != 0.0 or splits[-1] < 1e8:
        raise ValueError(f"{origin} must span [0, infinity], got {splits}")
    if any(b <= a for a, b in zip(splits[:-1], splits[1:])):
        raise ValueError(f"{origin} must increase strictly, got {splits}")
    if not all(math.isfinite(x) for x in splits[:-1]):
        raise ValueError(f"{origin} contains a non-finite interior boundary: {splits}")
    return splits


def resolve_radial_splits(juvenile_mdd_km, juvenile_shape, spec_value, quantiles=None):
    """Resolve ``juvenile_radial_splits_km`` to concrete km boundaries.

    Two accepted forms:

    * a **list** -- an explicit pin, validated and returned unchanged. Values are
      NOT rounded: the committed baseline's literals (155.36162529769288,
      482.7446923028151) must survive byte-for-byte, because ``model_inputs.py``'s
      ingest guard compares the resolved dispersal dict to the one recorded in an
      existing ``path_feature_meta.json`` for exact equality, and rounding them
      would invalidate the ~11 GB Z_disp cube built under them.
    * the string ``"derive"`` (or ``None``) -- equal-mass radial bands for THIS
      mdd, i.e. quantiles of the kernel itself. The hardcoded baseline splits are
      log-spaced edges from a deleted helper and do not move with mdd, so the
      three cohorts' mass shares swing wildly across a dispersal sweep (67/28/4%
      at 150 km vs 32/41/27% at 400 km). Deriving them keeps the radial
      discretization comparable across mdd instead of confounding it.

    Derived values are rounded to 6 dp (~1 mm, physically irrelevant against
    27 km cells) so that path-features and ingest -- separate SLURM jobs, possibly
    different nodes -- cannot disagree in ``gammaincinv``'s last bit and trip that
    same equality guard.
    """
    if isinstance(spec_value, (list, tuple)):
        return _validate_splits([float(x) for x in spec_value],
                                "juvenile_radial_splits_km")
    if spec_value is None or (isinstance(spec_value, str) and spec_value == "derive"):
        qs = [0.33, 0.66] if quantiles is None else [float(q) for q in quantiles]
        if not qs or any(not (0.0 < q < 1.0) for q in qs):
            raise ValueError(f"juvenile_radial_split_quantiles must lie in (0,1), got {qs}")
        if any(b <= a for a, b in zip(qs[:-1], qs[1:])):
            raise ValueError(f"juvenile_radial_split_quantiles must increase strictly, got {qs}")
        radii = get_dispersal_quantiles(juvenile_mdd_km, juvenile_shape, qs)
        if any((not math.isfinite(r)) or r <= 0.0 for r in radii):
            raise ValueError(
                f"derived radial splits are not finite and positive for "
                f"mdd={juvenile_mdd_km}, shape={juvenile_shape}: {radii}"
            )
        splits = [0.0] + [round(float(r), 6) for r in radii] + [1e9]
        return _validate_splits(splits, "derived juvenile_radial_splits_km")
    raise ValueError(
        "juvenile_radial_splits_km must be a list of km boundaries or the string "
        f'"derive" (equal-mass bands for the configured mdd); got {spec_value!r}'
    )


def dispersal_spec(config):
    """Return the validated, config-owned movement/path-feature specification.

    This is the SINGLE place radial splits become numbers. Both consumers --
    ``generate_all_path_features`` (which builds Z_disp) and ``model_inputs``
    (which builds the forward model's kernels) -- call this on the same config,
    and ingest compares the two resolved dicts for exact equality, so resolution
    must be deterministic and must live here alone. Do not add keys to the
    returned dict: every existing ``path_feature_meta.json`` records it verbatim,
    so a new key invalidates all previously built path features.
    """
    d = config.get("dispersal") or {}
    required = (
        "adult_mdd_km", "adult_shape", "juvenile_mdd_km", "juvenile_shape",
        "juvenile_radial_splits_km", "path_integration_steps", "path_operator",
    )
    missing = [key for key in required if key not in d]
    if missing:
        raise KeyError(f"age-model dispersal config missing {missing}")
    splits = resolve_radial_splits(
        float(d["juvenile_mdd_km"]), float(d["juvenile_shape"]),
        d["juvenile_radial_splits_km"],
        d.get("juvenile_radial_split_quantiles"),
    )
    if str(d["path_operator"]) != "land_conditioned_neighborhood":
        raise ValueError(
            "only path_operator='land_conditioned_neighborhood' is implemented"
        )
    if int(d["path_integration_steps"]) < 1:
        raise ValueError("path_integration_steps must be positive")
    for key in ("adult_mdd_km", "adult_shape", "juvenile_mdd_km", "juvenile_shape"):
        if float(d[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    return {
        "adult_mdd_km": float(d["adult_mdd_km"]),
        "adult_shape": float(d["adult_shape"]),
        "juvenile_mdd_km": float(d["juvenile_mdd_km"]),
        "juvenile_shape": float(d["juvenile_shape"]),
        "juvenile_radial_splits_km": splits,
        "path_integration_steps": int(d["path_integration_steps"]),
        "path_operator": str(d["path_operator"]),
    }
