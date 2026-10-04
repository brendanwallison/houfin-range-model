"""Shared machinery for posterior inference (probe, VI, NUTS) on the age model.

Every posterior entry point (``age_hmc_probe``, ``age_resume_svi_from_map``,
``age_resume_hmc``, ``age_run_hmc``) goes through this module so that the MAP source,
the unconstrained parameterization, the deterministic-hiding wrapper and the chunked,
resumable sampler cannot drift between them.

Import this BEFORE anything that creates JAX arrays: it sets ``jax_enable_x64`` from
``HOUFIN_MODEL_PRECISION`` exactly as ``age_run_map`` does.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PRECISION = os.environ.get("HOUFIN_MODEL_PRECISION", "float32")
if PRECISION not in {"float32", "float64"}:
    raise ValueError("HOUFIN_MODEL_PRECISION must be float32 or float64")
jax.config.update("jax_enable_x64", PRECISION == "float64")
# The precision the MAP was FITTED in, which may differ from the one inference runs
# in: a float64 sampler can start from the float32 MAP (and its float32 probe).
MAP_PRECISION = os.environ.get("HOUFIN_HMC_MAP_PRECISION", PRECISION)
if MAP_PRECISION not in {"float32", "float64"}:
    raise ValueError("HOUFIN_HMC_MAP_PRECISION must be float32 or float64")

import numpyro                                                     # noqa: E402
from numpyro.handlers import block                                 # noqa: E402
from numpyro.infer import MCMC                                     # noqa: E402
from numpyro.infer.initialization import init_to_value             # noqa: E402
from numpyro.infer.util import initialize_model                    # noqa: E402
from jax.flatten_util import ravel_pytree                          # noqa: E402

from src.config_utils import load_config                           # noqa: E402
from src.model.checkpoints import (                                # noqa: E402
    auto_delta_params_to_latents, load_map_params, save_pickle_atomic,
)

POSTERIOR_CONFIG = "age_posterior_config.json"
POSTERIOR_ENV = "AGE_POSTERIOR_CONFIG"

# The files that define the model's density. A MAP point is only a valid starting
# point / Laplace centre for the model it was fitted to, so a change to any of these
# (or to the inputs) since the MAP fit is refused unless explicitly allowed.
MODEL_SOURCES = ("age_priors.py", "age_fields.py", "age_forward.py",
                 "build_kernels.py", "data_loading.py")


# --------------------------------------------------------------------------- config

def load_posterior_config():
    return load_config(default_name=POSTERIOR_CONFIG, env_var=POSTERIOR_ENV)


def setting(section: dict, key: str, env: str, cast=float):
    """Config leaf with an env-var override (the MAP scripts' convention)."""
    raw = os.environ.get(env)
    return cast(raw) if raw is not None else cast(section[key])


def results_dir(name: str) -> str:
    from src.config_utils import load_age_model_config
    return os.path.join(load_age_model_config()["results_dir"], name)


def model_kwargs(data) -> dict:
    """Posterior inference always targets the NOMINAL priors (continuation scale 1)."""
    return {"data": data, "prior_scale": 1.0}


def split_data(data: dict):
    """(arrays, static): device arrays vs plain metadata (ints, strings, dicts).

    Compiled samplers must receive the ARRAYS as arguments. Closing over them makes
    every compile embed all 2.68 GB of inputs as constants, and numpyro's MCMC
    recompiles at every chunk, so host memory grew by several GB per chunk until
    TACC job 3486435 died at 64 GB RSS ("Failed to allocate buffer for Literal").
    The static part (time, inv_window, Nx, ...) must stay Python values: the model
    uses them as shapes.
    """
    arrays = {k: v for k, v in data.items() if isinstance(v, jax.Array)}
    static = {k: v for k, v in data.items() if k not in arrays}
    return arrays, static


def array_model(model, static: dict, prior_scale: float = 1.0):
    """``model(data, prior_scale)`` re-exposed as ``m(arrays=...)`` for jit_model_args.

    prior_scale is closed over (posterior inference is always at 1.0) so it cannot
    become a tracer either.
    """
    def wrapped(arrays):
        return model({**static, **arrays}, prior_scale=prior_scale)
    return wrapped


# ------------------------------------------------------------------ fingerprinting

def file_identity(path) -> dict:
    """Same content-sensitive identity age_run_map fingerprints with (imported, not copied)."""
    from src.model.age_run_map import _file_identity
    return _file_identity(Path(path))


def fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(canonical.encode()).hexdigest()


def source_identities(extra_files=()) -> list:
    here = Path(__file__).parent
    files = [here / name for name in MODEL_SOURCES] + [Path(__file__)] + [Path(p) for p in extra_files]
    return [file_identity(p) for p in files]


def versions() -> dict:
    return {"jax": jax.__version__, "numpyro": numpyro.__version__}


def check_model_drift(upstream_payload: dict, what: str = "MAP") -> None:
    """Refuse to use an upstream fit whose model sources or inputs have since changed.

    ``upstream_payload`` is the ``fingerprint_payload`` saved by age_run_map (or a VI
    checkpoint carrying the same ``sources``/``inputs`` lists). Set
    ``HOUFIN_HMC_ALLOW_MODEL_DRIFT=1`` to proceed anyway, e.g. after a comment-only edit.
    """
    from src.config_utils import load_age_model_config
    here = Path(__file__).parent
    old_src = {s["name"]: s["content_sha256"] for s in upstream_payload.get("sources", [])}
    changed = [name for name in MODEL_SOURCES
               if name in old_src and old_src[name] != file_identity(here / name)["content_sha256"]]
    old_inputs = {s["name"]: s for s in upstream_payload.get("inputs", [])}
    input_dir = Path(load_age_model_config()["input_dir"])
    for name, ident in old_inputs.items():
        path = input_dir / name
        if not path.exists() or file_identity(path)["content_sha256"] != ident["content_sha256"]:
            changed.append(f"input:{name}")
    if not changed:
        return
    msg = f"{what} was fitted to different model code/inputs; changed since: {changed}"
    if os.environ.get("HOUFIN_HMC_ALLOW_MODEL_DRIFT", "0") == "1":
        print(f"[warn] {msg} (proceeding: HOUFIN_HMC_ALLOW_MODEL_DRIFT=1)")
        return
    raise RuntimeError(msg + ". Refit MAP, or set HOUFIN_HMC_ALLOW_MODEL_DRIFT=1 if the "
                             "change cannot affect the density.")


# ------------------------------------------------------------------------ the MAP

def map_dir() -> str:
    """The MAP run directory, as age_run_map names it (run_names.map + profile suffix).

    Taken from age_run_map's own OUTPUT_DIR when inference runs at the MAP's
    precision. Otherwise age_run_map (which names by the CURRENT precision) cannot be
    asked, so the same rule is applied with MAP_PRECISION -- keep in step with
    age_run_map's _run_name block.
    """
    from src.model.age_run_map import OUTPUT_DIR
    if MAP_PRECISION == PRECISION:
        return OUTPUT_DIR
    from src.config_utils import load_age_model_config
    name = load_age_model_config()["run_names"]["map"].format(precision=MAP_PRECISION)
    profile = os.environ.get("HOUFIN_MAP_PROFILE", "standard")
    if profile != "standard":
        name = f"{name}_{profile}"
    return os.path.join(os.path.dirname(OUTPUT_DIR.rstrip(os.sep)), name)


def posterior_dir(pcfg: dict, key: str, at_map_precision: bool = False, **fmt) -> str:
    """results_dir/<run_names[key]>, keyed on the MAP run so different MAP fits never collide.

    When inference runs at a different precision than the MAP was fitted in, the name
    gets a ``__<precision>`` suffix so float32 and float64 runs from one MAP never share
    a directory. ``at_map_precision=True`` drops it (e.g. the MAP-precision probe).
    """
    name = pcfg["run_names"][key].format(map_run=os.path.basename(map_dir().rstrip(os.sep)), **fmt)
    if PRECISION != MAP_PRECISION and not at_map_precision:
        name = f"{name}__{PRECISION}"
    return results_dir(name)


def laplace_dir(pcfg: dict) -> str:
    """The probe holding laplace.npz: this precision's if it has one, else the MAP's."""
    own = posterior_dir(pcfg, "probe")
    if os.path.exists(os.path.join(own, "laplace.npz")):
        return own
    return posterior_dir(pcfg, "probe", at_map_precision=True)


def load_map():
    """Constrained MAP latents keyed by site name, plus the verified v2 checkpoint."""
    params, checkpoint = load_map_params(map_dir())
    return auto_delta_params_to_latents(params), checkpoint


# ------------------------------------------------------------- model + potential

def hide_deterministics(model):
    """Wrap ``model`` so MCMC never collects its deterministic sites.

    The age model emits full-grid, all-years arrays (simulated_density, Na_grid, Nj_grid,
    Q_flat, ...) as deterministics; MCMC collects deterministics per draw by default,
    which exhausts device memory within a few draws. Derived fields are recomputed from
    stored latents with ``Predictive`` instead.
    """
    def blocked(*args, **kwargs):
        with block(hide_fn=lambda site: site["type"] == "deterministic"):
            return model(*args, **kwargs)
    return blocked


def potential_with_args(model, kwargs: dict, init_values=None, rng_seed=0):
    """(z, potential(z, kwargs)): the potential with model kwargs as an ARGUMENT.

    Jit ``potential`` with the kwargs passed in, never closed over -- closed-over data
    is compiled in as constants (see split_data).
    """
    strategy = init_to_value(values=init_values) if init_values is not None else None
    kw = {} if strategy is None else {"init_strategy": strategy}
    info = initialize_model(jax.random.PRNGKey(rng_seed), model, model_kwargs=kwargs,
                            dynamic_args=True, **kw)
    gen = info.potential_fn

    def potential(z, kw_):
        return gen(**kw_)(z)
    return info.param_info.z, potential


def unconstrained_setup(model, data, init_values=None, rng_seed=0, kwargs=None):
    """``initialize_model`` at ``init_values`` (constrained; e.g. MAP latents).

    Returns ``(z, potential_fn, postprocess_fn)``: ``z`` is the UNCONSTRAINED latent dict
    NUTS samples in, ``potential_fn(z)`` is -log joint density there (Jacobians
    included), and ``postprocess_fn(z)`` maps back to constrained values. Pass
    ``kwargs`` instead of the default ``{"data": data, "prior_scale": 1}`` for an
    ``array_model``.
    """
    strategy = init_to_value(values=init_values) if init_values is not None else None
    kw = {} if strategy is None else {"init_strategy": strategy}
    info = initialize_model(
        jax.random.PRNGKey(rng_seed), model,
        model_kwargs=model_kwargs(data) if kwargs is None else kwargs, **kw,
    )
    return info.param_info.z, info.potential_fn, info.postprocess_fn


def flatten_sorted(z: dict):
    """Flatten ``z`` in numpyro's dense-mass-matrix order: sites sorted by name.

    numpyro builds a dense block for ``dense_mass=True`` as ``tuple(sorted(z))`` and
    ravels ``tuple(z[k] for k in sites)`` (hmc_util._initialize_mass_matrix), so a
    matrix indexed by this vector is directly usable as ``inverse_mass_matrix``.
    """
    names = tuple(sorted(z))
    flat, unravel_tuple = ravel_pytree(tuple(z[k] for k in names))

    def unravel(x):
        return dict(zip(names, unravel_tuple(x)))
    return names, flat, unravel


def dense_hessian(potential_fn, z: dict, chunk: int = 8, partial_path: str | None = None,
                  save_every: int = 8, log=print):
    """Dense Hessian of ``potential_fn`` at ``z`` by chunked Hessian-vector products.

    ``jax.hessian`` over the whole vector would materialize d forward-mode tangents of
    the full simulation at once; this does ``chunk`` at a time (forward-over-reverse)
    and, if ``partial_path`` is given, checkpoints completed columns so a wall-clock
    kill resumes rather than restarts. Returns (names, flat_z, H) with H symmetrized.
    """
    names, x0, unravel = flatten_sorted(z)
    d = x0.size

    def f(x):
        return potential_fn(unravel(x))

    grad_f = jax.grad(f)

    @jax.jit
    def hvp_batch(V):
        return jax.vmap(lambda v: jax.jvp(grad_f, (x0,), (v,))[1])(V)

    H = np.full((d, d), np.nan, dtype=np.float64)
    start = 0
    if partial_path and os.path.exists(partial_path):
        saved = np.load(partial_path)
        if saved["H"].shape == (d, d) and np.allclose(saved["x0"], np.asarray(x0)):
            H, start = saved["H"], int(saved["done"])
            log(f"[hessian] resuming at column {start}/{d}")
    eye = jnp.eye(d, dtype=x0.dtype)
    t0, n_chunks = time.time(), 0
    for lo in range(start, d, chunk):
        hi = min(lo + chunk, d)
        V = eye[lo:hi]
        if hi - lo < chunk:  # keep one compiled shape
            V = jnp.concatenate([V, jnp.zeros((chunk - (hi - lo), d), x0.dtype)])
        cols = np.asarray(hvp_batch(V), dtype=np.float64)[: hi - lo]
        H[:, lo:hi] = cols.T
        n_chunks += 1
        if n_chunks == 1 or hi == d or n_chunks % save_every == 0:
            rate = (time.time() - t0) / (hi - start)
            log(f"[hessian] {hi}/{d} columns, {rate:.2f} s/column, "
                f"eta {(d - hi) * rate / 60:.1f} min")
            if partial_path:
                tmp = partial_path + ".tmp.npz"
                np.savez(tmp, H=H, x0=np.asarray(x0), done=hi)
                os.replace(tmp, partial_path)
    return names, np.asarray(x0), 0.5 * (H + H.T)


def laplace_inverse_mass(H: np.ndarray, eig_floor: float):
    """A NUTS metric from a Hessian that need not be positive-definite (SoftAbs).

    Each principal axis gets width 1/sqrt(max(|lambda|, eig_floor)). A NEGATIVE
    eigenvalue means the point is a saddle along that axis -- there is no Gaussian
    there -- but |lambda| still measures how fast the density changes, which is what a
    step scale needs. Flooring negatives instead (the first version) gave strongly
    curved directions sd = 1/sqrt(eig_floor) = 10: e.g. alpha_a 4.5 against a prior sd
    of 0.5 at run_17's MAP, a metric that would have thrown NUTS along exactly the
    directions it cannot follow. eig_floor now only guards genuinely flat axes.
    """
    evals, evecs = np.linalg.eigh(H)
    clipped = np.maximum(np.abs(evals), eig_floor)
    inv_mass = (evecs / clipped) @ evecs.T
    report = {
        "d": int(H.shape[0]),
        "eig_min": float(evals[0]),
        "eig_max": float(evals[-1]),
        "n_negative": int((evals < 0).sum()),
        "n_below_floor": int((np.abs(evals) < eig_floor).sum()),
        "eig_floor": float(eig_floor),
        # |lambda| is not sorted even though lambda is: use max/min, not the ends.
        "condition_number_clipped": float(clipped.max() / clipped.min()),
        "eig_smallest_10": evals[:10].tolist(),
        "eig_largest_10": evals[-10:].tolist(),
        "median_adjacent_log_gap": float(np.median(np.diff(np.log(np.sort(clipped))))),
    }
    return 0.5 * (inv_mass + inv_mass.T), evals, evecs, report


def load_laplace(probe_dir: str):
    path = os.path.join(probe_dir, "laplace.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run the probe (34_model_hmc_probe.slurm) first")
    return dict(np.load(path, allow_pickle=False))


def jitter_unconstrained(z: dict, rng_key, scale: float, inv_mass: np.ndarray | None):
    """Overdispersed start: z + scale * L eps with L L^T = inv_mass (the Laplace covariance).

    Done in UNCONSTRAINED space, so every jittered start respects the supports (noise
    added to constrained MAP values can push a positive or unit-interval site out of
    its support before sampling begins).
    """
    if scale == 0:
        return z
    names, x0, unravel = flatten_sorted(z)
    eps = np.asarray(jax.random.normal(rng_key, x0.shape, dtype=x0.dtype))
    if inv_mass is None:
        step = eps
    else:
        step = np.linalg.cholesky(inv_mass + 1e-12 * np.eye(len(x0))) @ eps
    return unravel(x0 + scale * jnp.asarray(step, dtype=x0.dtype))


# ------------------------------------------------------------ chunked sampling

EXTRA_FIELDS = ("num_steps", "accept_prob", "energy", "potential_energy", "diverging",
                "adapt_state.step_size")
STATE_FILE = "mcmc_state.pkl"


def _host(tree):
    return jax.tree.map(lambda x: np.asarray(x), jax.device_get(tree))


def _grab(mcmc):
    samples = _host(mcmc.get_samples())
    extra = _host(mcmc.get_extra_fields())
    return samples, extra


def run_chunked_mcmc(kernel, *, out_dir: str, run_fingerprint: str, payload: dict,
                     num_warmup: int, num_samples: int, chunk: int, rng_key,
                     kwargs: dict, init_params=None, postprocess=None, log=print):
    """Warmup in one call, then sample in ``chunk``-draw pieces, checkpointing after each.

    Resume is exact: the pickled ``mcmc.last_state`` becomes ``post_warmup_state`` and
    sampling continues with its own rng key, so an interrupted-and-resumed chain draws
    the same values as an uninterrupted one. Each chunk is its own file
    (``chunks/chunk_XXXX.pkl``: samples, sampler extras, wall time), written before the
    state file advances, so a kill between the two just redoes that chunk.

    ``postprocess(samples) -> dict`` (optional) runs on each chunk before saving, e.g.
    NeuTra's warped-to-constrained transform.
    """
    if num_samples % chunk:
        raise ValueError(f"num_samples={num_samples} must be a multiple of chunk={chunk}")
    os.makedirs(os.path.join(out_dir, "chunks"), exist_ok=True)
    state_path = os.path.join(out_dir, STATE_FILE)
    # jit_model_args: kwargs are traced ARGUMENTS of the compiled sampler, not constants
    # baked into it (see split_data). Use an array_model so every kwarg leaf is an array.
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=chunk, num_chains=1,
                jit_model_args=True,
                progress_bar=os.environ.get("HOUFIN_HMC_PROGRESS", "1") == "1")

    if os.path.exists(state_path):
        with open(state_path, "rb") as fh:
            st = pickle.load(fh)
        if st["fingerprint"] != run_fingerprint:
            raise RuntimeError(
                f"Refusing incompatible HMC resume in {out_dir}: run fingerprint changed. "
                "Use HOUFIN_HMC_FRESH=1 to archive it and start over.")
        mcmc.post_warmup_state = st["state"]
        n_done = int(st["n_drawn"])
        log(f"[resume] {out_dir}: {n_done}/{num_samples} post-warmup draws on disk")
    else:
        log(f"[warmup] {num_warmup} iterations (compiles on the first)...")
        t0 = time.time()
        mcmc.warmup(rng_key, extra_fields=EXTRA_FIELDS, collect_warmup=True,
                    init_params=init_params, **kwargs)
        w_samples, w_extra = _grab(mcmc)
        save_pickle_atomic({"samples": w_samples, "extra": w_extra,
                            "seconds": time.time() - t0},
                           os.path.join(out_dir, "warmup.pkl"))
        _log_extra("warmup", w_extra, time.time() - t0, log)
        n_done = 0
        save_pickle_atomic({"format_version": 1, "fingerprint": run_fingerprint,
                            "payload": payload, "state": mcmc.post_warmup_state,
                            "n_drawn": 0}, state_path)

    while n_done < num_samples:
        t0 = time.time()
        # init_params is ignored once post_warmup_state is set, EXCEPT that a fresh process
        # must still build the kernel (sampler.init), which otherwise searches for a valid
        # random init -- and random unconstrained draws of this model can be non-finite.
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=EXTRA_FIELDS,
                 init_params=init_params, **kwargs)
        samples, extra = _grab(mcmc)
        seconds = time.time() - t0
        if postprocess is not None:
            samples = postprocess(samples)
        idx = n_done // chunk
        save_pickle_atomic({"samples": samples, "extra": extra, "seconds": seconds},
                           os.path.join(out_dir, "chunks", f"chunk_{idx:04d}.pkl"))
        mcmc.post_warmup_state = mcmc.last_state
        n_done += chunk
        save_pickle_atomic({"format_version": 1, "fingerprint": run_fingerprint,
                            "payload": payload, "state": mcmc.last_state,
                            "n_drawn": n_done}, state_path)
        _log_extra(f"chunk {idx} ({n_done}/{num_samples})", extra, seconds, log)
    log(f"[done] {out_dir}: {n_done} draws")


def _log_extra(label, extra, seconds, log):
    steps = np.asarray(extra.get("num_steps", [0]))
    div = np.asarray(extra.get("diverging", [False]))
    acc = np.asarray(extra.get("accept_prob", [np.nan]))
    ss = np.asarray(extra.get("adapt_state.step_size", [np.nan]))
    grads = int(steps.sum())
    log(f"[{label}] {seconds / 60:.1f} min, {grads} grads ({grads / max(seconds, 1e-9):.1f}/s), "
        f"steps/draw mean={steps.mean():.1f} max={steps.max()}, divergent={int(div.sum())}, "
        f"accept={np.nanmean(acc):.3f}, step_size(last)={float(ss.reshape(-1)[-1]):.3g}")


def archive_if_fresh(out_dir: str, env: str = "HOUFIN_HMC_FRESH", files=(STATE_FILE,)):
    """Move an existing run aside (never delete) when a fresh start is requested."""
    if os.environ.get(env, "0") != "1" or not os.path.isdir(out_dir):
        return
    if not any(os.path.exists(os.path.join(out_dir, f)) for f in files):
        return
    stamp = time.strftime("%Y%m%dT%H%M%S")
    archived = f"{out_dir.rstrip(os.sep)}.before_fresh_{stamp}"
    os.replace(out_dir, archived)
    print(f"[fresh] archived {out_dir} -> {archived}")


def load_chain(chain_dir: str):
    """Concatenate a chain's saved chunks: (samples, extra, total_seconds, n_chunks)."""
    paths = sorted(glob.glob(os.path.join(chain_dir, "chunks", "chunk_*.pkl")))
    if not paths:
        return None
    parts = []
    for p in paths:
        with open(p, "rb") as fh:
            parts.append(pickle.load(fh))
    cat = lambda key: {k: np.concatenate([np.asarray(c[key][k]) for c in parts])  # noqa: E731
                       for k in parts[0][key]}
    return cat("samples"), cat("extra"), float(sum(c["seconds"] for c in parts)), len(parts)


def chain_index() -> int:
    return int(os.environ.get("HOUFIN_HMC_CHAIN", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
