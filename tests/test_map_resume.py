"""Resuming a MAP fit from its checkpoint must actually be able to take a step.

WHAT BROKE. ``age_run_map`` restores a pickled ``SVIState`` and goes straight into the update
loop. But numpyro keeps ``constrain_fn`` on the SVI OBJECT rather than inside ``SVIState``:
``SVI.__init__`` sets it to ``None`` and only ``init()`` builds it from the model's parameter
transforms. So a resumed run restored its optimizer into a half-constructed object and died on
the first ``update()`` with ``TypeError: 'NoneType' object is not callable`` -- raised deep in
numpyro's ``loss_fn``, several frames from anything naming a checkpoint.

The failure was well disguised. The fingerprint check passed, the step count was right, and the
log printed ``[resume] compatible checkpoint at absolute step 1300/1800`` immediately before
dying. Nothing in the .o file suggested a resume problem, the batch script exits 0, and the
whole point of RESUBMITS is that it runs unattended -- so a wall-clock kill silently ended the
fit at whatever step it had reached, with a checkpoint on disk that looked resumable and was
not. Every chained resume job has failed this way since RESUBMITS was written.

These tests pin the library fact the fix depends on, and the fix itself, without needing data or
a GPU.
"""
import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("numpyro")
pytest.importorskip("jax")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpyro                                                    # noqa: E402
import numpyro.distributions as dist                              # noqa: E402
from numpyro.infer import SVI, Trace_ELBO                         # noqa: E402
from numpyro.infer.autoguide import AutoDelta                     # noqa: E402


def _toy():
    """The smallest model with a constrained site, which is what constrain_fn exists for."""
    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))      # constrained: needs transform
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    y = jnp.array([0.1, -0.3, 0.7, 0.2])
    svi = SVI(model, AutoDelta(model), numpyro.optim.Adam(1e-2), loss=Trace_ELBO())
    return svi, y


def test_constrain_fn_is_none_until_init_is_called():
    """The library fact the whole bug rests on, pinned so a numpyro upgrade cannot hide it.

    If a future numpyro moves constrain_fn into SVIState or builds it in __init__, this test
    fails and the workaround in age_run_map can be removed deliberately rather than left as
    cargo.
    """
    svi, y = _toy()
    assert svi.constrain_fn is None, "SVI.__init__ no longer leaves constrain_fn unset"
    svi.init(jax.random.PRNGKey(0), y)
    assert svi.constrain_fn is not None, "SVI.init no longer populates constrain_fn"


def test_a_restored_state_cannot_step_without_init_but_can_with_it():
    """The exact failure, reproduced end to end: pickle a state, restore into a NEW SVI, step.

    Asserting the TypeError as well as the fix matters -- without it a test of the fix alone
    would pass against an SVI that had been initialized for some unrelated reason, and would not
    be testing the trap.
    """
    svi, y = _toy()
    state = svi.init(jax.random.PRNGKey(0), y)
    for _ in range(3):
        state, _ = svi.update(state, y)
    blob = pickle.dumps(state)                                     # what map_checkpoint.pkl holds

    # A new process reconstructs the SVI object and unpickles the state -- age_run_map's resume.
    cold, y2 = _toy()
    restored = pickle.loads(blob)
    with pytest.raises(TypeError, match="NoneType"):
        cold.update(restored, y2)

    # ...and with init() called first, purely for its side effect, the SAME restored state steps.
    warm, y3 = _toy()
    warm.init(jax.random.PRNGKey(0), y3)
    stepped, loss = warm.update(pickle.loads(blob), y3)
    assert jnp.isfinite(loss)


def test_age_run_map_initialises_the_svi_object_on_the_resume_path():
    """Pinned against the source: the resume branch must call svi.init, not only svi.update.

    A behavioural test would need the model, its data tree and a GPU, so this checks the one
    line that distinguishes a working resume from the broken one. The branch is identified by
    the checkpoint guard above it so this cannot be satisfied by the fresh-start init.
    """
    src = os.path.join(os.path.dirname(__file__), "..", "src", "model", "age_run_map.py")
    body = open(src, encoding="utf-8").read()
    head, _, tail = body.partition('print(f"[resume] compatible checkpoint')
    assert tail, "the [resume] log line moved; this test no longer locates the resume branch"
    _, _, resume_branch = head.partition('svi_state = ckpt["svi_state"]')
    assert "svi.init(" in resume_branch, (
        "the resume branch restores SVIState without calling svi.init(); numpyro leaves "
        "constrain_fn None on the object, so the first update() raises TypeError")
