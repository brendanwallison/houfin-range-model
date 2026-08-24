"""Tests for the generated sweep overlays and the pins that make the grid comparable.

Every failure named here produced, or would produce, a full run of plausible-looking numbers
rather than an error:

* ``_deep_merge`` REPLACES lists, so an overlay that sets ``states.streams`` to anything less
  than all six streams deletes the rest. The run then trains on a narrower covariate grid, fits
  its normalization to that grid, and reports numbers nothing downstream can contradict --
  no count, shape or set check compares the stream list against the config.
* The buffer is derived as ``kernel//2``, so the swept ``spatial_kernel`` variants would each
  be graded on a differently separated split unless a floor pins it. ``kernel=0`` would be
  graded with no separation at all, which is the optimism the buffer exists to remove.
* The direction anchors are pinned in the committed temporal overlays and null in the base
  config, so an unpinned production row would measure its diagnostics over a longer interval
  than every temporal row -- and the longer chord would read as skill.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.sweep import generate_overlays as G          # noqa: E402
from src.config_utils import load_config                  # noqa: E402


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _generate(tmp_path, *args):
    """Run the generator as a subprocess with HOUFIN_PROCESSED pointed into tmp_path."""
    root = str(tmp_path / "sweeps" / "hp")
    env = dict(os.environ, HOUFIN_PROCESSED=str(tmp_path))
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sweep",
                                                       "generate_overlays.py"),
                          "--root", root, *args],
                         capture_output=True, text=True, cwd=REPO, env=env)
    assert out.returncode == 0, out.stderr
    return root, json.load(open(os.path.join(root, "sweep_manifest.json")))


def test_every_overlay_keeps_all_six_streams(tmp_path):
    """The measured failure: a partial ``states.streams`` list silently deletes the others.

    ``_deep_merge`` recurses into dicts and replaces everything else, lists included. The
    ``ema_tau`` configurations are the only ones that touch streams, and they must restate all
    six -- the four that carry an ``ema_tau`` with the new value, and the two static ones
    unchanged. A one-element list would leave the run training on the climate stream alone,
    with mu/sd fitted to it and every reported number internally consistent.
    """
    root, man = _generate(tmp_path, "--stage", "1")
    assert len(man["runs"]) == 17, [r["run_id"] for r in man["runs"]]
    base = load_config(os.path.join(REPO, "config", "esk_desk_config.json"))
    want = [s["name"] for s in base["states"]["streams"]]
    assert len(want) == 6, want
    for r in man["runs"]:
        cfg = load_config(r["overlay"])
        got = [s["name"] for s in cfg["states"]["streams"]]
        assert got == want, f"{r['run_id']}: streams {got} != {want}"
        assert int(cfg["states"].get("warmup", base["states"]["warmup"])) == \
            int(base["states"]["warmup"]), f"{r['run_id']} lost states.warmup"
    print("all six streams survive every overlay")


def test_only_the_intended_key_moves(tmp_path):
    """Each configuration must differ from the baseline in exactly the knob it names.

    A generator is a single point of failure for 17 runs: one stray key applied to every
    overlay would shift the whole grid uniformly, which is invisible -- the runs stay
    comparable with each other and incomparable with everything measured before.
    """
    root, man = _generate(tmp_path, "--stage", "1")
    by = {r["config"]: load_config(r["overlay"]) for r in man["runs"]}
    b = by["base"]

    def diff(a, c, path=""):
        keys = set(a) | set(c)
        out = []
        for k in sorted(keys):
            if k.startswith("_"):
                continue
            p = f"{path}.{k}" if path else k
            av, cv = a.get(k), c.get(k)
            if isinstance(av, dict) and isinstance(cv, dict):
                out += diff(av, cv, p)
            elif av != cv:
                out.append(p)
        return out

    expected = {
        "sk0": {"desk.spatial_conv.enabled"},
        "sk1": {"desk.spatial_conv.kernel"},
        "sk3": {"desk.spatial_conv.kernel"},
        "tau0": {"states.streams", "paths.hist_dir"},
        "tau1": {"states.streams", "paths.hist_dir"},
        "tau4": {"states.streams", "paths.hist_dir"},
        "hl10": {"desk.output_ema.half_life_bounds"},
        "hl4": {"desk.output_ema.half_life_bounds"},
        "mw20": {"desk.weights.metric"},
        "mw60": {"desk.weights.metric"},
        "w64": {"desk.hidden_width"},
        "wps": {"desk.hidden_width"},
        "do005": {"desk.dropout"},
        "do02": {"desk.dropout"},
        "wd1e3": {"desk.weight_decay"},
        "mlp2": {"desk.mlp_expansion"},
    }
    for tag, want in expected.items():
        got = set(diff(b, by[tag])) - {"paths.desk_output_dir", "_sweep.run_id",
                                       "_sweep.created_utc", "_sweep.git_sha"}
        got = {g for g in got if not g.startswith("_sweep")}
        assert got == want, f"{tag}: changed {sorted(got)}, expected {sorted(want)}"
    print("each configuration moves exactly its own knob")


def test_the_pins_are_on_every_overlay(tmp_path):
    """Seed, buffer floor and both direction anchors, identical in every run of the grid.

    Without the seed pin two configurations are compared on different held-out regions. Without
    the buffer floor the kernel variants are compared across different val/train separations.
    Without the anchors the production row's rotation and direction diagnostics run over a
    longer interval than every temporal row's -- the failure the pinning comment in
    ``desk_tempho_*.json`` was written about, where a shrinking chord read as a worse model.
    """
    root, man = _generate(tmp_path, "--stage", "1")
    for r in man["runs"]:
        t = load_config(r["overlay"])["desk"]["trend"]
        assert t["seed"] == G.SPATIAL_SEED, r["run_id"]
        assert t["buffer_floor"] == G.BUFFER_FLOOR, r["run_id"]
        assert t["direction_anchor_year"] == G.DIRECTION_ANCHOR, r["run_id"]
        assert t["direction_withheld_anchor_year"] == G.WITHHELD_ANCHOR, r["run_id"]
        assert load_config(r["overlay"])["desk"]["selection_metric"] == G.SELECTION_METRIC
    print("pins present on all 17 overlays")


def test_the_buffer_floor_covers_the_widest_swept_kernel(tmp_path):
    """The floor must be >= kernel//2 for every kernel in the grid, or the widest run is refused.

    The trainer refuses a floor narrower than its own kernel needs, so this would be caught --
    but only after the job started, once per affected run. A generator that can produce a grid
    the trainer will reject is a generator that wastes queue time.
    """
    root, man = _generate(tmp_path, "--stage", "1")
    for r in man["runs"]:
        cfg = load_config(r["overlay"])["desk"]
        k = int(cfg["spatial_conv"]["kernel"]) if cfg["spatial_conv"]["enabled"] else 0
        assert cfg["trend"]["buffer_floor"] >= k // 2, (r["run_id"], k)
    print("buffer floor covers every swept kernel")


def test_the_production_cell_is_in_both_stages(tmp_path):
    """Stage 1 selects at ``t0_f100``; stage 2's trajectory must REACH that cell.

    Otherwise the configuration is chosen at a data amount the trajectory never revisits, and
    the production retrain's data amount is an extrapolation past the grid's last point rather
    than a point on it. The two stages must also emit the BYTE-IDENTICAL overlay for that cell,
    or resume cannot dedupe it and it is paid for twice under two different configurations.
    """
    root2, man2 = _generate(tmp_path, "--stage", "2", "--configs", "base,sk0")
    cells = {r["cell"] for r in man2["runs"]}
    assert "t0_f100" in cells, sorted(cells)
    assert len(man2["runs"]) == 2 * 4 * 4, len(man2["runs"])       # 2 configs x 4 temporal x 4 frac
    ov2 = json.load(open(next(r["overlay"] for r in man2["runs"]
                              if r["run_id"] == "sweep_t0_f100_sk0")))
    root1, man1 = _generate(tmp_path, "--stage", "1")
    ov1 = json.load(open(next(r["overlay"] for r in man1["runs"]
                              if r["run_id"] == "sweep_t0_f100_sk0")))
    ov1.pop("_sweep"); ov2.pop("_sweep")                # provenance carries a timestamp
    assert ov1 == ov2, (ov1, ov2)
    print("t0_f100 is shared by both stages and byte-identical")


def test_train_frac_varies_and_holdout_frac_never_does(tmp_path):
    """The data axis must move ``train_frac``, never ``holdout_frac``.

    ``holdout_frac`` is the VALIDATION fraction. Varying it to change the training amount would
    give each column a different, differently-sized held-out set, so the columns would not be
    on one metric -- and at 0.05 of the ~3,900 cells raw BBS reaches, the run would not clear
    ``min_val_cells`` and would refuse to start.
    """
    root, man = _generate(tmp_path, "--stage", "2", "--configs", "base")
    base_hf = load_config(os.path.join(REPO, "config",
                                       "esk_desk_config.json"))["desk"]["trend"]["holdout_frac"]
    fracs = set()
    for r in man["runs"]:
        t = load_config(r["overlay"])["desk"]["trend"]
        assert t["holdout_frac"] == base_hf, f"{r['run_id']} moved holdout_frac"
        fracs.add(t.get("train_frac"))
    assert fracs == {None, 0.95, 0.85, 0.70}, fracs
    print("train_frac is the data axis; holdout_frac is untouched")


def test_the_tau_configs_are_the_only_ones_needing_a_states_build(tmp_path):
    """``ema_tau`` is applied at STATE-BUILD time, so only those runs need their own states dir.

    The manifest is what the submit script preflights against. If a non-tau run claimed a
    states dir it would be blocked on a build it does not need; if a tau run claimed none it
    would silently read the production states and measure nothing at all -- the tau it asked
    for would be absent from the arrays and present in the overlay.
    """
    root, man = _generate(tmp_path, "--stage", "1")
    need = {r["config"] for r in man["runs"] if r["requires_states_dir"]}
    assert need == {"tau0", "tau1", "tau4"}, sorted(need)
    for r in man["runs"]:
        if not r["requires_states_dir"]:
            continue
        cfg = load_config(r["overlay"])
        taus = {s.get("ema_tau") for s in cfg["states"]["streams"]
                if s.get("ema_tau") is not None}
        assert len(taus) == 1, (r["run_id"], taus)
        assert str(taus.pop()) in r["requires_states_dir"], r["requires_states_dir"]
    print("only the tau configs require a states build, and the dir names their tau")


def test_stage_two_without_configs_is_refused(tmp_path):
    """The full 17 x 16 crossing is ~226 GPU-hours and was never the plan.

    Stage 2 exists to carry the FEW configurations stage 1 selected onto the data grid. A
    default that silently ran everything would be a 4x cost overrun with no error.
    """
    env = dict(os.environ, HOUFIN_PROCESSED=str(tmp_path))
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sweep",
                                                       "generate_overlays.py"),
                          "--root", str(tmp_path / "s"), "--stage", "2"],
                         capture_output=True, text=True, cwd=REPO, env=env)
    assert out.returncode != 0
    assert "needs --configs" in (out.stdout + out.stderr)
    print("stage 2 refuses to run the whole crossing by default")


def test_seed_replicates_need_exactly_one_config(tmp_path):
    """Stage 3 measures ONE configuration's seed-to-seed spread -- the threshold a knob must beat.

    Spreading the seed budget over several configurations measures nothing well enough to be a
    threshold, so the generator refuses rather than producing a grid that looks like stage 3.
    """
    env = dict(os.environ, HOUFIN_PROCESSED=str(tmp_path))
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sweep",
                                                       "generate_overlays.py"),
                          "--root", str(tmp_path / "s"), "--configs", "sk0,base",
                          "--seeds", "0,1,2"], capture_output=True, text=True,
                         cwd=REPO, env=env)
    assert out.returncode != 0
    assert "exactly one" in (out.stdout + out.stderr)
    root, man = _generate(tmp_path, "--configs", "sk0", "--seeds", "0,1,2")
    seeds = sorted(load_config(r["overlay"])["desk"]["trend"]["seed"] for r in man["runs"])
    assert seeds == [0, 1, 2], seeds
    print("stage 3 varies only the spatial seed, for exactly one configuration")


def test_the_per_stream_widths_are_parameter_matched_on_the_branches():
    """``wps`` must isolate ALLOCATION, not confound it with total capacity.

    The two encoder-branch layouts are exactly equal in parameter count -- the blocks hold
    ``16*h**2`` each and ``256**2 + 128**2 + 4*64**2 == 6*128**2`` -- so a difference between
    ``base`` and ``wps`` cannot be read as "more parameters won". The mixer and decoder are
    sized on ``sum(h)``, which does differ (768 vs 640), so the nets are NOT identical in
    total; this asserts the branch equality it actually has rather than the stronger claim.
    """
    ps = G.per_stream_widths(G.STREAM_NAMES)
    assert sum(w * w for w in ps) == 6 * 128 * 128, ps
    assert sum(ps) == 640 and 6 * 128 == 768        # mixer/decoder DO differ -- stated, not hidden
    print(f"per-stream widths {ps} are branch-parameter-matched against uniform 128")


def test_resolve_hidden_widths_refuses_a_length_mismatch():
    """Widths are POSITIONAL against ``dims``; a short list must not be padded.

    Padding would apply the default width to whichever streams fell off the end -- the same
    off-by-one class as the 94-of-96 species-column misalignment, invisible to every count and
    set check because the shapes would still all agree.
    """
    from src.community_encoder.train_DESK.model_arch import resolve_hidden_widths as R
    assert R(None, 6, 64) == [256] * 6              # historical max(128, latent*4)
    assert R(0, 6, 64) == [256] * 6                 # falsy scalar -> default, not width 0
    assert R(128, 6, 64) == [128] * 6
    assert R([256, 128, 64, 64, 64, 64], 6, 64) == [256, 128, 64, 64, 64, 64]
    with pytest.raises(ValueError, match="positional"):
        R([256, 128], 6, 64)
    with pytest.raises(ValueError, match=">= 1"):
        R([256, 128, 64, 64, 64, 0], 6, 64)
    print("per-stream width resolution refuses a length mismatch")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------- the packing and resume mechanics --------------------------------

SUBMIT = os.path.join(REPO, "scripts", "tacc", "submit_sweep.sh")
SLURM = os.path.join(REPO, "scripts", "tacc", "21_sweep_desk.slurm")


def test_three_packed_tasks_cannot_collide():
    """Each of the three tasks per node must get its own GPU, output dir, and scratch.

    A static check on the job script, because the collision is only reproducible on a 3-GPU node
    and every one of its forms is silent: three tasks sharing a GPU is a 3x slowdown and an OOM;
    three sharing an output dir means the last writer owns ``desk_meta.npz`` while all three
    logs report success; three sharing ``TMPDIR`` is a documented source of intermittent
    torch/matplotlib cache corruption.

    The GPU ordinal must come from ``SLURM_LOCALID`` (the rank's index WITHIN its node), never
    ``SLURM_PROCID`` (global across nodes) -- with ``-N 2`` the latter reaches 5 and would index
    past a 3-GPU node.
    """
    src = open(SLURM).read()
    assert "CUDA_VISIBLE_DEVICES=\"${SLURM_LOCALID}\"" in src, \
        "the GPU must be bound from SLURM_LOCALID, not PROCID"
    assert "SLURM_PROCID + 1" in src, "the joblist line must be picked by the GLOBAL rank"
    assert 'export TMPDIR="$OUT_DIR/tmp"' in src, "each task needs its own scratch"
    assert "MPLCONFIGDIR" in src, "matplotlib's cache is shared by default"
    assert 'export ESK_DESK_CONFIG="$OVERLAY"' in src, \
        "each task must load its OWN overlay, or all three train the same configuration"
    # Isolation must come from the OVERLAY, not from an environment root. env.sh exports
    # HOUFIN_PROCESSED unconditionally, so a root set in the submitting shell is silently
    # discarded by every job -- and all three tasks would then write to the production
    # directory, overwriting the checkpoint every downstream stage reads. The task body must
    # therefore never try to set those roots itself.
    inner = src.split("<<'INNER'")[1].split("INNER\n")[0]
    for bad in ("export HOUFIN_PROCESSED", "export HOUFIN_DATA"):
        assert bad not in inner, (
            f"the task body sets {bad}: env.sh overwrites it unconditionally, so isolation "
            f"would silently fall back to the production root")
    assert "$OUT_DIR" in inner and 'mkdir -p "$OUT_DIR"' in inner
    print("packed tasks are isolated by GPU, output dir and scratch")


def test_resume_keys_on_the_last_written_artifact():
    """Resume must key on ``run_summary.json``, not ``desk_meta.npz``.

    ``desk_meta.npz`` is written BEFORE the summary, so a job killed between the two writes
    leaves a meta behind for a run that never finished -- and keying on it would skip that run
    forever, leaving a permanent hole in the grid that looks like a completed cell. The summary
    is written last, so its presence is exactly "this run finished".
    """
    sub, slurm = open(SUBMIT).read(), open(SLURM).read()
    assert "run_summary.json" in sub and "run_summary.json" in slurm
    # and the trainer must really write it last
    trainer = open(os.path.join(REPO, "src", "community_encoder", "train_DESK",
                                "desk_training.py")).read()
    i_meta = trainer.index("desk_meta.npz")
    i_sum = trainer.index("run_summary.json")
    assert i_meta < i_sum, "run_summary.json must be written AFTER desk_meta.npz"
    # the double check inside the task exists too: a resubmission can race a job still running
    assert slurm.count("run_summary.json") >= 2, \
        "the per-task body must re-check completion, not trust the wrapper's snapshot"
    print("resume keys on the last-written artifact, and is re-checked per task")


def test_the_resume_predicate_skips_only_finished_runs(tmp_path):
    """Exercise the predicate itself against a half-finished grid.

    A run dir holding only ``desk_meta.npz`` (killed mid-write) must be RETRIED; one holding
    ``run_summary.json`` must be skipped.
    """
    root, man = _generate(tmp_path, "--stage", "1")
    runs = man["runs"]
    finished = runs[0]["desk_output_dir"]
    half = runs[1]["desk_output_dir"]
    os.makedirs(finished, exist_ok=True); os.makedirs(half, exist_ok=True)
    open(os.path.join(finished, "run_summary.json"), "w").write("{}")
    open(os.path.join(half, "desk_meta.npz"), "w").write("")        # killed mid-write
    pending = [r for r in runs
               if not os.path.exists(os.path.join(os.path.expandvars(r["desk_output_dir"]),
                                                  "run_summary.json"))]
    ids = {r["run_id"] for r in pending}
    assert runs[0]["run_id"] not in ids, "a finished run must be skipped"
    assert runs[1]["run_id"] in ids, "a run killed after desk_meta.npz must be retried"
    assert len(pending) == len(runs) - 1
    print(f"{len(pending)} of {len(runs)} pending; the half-written run is retried")


def test_the_submit_script_defaults_to_a_dry_run():
    """A sweep is tens of GPU-hours; the default must plan, not spend.

    Mirrors ``submit_juv_mdd_sweep.sh``, whose ``DRY_RUN`` defaults to 1 for the same reason.
    The script must also refuse a dirty tree: a source edit mid-sweep makes runs incomparable,
    so two configurations would differ by code as well as by config.
    """
    src = open(SUBMIT).read()
    assert 'DRY_RUN="${DRY_RUN:-1}"' in src, "DRY_RUN must default to 1"
    assert "--untracked-files=no" in src, \
        "the dirty check must ignore untracked files -- SLURM drops logs and CSVs here"
    assert "ALLOW_DIRTY" in src
    assert "SWEEP_TRAIN_STAGES" in src, "the stages a sweep run executes must be overridable"
    print("submit defaults to a dry run and refuses a dirty tree")


def test_a_missing_states_build_blocks_submission_but_not_a_dry_run():
    """``ema_tau`` needs its own yearly_states, and a missing one must stop a real submission.

    Pointing a tau run at a directory that does not exist is not a slow path -- it is a per-year
    FileNotFoundError, or worse a silent read of a states dir built at a DIFFERENT tau, in which
    case the run measures the production smoothing while its overlay claims otherwise. That
    silent case is what ``state_schema.json``'s ``ema_tau`` provenance closes.

    But it must NOT block ``DRY_RUN=1``. The one command whose entire job is "show me what you
    would do" was refusing to answer until hours of unrelated preprocessing had finished, so the
    packing plan could not be reviewed before committing to it.
    """
    src = open(SUBMIT).read()
    assert "requires_states_dir" in src and "yearly_states" in src
    assert "STATE-BUILD time" in src
    assert 'if [ "$DRY_RUN" != "1" ]; then' in src, \
        "the states gate must refuse a real submission but let a dry run through"
    assert "REFUSING to submit" in src
    # and it must point at the submitter that actually exists and forwards the overlay
    assert "submit_tau_states.sh" in src
    assert os.path.exists(os.path.join(REPO, "scripts", "tacc", "submit_tau_states.sh"))
    assert "submit_preprocess.sh" not in src, \
        "submit_preprocess.sh runs the wrong slurm script for a states build"
    print("states gate blocks submission, not planning")


def test_every_states_submitter_forwards_the_environment():
    """A states job that loses ``ESK_DESK_CONFIG`` builds into the PRODUCTION states dir.

    ``build_states`` resolves its output as ``load_config()["paths"]["hist_dir"]``, and
    ``load_config()`` with no overlay returns the committed config. So a wrapper missing
    ``--export=ALL`` does not fail -- it silently rebuilds production while the caller believes
    it built a tau variant, overwriting the covariates every other run in the grid was
    normalized against. Every other submit_*.sh wrapper in this tree already sets it; these two
    are the ones a sweep depends on.
    """
    for name in ("submit_states.sh", "submit_tau_states.sh"):
        src = open(os.path.join(REPO, "scripts", "tacc", name)).read()
        assert "--export=ALL" in src, f"{name} must forward the environment to the job"
    tau = open(os.path.join(REPO, "scripts", "tacc", "submit_tau_states.sh")).read()
    # the two guards that make a mistake here loud instead of silent
    assert "PRODUCTION states dir" in tau, "a tau build must refuse to target production"
    assert "STAGES=states" in tau, \
        "04_states defaults to the whole pre-encoder chain; those products are tau-independent"
    assert "manifest says" in tau, \
        "the overlay's resolved hist_dir must be checked against the manifest's"
    print("states submitters forward the environment and refuse production")
