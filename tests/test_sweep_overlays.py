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
    assert len(man["runs"]) == 19, [r["run_id"] for r in man["runs"]]
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
        "mw40": {"desk.weights.metric"},
        "mw60": {"desk.weights.metric"},
        "mw100": {"desk.weights.metric"},
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
    assert "ema_alpha" in tau, \
        "the EMA coefficient must be evaluated at submission, not hours into the build"
    print("states submitters forward the environment and refuse production")


def test_every_swept_tau_resolves_to_a_usable_ema_coefficient():
    """Each tau in the grid must produce a finite alpha in (0, 1] -- checked without a cluster.

    ``tau0`` is the case that mattered: ``1 - exp(-1/tau)`` raises ZeroDivisionError at exactly
    0, so that build would have failed partway through an 86-year run, after the queue wait,
    rather than at submission. alpha=1 is the correct limit and means every year passes through
    raw, which is what "no input smoothing" is.
    """
    from src.data.combine.streams import ema_alpha

    for spec in G.configurations():
        streams = (spec[2].get("states") or {}).get("streams")
        if not streams:
            continue
        for st in streams:
            if st.get("ema_tau") is None:
                continue
            a = ema_alpha(st["ema_tau"])
            assert 0.0 < a <= 1.0, (spec[0], st["name"], st["ema_tau"], a)
    assert ema_alpha(0) == 1.0, "tau=0 must mean no smoothing, not a crash"
    print("every swept ema_tau resolves to a usable coefficient")


def test_no_command_substitution_can_silently_abort_the_submission():
    """``set -euo pipefail`` + a redirected stderr makes a failing pipeline abort SILENTLY.

    Measured: ``EXISTING=$(squeue ... 2>/dev/null | wc -l)`` killed the whole script the moment
    squeue returned non-zero -- after the manifest was written, before anything was submitted,
    with no message at all, because the redirect had already swallowed the only clue. squeue
    exits non-zero for reasons unrelated to the check (a slurmctld timeout, an unrecognised
    partition), so every such probe has to absorb failure explicitly.

    Also asserts the two states are distinguished: "no jobs queued" and "could not ask" pass a
    cap check identically and only one of them is safe.
    """
    for name in ("submit_sweep.sh", "submit_tau_states.sh"):
        src = open(os.path.join(REPO, "scripts", "tacc", name)).read()
        assert "set -euo pipefail" in src, name
        for line in src.splitlines():
            line = line.strip()
            if not line.startswith(("EXISTING=$(", "AVAIL_GB=$(", "ONE_GB=$(", "DIRTY=$(")):
                continue
            assert "|| true" in line or line.startswith("DIRTY="), (
                f"{name}: `{line}` can abort the script silently under pipefail; "
                f"absorb the failure with `|| true` or an if-guard")
    sweep = open(SUBMIT).read()
    assert 'EXISTING="unknown"' in sweep, \
        "a squeue that could not be asked must not be reported as zero queued jobs"
    print("no probe can abort submission silently")


def test_the_su_per_gpu_hour_figure_is_computed_not_mangled():
    """``$(awk "BEGIN{printf \\"%.2f\\", ...}")`` inside a double-quoted echo loses its escapes.

    Measured: awk received ``BEGIN{printf %.2f, 3.0/3}``, printed a syntax error twice, and the
    cost figure came out blank -- the one number that justifies packing at all. Not
    platform-specific; awk is awk. The fix is ``awk -v`` with single quotes on its own line.
    """
    src = open(SUBMIT).read()
    assert "awk -v n=" in src and "'BEGIN{printf" in src, \
        "compute the figure with awk -v and single quotes, on its own line"
    # Non-comment lines only: the comment above the fix quotes the broken form on purpose, and
    # a check that cannot tell code from prose would forbid documenting the bug it prevents.
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert '\\"%.2f\\"' not in code, "escaped quotes inside a double-quoted echo get eaten"
    # and the documented cost claim must be the one the arithmetic produces
    out = subprocess.run(["awk", "-v", "n=3", 'BEGIN{printf "%.2f", 3.0/n}'],
                         capture_output=True, text=True)
    assert out.stdout == "1.00", out
    print("packed cost computes to 1.00 SU per GPU-hour against 3.00 unpacked")


def test_the_queue_caps_distinguish_rejection_from_queueing():
    """A nodes-per-user overrun QUEUES; a jobs-per-user overrun is REJECTED at submit time.

    Conflating them either blocks a submission that would have been fine, or lets the tail
    batches of a large stage be silently discarded by sbatch. The rates and limits come from
    hpc/lonestar6.md.
    """
    src = open(SUBMIT).read()
    assert "gpu-a100)       MAX_JOBS=8;  CAP_NODES=12" in src
    assert "REJECTS the overflow" in src, "the jobs cap must be a hard error"
    assert "That is queueing, not rejection -- nothing is lost." in src, \
        "the nodes cap must be a note, not an error"
    # an unknown queue must skip the check rather than guess a cap
    assert "MAX_JOBS=0;  CAP_NODES=0" in src and "don't guess" in src
    print("job cap errors, node cap notes, unknown queue skips")


def test_the_smoke_run_cannot_be_mistaken_for_the_real_baseline(tmp_path):
    """The instrumentation check must not share a run_id or a manifest with the grid.

    Its configuration IS the baseline, so the obvious implementation gives it
    ``sweep_t0_f100_base`` -- the same run_id as the real baseline run, in the same output dir.
    Its ``run_summary.json`` would then satisfy the resume marker and the grid would skip the
    500-epoch run it exists to gate. Writing over ``sweep_manifest.json`` has the mirror
    failure: the submit script would resume a one-run grid and report the other 16 complete.
    """
    root, _man = _generate(tmp_path, "--stage", "1")
    sroot = str(tmp_path / "sweeps" / "hp")
    env = dict(os.environ, HOUFIN_PROCESSED=str(tmp_path))
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sweep",
                                                       "generate_overlays.py"),
                          "--root", sroot, "--smoke", "30"],
                         capture_output=True, text=True, cwd=REPO, env=env)
    assert out.returncode == 0, out.stderr
    smoke = json.load(open(os.path.join(sroot, "smoke_manifest.json")))
    assert os.path.exists(os.path.join(sroot, "sweep_manifest.json")), \
        "the stage manifest must survive a smoke generation"
    assert len(smoke["runs"]) == 1
    r = smoke["runs"][0]
    assert r["run_id"] == "smoke30ep_base"
    assert "sweep_t0_f100" not in r["run_id"], "must not collide with the real baseline run"
    assert r["desk_output_dir"] != os.path.join(sroot, "sweep_t0_f100_base")
    cfg = load_config(r["overlay"])
    assert cfg["desk"]["epochs"] == 30
    # and it still carries every pin, so what it exercises is the real code path
    assert cfg["desk"]["selection_metric"] == G.SELECTION_METRIC
    assert cfg["desk"]["trend"]["buffer_floor"] == G.BUFFER_FLOOR
    assert len(cfg["states"]["streams"]) == 6
    print("the smoke run is isolated from the grid it gates")


def test_the_tau_builds_do_not_default_to_a_one_job_queue():
    """``development`` caps jobs-per-user at 1, and that cap REJECTS rather than queues.

    ``submit_states.sh`` defaults to ``development`` because it submits a single build. This
    script submits one per ema_tau -- three for the current grid -- so the inherited default
    would have taken one and silently lost the other two, with nothing in sbatch's output naming
    a cap as the reason. From hpc/lonestar6.md: development is 8 nodes / 2 h / 1 job per user,
    normal is 64 nodes / 48 h / 20 jobs per user, and BOTH bill at 1 SU per node-hour -- so the
    correct queue is also the free one.
    """
    src = open(os.path.join(REPO, "scripts", "tacc", "submit_tau_states.sh")).read()
    assert 'QUEUE="${QUEUE:-vm-small}"' in src, "must not inherit the one-job development default"
    assert "caps jobs-per-user at ONE" in src, \
        "why development is wrong here has to be recorded, not just avoided"
    assert "SCHEDULING LATENCY" in src, \
        "the deciding factor is time-to-start, not the charge rate -- record which"
    # vm-small is 16 cores / ~29 GB, below what build_states assumes; the settings 04_states.slurm
    # documents for that case must be applied, not left as a trap that OOMs an hour in
    for var in ("HOUFIN_STATES_READ_WORKERS", "HOUFIN_STATES_WORKERS", "HOUFIN_STATES_SAMPLES"):
        assert var in src, f"{var} must be set for a virtual node"
    assert "vm-small)    Q_MAX_JOBS=4" in src, "vm-small allows 4 jobs/user; 3 builds fit"
    assert "development) Q_MAX_JOBS=1" in src, \
        "an explicit QUEUE=development override must be caught, not silently truncated"
    assert "REJECTS the overflow" in src
    # and the single-build script may keep its own default -- one job fits in development
    single = open(os.path.join(REPO, "scripts", "tacc", "submit_states.sh")).read()
    assert 'QUEUE="${QUEUE:-development}"' in single, \
        "submit_states.sh submits one build; changing its default is out of scope here"
    print("tau builds default to normal; a development override is refused")


def test_the_production_overlay_holds_nothing_out_and_keeps_the_swept_budget(tmp_path):
    """The three properties the production retrain turns on, asserted rather than assumed.

    ``epochs`` must stay at the SWEPT value while ``stop_at_epoch`` does the stopping.
    ``_warmup_cosine`` is parameterised on the budget, so lowering ``epochs`` to the stopping
    point changes the learning rate at every preceding step and trains a different model rather
    than the same one stopped earlier -- the exact mistake §11 of the plan exists to prevent, and
    the one a hand-written overlay would make.

    Its manifest is separate, and its run_id carries the stopping epoch, so it can never collide
    with a grid run's resume marker.
    """
    env = dict(os.environ, HOUFIN_PROCESSED=str(tmp_path))
    sroot = str(tmp_path / "sweeps" / "hp")
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sweep",
                                                       "generate_overlays.py"),
                          "--root", sroot, "--production", "sk0", "137"],
                         capture_output=True, text=True, cwd=REPO, env=env)
    assert out.returncode == 0, out.stderr
    assert "INFERRED from the grid, not measured on itself" in out.stdout, out.stdout
    man = json.load(open(os.path.join(sroot, "production_manifest.json")))
    assert len(man["runs"]) == 1
    r = man["runs"][0]
    assert r["run_id"] == "production_sk0_stop137"
    assert not r["run_id"].startswith("sweep_"), "must not collide with a grid run"
    cfg = load_config(r["overlay"])["desk"]
    base = load_config(os.path.join(REPO, "config", "esk_desk_config.json"))["desk"]
    assert float(cfg["trend"]["holdout_frac"]) == 0.0
    assert int(cfg["trend"]["min_val_cells"]) == 0, "the val-cell floor must not block it"
    assert int(cfg["stop_at_epoch"]) == 137
    assert int(cfg["epochs"]) == int(base["epochs"]), \
        "epochs must stay at the swept budget, or the LR schedule is re-parameterised"
    # the winning configuration's own knob must survive into it
    assert cfg["spatial_conv"]["enabled"] is False, "the sk0 knob was lost"
    print("production overlay: nothing held out, swept budget kept, knob preserved")


def test_the_analysis_script_flags_a_false_leader():
    """A configuration that leads only because of a spike must not be reported as robust.

    The first version of this script compared the two top-4 SETS, which can coincide while the
    ordering is completely different -- so a run whose winning value came from one lucky
    evaluation was printed as "robust to the spike problem". That is exactly the
    plausible-looking summary the script exists to prevent, produced by the script itself. It now
    compares the two RANKINGS and names any configuration that moves >= 2 places.
    """
    src = open(os.path.join(REPO, "scripts", "sweep", "analyze.py")).read()
    assert "FALSE LEADER" in src
    assert "rank_k" in src and "rank_t" in src, "must compare rankings, not just top-N sets"
    assert "DO NOT carry the argmin top-4 forward as-is" in src
    # the provisional threshold must be labelled as such until stage 3 measures the seed spread
    assert "PROVISIONAL" in src
    print("the analysis script detects a spike-driven false leader")


def test_stop_at_shortens_the_run_not_the_schedule_and_spares_production(tmp_path):
    """``--stop-at`` must set stop_at_epoch while leaving ``epochs`` at the full budget.

    The measured optimum on the production cell is epoch 117 of 500, so most of every run is
    spent past it. Lowering ``epochs`` would re-parameterise the cosine and train a different
    model; ``stop_at_epoch`` halts with the schedule intact, so a truncated run stays comparable
    with the full-length one that measured the optimum.

    It must NEVER touch the production retrain, which carries its own stop_at_epoch from the
    stage-3 median. Overwriting that would stop the shipped model at the wrong epoch -- and
    since a no-holdout run has no validation curve, nothing downstream could reveal it.
    """
    env = dict(os.environ, HOUFIN_PROCESSED=str(tmp_path))
    sroot = str(tmp_path / "sweeps" / "hp")
    base = load_config(os.path.join(REPO, "config", "esk_desk_config.json"))["desk"]

    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sweep",
                                                       "generate_overlays.py"),
                          "--root", sroot, "--stage", "1", "--stop-at", "300"],
                         capture_output=True, text=True, cwd=REPO, env=env)
    assert out.returncode == 0, out.stderr
    man = json.load(open(os.path.join(sroot, "sweep_manifest.json")))
    for r in man["runs"]:
        cfg = load_config(r["overlay"])["desk"]
        assert int(cfg["stop_at_epoch"]) == 300, r["run_id"]
        assert int(cfg["epochs"]) == int(base["epochs"]), (
            f"{r['run_id']}: epochs must stay at the full budget or the LR schedule changes")

    # production keeps its own stopping epoch even when --stop-at is passed
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sweep",
                                                       "generate_overlays.py"),
                          "--root", sroot, "--production", "sk0", "137", "--stop-at", "300"],
                         capture_output=True, text=True, cwd=REPO, env=env)
    assert out.returncode == 0, out.stderr
    pman = json.load(open(os.path.join(sroot, "production_manifest.json")))
    pcfg = load_config(pman["runs"][0]["overlay"])["desk"]
    assert int(pcfg["stop_at_epoch"]) == 137, \
        "--stop-at must not override the production retrain's own stopping epoch"
    assert int(pcfg["epochs"]) == int(base["epochs"])
    print("--stop-at truncates the grid, keeps the schedule, and spares production")


def test_the_submit_wrapper_forwards_stop_at_to_the_generator():
    """The wrapper regenerates overlays, so it must forward --stop-at or silently drop it.

    Running the generator by hand with --stop-at and then submitting would have the wrapper
    rewrite every overlay WITHOUT it: the files on disk would say one thing and the submitted
    jobs would run full-length. Nothing would error, and the cost overrun would only show up in
    the SU bill.
    """
    src = open(SUBMIT).read()
    assert "SWEEP_STOP_AT" in src
    assert '--stop-at "$SWEEP_STOP_AT"' in src, "must reach the generator, not just be read"
    print("submit wrapper forwards --stop-at")


def test_the_states_preflight_checks_completeness_not_just_existence():
    """A states build killed partway leaves a directory that an isdir() test accepts.

    Out of disk or out of wall clock, and yearly_states exists holding half the timeline. The run
    then trains on however many years got written -- a silent change to the amount of data that
    cell sees, which is the sweep's own independent variable, and nothing downstream compares the
    year count against the timeline. The expected count comes from the PRODUCTION states dir so
    it tracks the timeline rather than drifting from a hardcoded number.

    The schema check matters for the same reason: without state_schema.json there is no ema_tau
    provenance, so the dir cannot be told apart from one built at a different tau.
    """
    src = open(SUBMIT).read()
    assert "INCOMPLETE" in src and "state_*.npz" in src, \
        "the preflight must count years, not just test isdir()"
    assert "n_prod" in src, "the expected count must come from the production dir"
    assert "NO state_schema.json" in src, "a dir with no ema_tau provenance must be refused"
    assert "absent or incomplete" in src
    print("states preflight checks completeness and provenance")


def test_resume_refuses_to_mix_runs_made_under_different_metric_settings():
    """A finished run is skippable only if it was produced under the settings now configured.

    metric_pairs changes the gradient's variance and therefore the optimization trajectory; the
    eval settings change the estimator the ranking is built from. Keying resume on
    run_summary.json alone would keep the 17 existing stage-1 runs -- made at 4,096 training pairs
    and a single eval draw -- and rank them against new ones: two estimators in one table, with
    nothing in the output saying so.
    """
    src = open(SUBMIT).read()
    assert "metric_pairs" in src and "eval_kernel_draws" in src
    assert "STALE" in src and "queued for rerun" in src
    # A MISSING key must count as stale. The first version wrote `if k in got and ...`, which made
    # the guard inert for exactly the runs it existed to catch: the 17 stage-1 runs predated the
    # recording of these settings, so their summaries had no such key, the comparison was skipped,
    # and all 17 reported "comparable and complete" while having been trained at 4,096 pairs on a
    # single eval draw. Absence of provenance is not evidence of comparability.
    assert "if k not in got or int(got[k]) != int(v)" in src, \
        "an unrecorded setting must count as stale, not as matching"
    assert "UNRECORDED" in src, "the report must distinguish absent from mismatched"
    # the per-task re-check must honour the flag, or the wrapper's reruns are skipped again
    slurm = open(SLURM).read()
    assert 'cut -f4' in slurm, "the task must read the stale flag from the joblist"
    assert '[ "${STALE:-0}" != "1" ]' in slurm, \
        "a stale run has a run_summary.json AND needs redoing; the file alone must not skip it"
    # and the trainer must record what makes runs comparable
    trainer = open(os.path.join(REPO, "src", "community_encoder", "train_DESK",
                                "desk_training.py")).read()
    for key in ('"metric_pairs":', '"eval_kernel_pairs":', '"eval_kernel_draws":'):
        assert key in trainer, key
    print("resume detects and reruns runs made under different metric/eval settings")


def test_the_submit_script_uses_no_process_substitution_with_a_heredoc():
    """A heredoc nested in `< <(...)` with a redirection is unparseable on bash 3.2.

    It fails at runtime with "bad substitution: no closing )" while passing `bash -n`, so the
    syntax check gives no warning -- and this repo is developed on macOS, whose /bin/bash is 3.2.
    """
    src = open(SUBMIT).read()
    code = [l for l in src.splitlines() if not l.strip().startswith("#")]
    assert not any("< <(" in l for l in code), \
        "use plain temp files; a heredoc inside a process substitution breaks on bash 3.2"
    assert "_RESUME_OUT" in src and "_RESUME_ERR" in src
    print("no heredoc nested in a process substitution")


def test_the_rank_curve_report_flags_an_inversion_rather_than_printing_zero():
    """An error curve that RISES with rank is the interesting case, and the first version hid it.

    The old message computed ``max(curve)/full - 1``, which assumes the curve falls with rank. On
    the first real run the curve rose -- r8=0.0109 up to r64=0.0121 -- so max and full were the
    same value and it printed "costs 0% against full rank", saying nothing about the fact that
    rank 8 beat full rank by 11%. That matters twice over: the higher components are adding
    variance to the dot product rather than signal, and the full-rank value selection runs on is
    therefore not the model's best kernel.
    """
    src = open(os.path.join(REPO, "scripts", "sweep", "check_run.py")).read()
    assert "INVERTED" in src
    assert "min(rc, key=lambda rv: rv[1])" in src, "must find the BEST rank, not the worst"
    assert "max(v for _r, v in rc)" not in src, "the old max/full-1 formula must be gone"
    # A phrase that is contiguous in the SOURCE -- the message is built from an f-string split
    # across lines, so asserting on the rendered sentence would fail on the line break rather
    # than on the behaviour.
    assert "make the kernel approximation WORSE" in src
    # and the monotone branch must still report the truncation cost
    assert "rank curve is monotone" in src
    print("the rank-curve report distinguishes an inversion from a monotone curve")


def test_the_checker_reports_per_epoch_cost_excluding_the_first_epoch():
    """Cost has to be measured, not assumed, and the first epoch is not representative.

    It carries one-off setup -- ~41 s against a ~6 s steady state -- so including it overstates a
    30-epoch run's per-epoch cost about sixfold, which is exactly the wrong direction when the
    question is whether an added diagnostic is affordable.
    """
    src = open(os.path.join(REPO, "scripts", "sweep", "check_run.py")).read()
    assert "s/epoch" in src and "excluding the first" in src
    assert "secs[1:]" in src, "the first epoch must be excluded from the median"
    assert "GPU-hours for a 500-epoch run" in src
    print("per-epoch cost is reported from the trajectory, first epoch excluded")


def test_the_noise_floor_is_not_borrowed_from_runs_outside_the_table():
    """A floor measured on one configuration must not be quoted for runs that lack it.

    Pooling over every run under the sweep root attributed a noise floor measured on an 8-draw
    smoke run to a table of single-draw runs with no error bar at all -- a number from a different
    instrument, printed as though it described these. The floor is a property of the estimator each
    run used, so it may only be quoted for the runs that used it, and a table where most runs lack
    it must report the floor as unavailable rather than borrow one.
    """
    src = open(os.path.join(REPO, "scripts", "sweep", "analyze.py")).read()
    assert "in_table" in src, "the floor must be pooled over the table's runs only"
    assert "carry an error bar" in src
    assert "borrowed" in src or "different configuration" in src
    # and a table mixing estimator settings must say so before anyone reads the ranking
    assert "DIFFERENT metric/eval settings" in src
    assert "different instruments, not just different configurations" in src
    print("the noise floor is scoped to the table, and mixed settings are flagged")


def test_a_threshold_far_above_the_resolvable_limit_is_explained():
    """Both directions matter, and only one was covered.

    A threshold BELOW the estimator's standard error cannot be met by evidence -- already warned.
    A threshold far ABOVE it is not a measurement limit at all: it is standing in for the
    seed-to-seed spread stage 3 measures, and until then margins between the two are resolvable by
    the instrument but unproven against training noise. Saying nothing there invites reading
    'NOT distinguishable' as 'the instrument cannot see it', which is the opposite of the truth.
    """
    src = open(os.path.join(REPO, "scripts", "sweep", "analyze.py")).read()
    assert "threshold > 5 * se" in src
    assert "SEED-TO-SEED spread it stands in for" in src
    assert "resolvable by the instrument but unproven" in src
    print("a threshold far above the noise floor is explained, not just accepted")


def test_env_sh_does_not_abort_every_stage_on_one_unavailable_root():
    """A single failing mkdir killed runs that never needed the directory.

    env.sh is sourced by pipeline.sh under `set -euo pipefail`, and it created all four data roots
    in one unconditional mkdir. When a $SCRATCH subtree was briefly unavailable
    ("mkdir: cannot create directory '/scratch/07980': Permission denied") that aborted the whole
    pipeline before any stage started -- killing the mw40 and mw100 runs, whose inputs (hist_dir,
    points_dir, z_dir) all live under $HOUFIN_PROCESSED on $WORK and never touch $HOUFIN_DATA.

    Warned rather than silenced: a stage that genuinely needs the missing root must still fail, at
    the point of use, naming what it wanted.
    """
    src = open(os.path.join(REPO, "scripts", "tacc", "env.sh")).read()
    assert 'mkdir -p "$HOUFIN_DATA" "$HOUFIN_PROCESSED"' not in src, \
        "one unconditional mkdir over all roots makes an unrelated root's absence fatal"
    assert "for _houfin_root in" in src, "each root must be created independently"
    assert "WARNING: cannot create" in src, "a failure must be reported, not swallowed"
    assert "2>/dev/null ||" in src, "the failure must not propagate under set -e"
    # and pipeline.sh must still be the strict script it was -- the fix belongs in env.sh, not in
    # relaxing error handling for every stage
    pl = open(os.path.join(REPO, "scripts", "tacc", "pipeline.sh")).read()
    assert "set -euo pipefail" in pl
    print("one unavailable root warns; the other roots and the run proceed")


def test_the_eigenbasis_table_reads_across_configurations_not_per_run():
    """The finding it exists for is a TREND that no single run shows.

    Every one of 19 stage-1 runs is individually flagged "rank curve inverted, spectrum not
    descending" -- identical warnings, so per-run output carries no signal about which
    configuration helps. Side by side the trend appears: the cost of using all 64 dimensions falls
    from 4.8% at metric weight 5 to 1.3% at 60, i.e. the weight makes more of the latent space
    carry signal while leaving the full-rank kernel value -- what selection reads -- almost
    unchanged.

    Strictly diagnostic: selection remains the held-out kernel alone, so this table must never
    feed the ranking.
    """
    src = open(os.path.join(REPO, "scripts", "sweep", "analyze.py")).read()
    assert "def eigenbasis_table(" in src
    assert "NOT selected on" in src, "the table must be labelled as diagnostic"
    assert "all-64 cost against metric weight" in src, "the cross-config trend is the point"
    assert "wider than" in src, "a best rank below 64 must be called out"
    # it must not touch the ranking: the only ranked column is the kernel
    rank_section = src[src.index("def stage1("):src.index("def eigenbasis_table(")]
    for key in ("eig_nesting", "eig_subspace", "eig_spectrum"):
        assert key not in rank_section, f"{key} must not reach the stage-1 ranking"
    print("the eigenbasis table is cross-configuration and never feeds selection")


def test_the_smoothed_ranking_rejects_a_spike_and_needs_no_rerun():
    """The robust ranking must be recomputable from the trajectories already on disk.

    14 of 19 stage-1 configurations moved >= 2 places between the argmin ranking and the
    spike-free tail ranking, one of them by 15 places -- so the argmin is noise-dominated and
    unusable for selection. The tool's own advice was to set desk.selection_smooth and rerun, which
    is ~17 GPU-hours. But smoothing only changes WHICH epoch is chosen, and every epoch's value is
    recorded, so the ranking can be recomputed for free. Only the saved checkpoints would come from
    the unsmoothed epoch, and stage 1 needs the ranking.

    The window is trailing, not centred, because the reported epoch is what a production retrain
    would be told to stop at -- it has to be an epoch the run actually reached.
    """
    import numpy as np

    from scripts.sweep.analyze import _smoothed_min

    rng = np.random.default_rng(0)
    v = [0.010 * (1 + 0.8 * np.exp(-e / 12)) + rng.normal(0, 0.0004) for e in range(1, 101)]
    v[11] = 0.0062                      # an isolated lucky evaluation, far from the basin
    series = [{"epoch": i + 1, "v": x} for i, x in enumerate(v)]

    raw_ep = min(range(len(v)), key=lambda i: v[i]) + 1
    sm_v, sm_ep = _smoothed_min(series, 5)
    assert sm_ep != raw_ep, "smoothing must reject the spike the raw argmin picks"
    assert sm_v > min(v), "the smoothed value must not inherit the spike's biased-low value"

    # window 1 is a no-op, so an unsmoothed run stays exactly comparable
    one_v, one_ep = _smoothed_min(series, 1)
    assert one_ep == raw_ep and one_v == pytest.approx(min(v))

    # an all-nan series is unavailable, not zero
    nan_v, nan_ep = _smoothed_min([{"epoch": 1, "v": float("nan")}], 3)
    assert nan_ep is None and not np.isfinite(nan_v)

    src = open(os.path.join(REPO, "scripts", "sweep", "analyze.py")).read()
    assert "--smooth" in src and "no rerun was needed" in src
    print("the smoothed ranking rejects a spike and is recomputed from saved trajectories")
