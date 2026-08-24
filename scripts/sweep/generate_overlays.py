"""Generate the DESK hyperparameter-sweep overlays and their manifest.

204 hand-written overlay files is not viable, and hand-writing even one of them is a trap:
``src.config_utils._deep_merge`` recurses into dicts and REPLACES everything else, lists
included, so an overlay that sets ``states.streams`` to a one-element list silently deletes
the other five streams. Anything touching a stream property therefore has to restate all six
in full -- trivial for a generator, a silent footgun by hand.

The grid is a CROSS of two things that are varied for different reasons:

* **configurations** -- one knob moved at a time from the committed baseline. A full crossing
  is 5,184 runs before the data grid even starts, so stage 1 varies knobs singly and only
  those that actually move the metric get crossed later.
* **data cells** -- how much data the run sees: four temporal-holdout widths x three
  training-block fractions.

Three things are pinned in EVERY overlay, and the sweep is not readable without them:

1. ``desk.trend.seed`` -- one spatial split for every configuration in a column, so two
   configurations are compared on identical held-out regions. Varied only at stage 3, whose
   whole job is to measure the seed-to-seed spread that a knob has to beat.
2. ``desk.trend.buffer_floor`` -- the buffer is otherwise derived as ``kernel//2``, so the
   ``spatial_kernel`` variants would each be graded on a differently separated split, and
   ``kernel=0`` would be graded with no separation at all.
3. ``desk.trend.direction_anchor_year`` / ``direction_withheld_anchor_year`` -- the existing
   temporal overlays pin these and the committed base leaves them null, so without pinning
   them here the production row's diagnostics would be measured over a different (and
   longer) interval than every temporal row's, and the difference would read as skill.

Usage::

    python scripts/sweep/generate_overlays.py --root $HOUFIN_PROCESSED/sweeps/desk_hp \\
        --stage 1 [--dry-run]
"""
import argparse
import copy
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config_utils import config_dir, load_config          # noqa: E402
from src.community_encoder.train_DESK.model_arch import resolve_hidden_widths  # noqa: E402

# --- the pins, applied to every overlay ------------------------------------------------------
SPATIAL_SEED = 0
# 2 = the widest kernel in the sweep (5) // 2. Every configuration is then graded on the same
# held-out regions with the same val/train separation. Raising the largest swept kernel means
# raising this too, or the widest variant's own receptive field would exceed the buffer -- which
# the trainer refuses outright rather than silently allowing.
BUFFER_FLOOR = 2
DIRECTION_ANCHOR = 1996             # trained-era CONTROL, trained in every temporal row
WITHHELD_ANCHOR = 1975              # the MEASUREMENT, withheld in every temporal row
# Selection moves to the kernel term because that is what the population model consumes: it
# reads Z through learned linear weights, so its covariance IS this similarity. z-MSE stays
# logged, and a disagreement between the two is itself a finding rather than a problem.
SELECTION_METRIC = "val_kernel"

# --- streams, restated in full wherever ema_tau moves ---------------------------------------
# Read from the committed config rather than duplicated here: a stream added to the config and
# not to a hardcoded copy would be deleted by the deep-merge in exactly the runs that touch
# streams, and every count would still agree.
_BASE = load_config(config_dir() / "esk_desk_config.json")
BASE_STREAMS = _BASE["states"]["streams"]
STREAM_NAMES = [s["name"] for s in BASE_STREAMS]
EMA_STREAMS = [s["name"] for s in BASE_STREAMS if s.get("ema_tau") is not None]


def streams_with_tau(tau):
    """All six streams restated, with ``ema_tau`` set to ``tau`` on the four that carry one.

    ``_deep_merge`` replaces lists wholesale, so a partial list here deletes the rest. The two
    static streams (soil, elevation) have no ``ema_tau`` and are restated unchanged rather than
    given one: ``build_streamer`` only reads it for ``per_variable``, and adding the key to a
    static stream would record a smoothing in the schema that nothing applied.
    """
    out = []
    for spec in BASE_STREAMS:
        spec = copy.deepcopy(spec)
        if spec.get("ema_tau") is not None:
            spec["ema_tau"] = tau
        out.append(spec)
    return out


def per_stream_widths(dims_names, wide=256, mid=128, narrow=64):
    """Per-stream widths, wide for climate and narrow for the static streams.

    Deliberately close to PARAMETER-MATCHED against the uniform ``h=128`` baseline, so the
    comparison isolates how capacity is ALLOCATED rather than confounding allocation with
    total capacity. The branch blocks hold ``16*h**2`` parameters each and
    ``256**2 + 128**2 + 4*64**2 == 6*128**2`` exactly, so the six encoder branches carry
    1,572,864 parameters either way. The mixer and decoder are sized on ``sum(h)``, which
    falls from 768 to 640, so the nets are not identical in total -- stated here because a
    "parameter-matched" claim that is only true of one component is the kind of half-true
    number this project has had to withdraw before.
    """
    return [wide if n == "climate" else (mid if n == "landuse" else narrow)
            for n in dims_names]


# --- configurations: one knob moved at a time -----------------------------------------------
# Every entry is (tag, human reason, overlay fragment). The baseline moves nothing, and exists
# so a knob's effect is measured against a run made under identical pins rather than against a
# historical number produced under different ones.
def configurations():
    ps = per_stream_widths(STREAM_NAMES)
    return [
        ("base", "the committed configuration under the sweep's pins", {}),
        # Tier 1 -- mechanisms the validation implicates. Validation found the model
        # under-moving in time by 2-4x and treating neighbours as co-moving 4.7x too strongly;
        # the spatial residual is the mechanism for the second.
        ("sk0", "spatial residual OFF -- the pure point-wise model, the most direct test of "
                "the over-smoothing finding", {"desk": {"spatial_conv": {"enabled": False}}}),
        ("sk1", "1x1 spatial residual: a per-cell channel mix with no neighbourhood at all",
         {"desk": {"spatial_conv": {"enabled": True, "kernel": 1}}}),
        ("sk3", "3x3 spatial residual (one 27 km ring) against the committed 5x5",
         {"desk": {"spatial_conv": {"enabled": True, "kernel": 3}}}),
        # Input smoothing. NOT a DESK-time knob: ema_tau is consumed at state-build time, so
        # each of these needs its own yearly_states build and its own hist_dir.
        ("tau0", "no input smoothing -- raw annual weather reaches the encoder",
         {"states": {"streams": streams_with_tau(0)}}),
        ("tau1", "input smoothing halved (tau 1, half-life ~0.7 yr)",
         {"states": {"streams": streams_with_tau(1)}}),
        ("tau4", "input smoothing doubled (tau 4, half-life ~2.8 yr)",
         {"states": {"streams": streams_with_tau(4)}}),
        # Output smoothing: bound the learned half-life, which settles near 10.8 yr.
        ("hl10", "output-EMA half-life capped at 10 yr (it settles near 10.8, so this is the "
                 "boundary of the current solution)",
         {"desk": {"output_ema": {"half_life_bounds": [1.0, 10.0]}}}),
        ("hl4", "output-EMA half-life capped at 4 yr -- forces the model to move in time",
         {"desk": {"output_ema": {"half_life_bounds": [1.0, 4.0]}}}),
        # Loss mix. The kernel term is what the downstream consumes and carries weight 5
        # against the stabilizing term's 64; both are per-element, so those are comparable.
        ("mw20", "kernel-loss weight 5 -> 20", {"desk": {"weights": {"metric": 20.0}}}),
        ("mw60", "kernel-loss weight 5 -> 60", {"desk": {"weights": {"metric": 60.0}}}),
        # Tier 2 -- capacity. ~2.5M parameters against ~12k training cells, generalizing along
        # the spatial axis.
        ("w64", "uniform width halved, 128 -> 64 (branch params scale as h^2)",
         {"desk": {"hidden_width": 64}}),
        ("wps", f"per-stream widths {ps} for streams {STREAM_NAMES} -- the branch blocks are "
                f"identical in every stream while only Linear(d,h) depends on input width, so "
                f"a uniform width gives a 3-channel static stream the same 262k-parameter "
                f"encoder as the 240-channel climate stream",
         {"desk": {"hidden_width": ps}}),
        ("do005", "dropout 0.1 -> 0.05 (input channel-group masking already supplies noise)",
         {"desk": {"dropout": 0.05}}),
        ("do02", "dropout 0.1 -> 0.2", {"desk": {"dropout": 0.2}}),
        ("wd1e3", "weight decay 1e-4 -> 1e-3", {"desk": {"weight_decay": 0.001}}),
        ("mlp2", "MLP expansion 4 -> 2 (halves every branch block)",
         {"desk": {"mlp_expansion": 2}}),
    ]


# --- data cells ------------------------------------------------------------------------------
# Four temporal rows, not three: the extra row is the PRODUCTION configuration, so the
# trajectory reaches the point it will be extrapolated to instead of stopping short of it.
TEMPORAL_ROWS = [
    ("t0", [], "production: no temporal holdout, all 60 supervised years"),
    ("t1975", list(range(1966, 1976)), "1966-1975 withheld (50 training years)"),
    ("t1985", list(range(1966, 1986)), "1966-1985 withheld (40 training years)"),
    ("t1995", list(range(1966, 1996)), "1966-1995 withheld (30 training years)"),
]
# Fractions of the AVAILABLE TRAINING blocks, applied after the split is drawn -- so the
# validation set is byte-identical in every cell and the columns are on one metric. Varying
# holdout_frac instead would move the val set, and at 0.05 of the ~3,900 cells raw BBS reaches
# it would not clear min_val_cells at all.
# f100 is included for the same reason there are four temporal rows and not three: it is the
# cell stage 1 selected at, so the trajectory REACHES the configuration it will be read off at
# instead of stopping one point short and extrapolating to it. Its four runs also duplicate
# stage 1's exactly -- same run_id, same overlay -- so the submit script's resume skips them
# rather than paying for them twice.
TRAIN_FRACS = [("f100", None), ("f95", 0.95), ("f85", 0.85), ("f70", 0.70)]
# The window withheld in EVERY temporal row, so the reported buckets are comparable: each
# row's own block is dominated by its shallow, cheap end (BBS coverage grows ~7x from 1966 to
# 1995, and a ~10.8 yr EMA half-life makes a 1-2 year reach close to interpolation).
COMMON_HOLDOUT = list(range(1966, 1976))


def literalize(path):
    """Rewrite an expanded path back to its ``${HOUFIN_PROCESSED}`` form where it starts there.

    Overlay paths are stored LITERAL and expanded by ``config_utils`` inside the job, because
    ``env.sh`` exports the roots unconditionally and a value baked in at generation time would
    be wrong the moment the roots move (a $SCRATCH purge, a different account). Applied to
    every generated path rather than to ``desk_output_dir`` alone: the ``ema_tau`` variants'
    ``hist_dir`` is just as much a data root, and having one of the two literal and the other
    absolute is how they drift apart.

    A path outside the root is returned unchanged rather than rejected -- pointing a sweep at
    a scratch directory is legitimate, it just cannot be relocated later.
    """
    root = os.environ.get("HOUFIN_PROCESSED")
    if not root:
        return path
    root = root.rstrip("/")
    if path == root:
        return "${HOUFIN_PROCESSED}"
    if path.startswith(root + "/"):
        return "${HOUFIN_PROCESSED}" + path[len(root):]
    return path


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def states_dir_for(cfg_frag, states_root):
    """The ``paths.hist_dir`` a configuration needs, or ``None`` to use the committed one.

    Only the ``ema_tau`` variants need their own, because ``ema_tau`` is applied when
    ``state_{year}.npz`` is written. Returning ``None`` rather than the base path keeps the
    overlay silent about ``hist_dir`` for the 14 configurations that do not care, so a run that
    should read the production states cannot be pointed elsewhere by a stray default.
    """
    streams = (cfg_frag.get("states") or {}).get("streams")
    if not streams:
        return None
    taus = sorted({s["ema_tau"] for s in streams if s.get("ema_tau") is not None})
    if len(taus) != 1:
        raise ValueError(f"expected one ema_tau across the restated streams, got {taus}")
    return f"{states_root}/states_tau{taus[0]}"


def build_grid(stage, configs=None):
    """``[(run_id, cell_tag, cfg_tag, holdout_years, train_frac, cfg_frag, reason)]``.

    Stage 1 is every configuration at the production cell only -- no temporal holdout and the
    full training set -- so 17 runs decide which knobs are worth carrying onto the data grid.
    Stage 2 crosses the surviving configurations (named by ``configs``) with all 12 cells.
    """
    all_cfg = {t: (r, f) for t, r, f in configurations()}
    if stage == 1:
        rows = [(t, (r, f)) for t, r, f in configurations()]
        out = []
        for tag, (reason, frag) in rows:
            out.append((f"sweep_t0_f100_{tag}", "t0_f100", tag, [], None, frag, reason))
        return out
    if stage == 2:
        if not configs:
            raise SystemExit("stage 2 needs --configs: the tags stage 1 selected. Running the "
                             "full 17 x 12 = 204 grid is 170 GPU-hours and was never the plan.")
        missing = [c for c in configs if c not in all_cfg]
        if missing:
            raise SystemExit(f"unknown config tag(s) {missing}; have {sorted(all_cfg)}")
        out = []
        for tag in configs:
            reason, frag = all_cfg[tag]
            for t_tag, ho, _t_why in TEMPORAL_ROWS:
                for f_tag, frac in TRAIN_FRACS:
                    out.append((f"sweep_{t_tag}_{f_tag}_{tag}", f"{t_tag}_{f_tag}", tag,
                                ho, frac, frag, reason))
        return out
    raise SystemExit(f"--stage must be 1 or 2; got {stage}")


def make_overlay(run_id, cfg_frag, holdout_years, train_frac, root, states_root, sha,
                 seed=SPATIAL_SEED, extra=None):
    """One overlay dict. ``${HOUFIN_PROCESSED}`` stays literal -- expanded inside the job."""
    ov = copy.deepcopy(cfg_frag)
    ov.setdefault("desk", {})
    ov["desk"]["selection_metric"] = SELECTION_METRIC
    trend = ov["desk"].setdefault("trend", {})
    trend["seed"] = int(seed)
    trend["buffer_floor"] = BUFFER_FLOOR
    trend["direction_anchor_year"] = DIRECTION_ANCHOR
    trend["direction_withheld_anchor_year"] = WITHHELD_ANCHOR
    trend["holdout_years"] = list(holdout_years)
    trend["common_holdout_years"] = list(COMMON_HOLDOUT) if holdout_years else []
    if train_frac is not None:
        trend["train_frac"] = float(train_frac)
    ov["paths"] = {"desk_output_dir": f"{root}/{run_id}"}
    sd = states_dir_for(cfg_frag, states_root)
    if sd:
        ov["paths"]["hist_dir"] = sd
    if extra:
        ov = _merge(ov, extra)
    ov["_sweep"] = {"run_id": run_id, "git_sha": sha,
                    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return ov


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def verify_overlay(path, expect_streams, expect_widths_len, expect_seed=SPATIAL_SEED):
    """Round-trip an overlay through ``load_config`` and assert what must have survived.

    The check that matters is the stream count. ``_deep_merge`` replaces lists, so an overlay
    touching ``states.streams`` deletes every stream it does not restate -- and the run would
    then train on a narrower covariate grid, fit its normalization to it, and report entirely
    plausible numbers. Nothing downstream compares the stream count against the config.
    """
    cfg = load_config(path)
    names = [s["name"] for s in cfg["states"]["streams"]]
    if names != expect_streams:
        raise SystemExit(f"{path}: merged streams are {names}, expected {expect_streams}. "
                         f"_deep_merge REPLACES lists -- restate all of them.")
    hw = cfg["desk"].get("hidden_width")
    widths = resolve_hidden_widths(hw, len(names), int(cfg["desk"]["latent_dim"]))
    if len(widths) != expect_widths_len:
        raise SystemExit(f"{path}: {len(widths)} widths for {expect_widths_len} streams")
    trend = cfg["desk"]["trend"]
    # expect_seed, not SPATIAL_SEED: stage 3 varies the seed ON PURPOSE -- measuring the
    # seed-to-seed spread is the only thing that turns "this knob helped" into a threshold a
    # knob has to clear. Asserting the constant here refused the very grid the pin exists for.
    for key, want in (("seed", int(expect_seed)), ("buffer_floor", BUFFER_FLOOR),
                      ("direction_anchor_year", DIRECTION_ANCHOR),
                      ("direction_withheld_anchor_year", WITHHELD_ANCHOR)):
        if trend.get(key) != want:
            raise SystemExit(f"{path}: trend.{key} is {trend.get(key)!r}, expected {want!r}. "
                             f"Every overlay must pin it or the columns are not comparable.")
    if cfg["desk"]["selection_metric"] != SELECTION_METRIC:
        raise SystemExit(f"{path}: selection_metric is {cfg['desk']['selection_metric']!r}")
    if int(cfg["desk"]["trend"]["buffer_floor"]) < int(
            cfg["desk"]["spatial_conv"]["kernel"]) // 2 and cfg["desk"]["spatial_conv"]["enabled"]:
        raise SystemExit(f"{path}: buffer_floor {trend['buffer_floor']} is narrower than "
                         f"kernel//2; the trainer would refuse this run.")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="sweep root; overlays and per-run output dirs live under it")
    ap.add_argument("--states-root", default=None,
                    help="where the per-ema_tau yearly_states builds live "
                         "(default: <root>/states)")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--configs", default="",
                    help="stage 2 only: comma-separated config tags stage 1 selected")
    ap.add_argument("--seeds", default="",
                    help="stage 3: comma-separated spatial seeds for ONE config "
                         "(with --configs naming exactly that one)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="emit ONE short baseline run of N epochs, in its own output dir, to "
                         "verify the instrumentation before spending the grid")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and write nothing")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expandvars(args.root)).rstrip("/")
    states_root = literalize(os.path.abspath(os.path.expandvars(
        args.states_root or f"{root}/states")).rstrip("/"))
    # The literal form of the sweep root, so a run's output dir is the SAME path the overlay
    # lives beside. basename(root) was wrong for any nested --root: overlays would be written
    # under sweeps/a/b while every run wrote to sweeps/b, and the manifest would agree with
    # neither.
    root_literal = literalize(root)
    sha = _git_sha()
    cfgs = [c for c in args.configs.split(",") if c]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    if args.smoke:
        # The plumbing check from the verification plan: does the trajectory JSONL appear, does
        # the val kernel pool build, is a best epoch recorded, does run_summary.json land. Its
        # own output dir so it cannot be mistaken for -- or overwrite -- the real baseline run,
        # whose run_id it would otherwise share and whose resume marker it would satisfy.
        #
        # NOT a model to draw any conclusion from. warmup_epochs is 20 against this budget, so
        # a 30-epoch run is almost entirely LR warmup and its metrics mean nothing; the point is
        # that every artifact exists and every column is populated.
        reason, frag = {t: (r, f) for t, r, f in configurations()}["base"]
        grid = [(f"smoke{args.smoke}ep_base", "smoke", "base", [], None, frag,
                 f"{args.smoke}-epoch instrumentation check -- artifacts only, not a result")]
        seeds = [SPATIAL_SEED]
        smoke_extra = {"desk": {"epochs": int(args.smoke)}}
    elif seeds:
        smoke_extra = None
        if len(cfgs) != 1:
            raise SystemExit("--seeds is the stage-3 seed replicate: name exactly one "
                             "--configs tag, since its whole purpose is to measure the "
                             "seed-to-seed spread of ONE configuration.")
        reason, frag = {t: (r, f) for t, r, f in configurations()}[cfgs[0]]
        grid = [(f"sweep_t0_f100_{cfgs[0]}_s{s}", "t0_f100", cfgs[0], [], None, frag, reason)
                for s in seeds]
    else:
        smoke_extra = None
        grid = build_grid(args.stage, cfgs)
        seeds = [SPATIAL_SEED] * len(grid)

    entries = []
    for i, (run_id, cell, tag, ho, frac, frag, reason) in enumerate(grid):
        seed = seeds[i] if len(seeds) == len(grid) else SPATIAL_SEED
        ov = make_overlay(run_id, frag, ho, frac, root_literal,
                          states_root=states_root, sha=sha, seed=seed, extra=smoke_extra)
        # Resolve the real (expanded) directory the job will actually write to, and refuse a
        # collision with production. A sweep run that overwrote $HOUFIN_PROCESSED/encoder/desk
        # would destroy the checkpoint every downstream stage reads, with no way back.
        real_out = os.path.join(root, run_id)
        prod = _BASE["paths"]["desk_output_dir"]
        if os.path.abspath(real_out) == os.path.abspath(prod):
            raise SystemExit(f"ABORT: {run_id} resolves to the PRODUCTION desk dir ({prod})")
        ov_path = os.path.join(root, "overlays", f"{run_id}.json")
        needs_states = states_dir_for(frag, states_root)
        entries.append({"run_id": run_id, "cell": cell, "config": tag, "reason": reason,
                        "overlay": ov_path, "desk_output_dir": real_out,
                        "holdout_years": list(ho), "train_frac": frac, "seed": seed,
                        "requires_states_dir": needs_states, "git_sha": sha,
                        "exit_status": None})
        if args.dry_run:
            print(f"  {run_id:<34} cell={cell:<10} states="
                  f"{'base' if not needs_states else os.path.basename(needs_states)}  {reason}")
            continue
        os.makedirs(os.path.dirname(ov_path), exist_ok=True)
        with open(ov_path, "w", encoding="utf-8") as fh:
            json.dump(ov, fh, indent=2)
        verify_overlay(ov_path, STREAM_NAMES, len(STREAM_NAMES), expect_seed=seed)

    tau_dirs = sorted({e["requires_states_dir"] for e in entries if e["requires_states_dir"]})
    print(f"\n{len(entries)} runs; {len(tau_dirs)} extra yearly_states build(s) required:")
    for d in tau_dirs:
        print(f"  {d}   <-- build BEFORE the runs that need it "
              f"(STAGES=states with paths.hist_dir set to this)")
    if not tau_dirs:
        print("  (none)")
    if args.dry_run:
        print("dry run: nothing written")
        return
    # A distinct filename: writing the smoke run over a stage manifest would leave the submit
    # script resuming a one-run grid and reporting the other 16 as complete.
    man = os.path.join(root, "smoke_manifest.json" if args.smoke else "sweep_manifest.json")
    with open(man, "w", encoding="utf-8") as fh:
        json.dump({"stage": args.stage, "git_sha": sha, "root": root,
                   "states_root": states_root,
                   "pins": {"seed": SPATIAL_SEED, "buffer_floor": BUFFER_FLOOR,
                            "direction_anchor_year": DIRECTION_ANCHOR,
                            "direction_withheld_anchor_year": WITHHELD_ANCHOR,
                            "selection_metric": SELECTION_METRIC},
                   "runs": entries}, fh, indent=2)
    print(f"manifest -> {man}")


if __name__ == "__main__":
    main()
