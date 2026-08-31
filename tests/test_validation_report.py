"""The figure suite's readers: each test pins a misreading the figures exist to prevent.

Every failure mode here is one this project has already paid for once. A ceiling that shares the
target's noise read as an achievable bound. A structurally-unavailable rung rendered as a zero. A
predictor table found by fixed depth, so a nested one was silently never inspected. A missing bar
drawn as a tie. These are the assertions, not "the figure ran".
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "viz")))

import _validation_load as L          # noqa: E402
import _validation_style as S         # noqa: E402
import validation_report as V         # noqa: E402


def _leaf(**preds):
    """A compare_predictors-shaped leaf: {name: pearson_r}."""
    return {"n": 500, "reference": "no_change",
            "predictors": {k: {"pearson_r": v, "rmse": 1.0 - v, "r2": v ** 2}
                           for k, v in preds.items()},
            "skill_vs": {k: v for k, v in preds.items()}, "unavailable": {}}


def _epochs(question="pair_convergence", **preds):
    leaf = _leaf(**preds)
    return {"types": {question: {"heldout": {"all_distances": {"dot": leaf}}}},
            "ceiling": {"types": {question: {"heldout": {"all_distances": {"dot": leaf}}}}}}


# --- the ceiling ---------------------------------------------------------------------------------

def test_the_room_uses_the_independent_ceiling_not_the_noise_sharing_one():
    """`esk_truncation` is the target passed through a rank-64 filter, noise included. Falling back
    to it inflates the room by roughly half (measured: 0.959 against 0.654), which is exactly how a
    pearson of 0.995 came to be quoted as a bound DESK 'fell short of'."""
    ep = _epochs(no_change=0.0, desk=0.2, esk_oracle_independent=0.65, esk_truncation=0.96)
    got = L.honest_room(ep, "pair_convergence")
    assert got["ceiling"] == "esk_oracle_independent"
    assert got["room"] == pytest.approx(0.65)
    assert got["share_of_room"] == pytest.approx(0.2 / 0.65)
    assert not got["shares_target_noise"]


def test_a_ceiling_that_shares_target_noise_is_flagged_rather_than_silently_used():
    """When only the noise-sharing row exists the number is still produced -- refusing outright
    would leave the row blank -- but the flag must be set so the figure can mark it."""
    ep = _epochs(no_change=0.0, desk=0.2, esk_truncation=0.96)
    got = L.honest_room(ep, "pair_convergence")
    assert got["ceiling"] == "esk_truncation"
    assert got["shares_target_noise"] is True


def test_a_spatial_question_is_refused_because_the_null_is_a_competitor_not_a_floor():
    """For two cells in one era the frozen-modern null supplies the whole modern spatial structure
    and is close to the right answer. Computing ceiling - null there and calling it room reads as
    'no model could do better here' when the comparison has no floor at all."""
    ep = _epochs("cross_cell_same_era_modern", no_change=0.87, desk=0.88,
                 esk_oracle_independent=0.95)
    got = L.honest_room(ep, "cross_cell_same_era_modern")
    assert "refused" in got and "not a no-information baseline" in got["refused"]
    assert "share_of_room" not in got


def test_a_narrow_room_is_marked_so_a_small_difference_is_not_read_as_skill():
    ep = _epochs("same_cell_over_time", no_change=0.62, desk=0.63,
                 esk_oracle_independent=0.70)
    got = L.honest_room(ep, "same_cell_over_time")
    assert got["room"] == pytest.approx(0.08)
    assert got["narrow"] is True


def test_room_is_read_from_the_ceiling_block_where_all_three_share_one_truth():
    """The `ceiling` block re-runs each question against a split-half truth, so its `desk` and
    `no_change` rows differ from the `types` block's. Taking a ceiling from one and a model from
    the other would put the numerator and the denominator on different targets."""
    ep = _epochs(no_change=0.0, desk=0.20, esk_oracle_independent=0.65)
    ep["types"]["pair_convergence"]["heldout"]["all_distances"]["dot"] = _leaf(
        no_change=0.0, desk=0.55, esk_truncation=0.99)          # a DIFFERENT truth
    got = L.honest_room(ep, "pair_convergence")
    assert got["model_r"] == pytest.approx(0.20)                 # not 0.55
    assert got["ceiling_r"] == pytest.approx(0.65)


# --- structural n/a is not a zero ----------------------------------------------------------------

def test_a_rung_with_no_rows_is_unavailable_and_never_a_win_rate_of_zero():
    """Under a spatial holdout a held-out cell has no training years of its own, so cell_trend
    cannot run. A 0% cell would read as 'never beats the null', which is a different claim."""
    rep = {"baseline_ladder": {"by_era": {"1970s": {
        "n": 100, "reference": "no_change",
        "predictors": {"no_change": {"n": 100, "median_err": 0.5},
                       "cell_trend": {"n": 0, "median_err": float("nan")},
                       "desk": {"n": 100, "median_err": 0.4}},
        "win_rate_vs": {"cell_trend": float("nan"), "desk": 0.8}}}}}
    cols, rungs, cell = L.ladder_table(rep)
    assert cell[("cell_trend", "1970s")] == {"unavailable": "structurally n/a"}
    assert cell[("desk", "1970s")]["win_rate"] == pytest.approx(0.8)


def test_a_missing_idw_bar_stays_nan_and_carries_the_no_bar_verdict():
    """NaN means 'no admissible bar' -- for an epoch inside the temporal holdout there are no
    training cells that year. Rendered as 0 it would invert the reading into 'the bar scored
    nothing', which is evidence of parity where none was measured."""
    rep = {"epoch_directions": {"windowed": {"pairs": {"1967_2025": {
        "n": 150, "model_dir_cos": 0.23, "idw_dir_cos": float("nan"),
        "null_dir_cos": 0.07, "verdict": "no-bar", "change_magnitude_ratio": 0.5,
        "err_magnitude_share": 0.3, "err_angular_share": 0.7,
        "magnitude_calibration": 2.1, "spacetime_idw_dir_cos": 0.17,
        "ceiling_dir_cos": 0.68, "room": 0.61, "share_of_room": 0.27}}}}}
    rows = L.epoch_direction_rooms(rep, "windowed")
    assert len(rows) == 1 and rows[0]["pair"] == "1967→2025"
    assert np.isnan(rows[0]["idw_dir_cos"])
    assert rows[0]["verdict"] == "no-bar"


# --- the walk ------------------------------------------------------------------------------------

def test_tidy_finds_a_predictor_table_at_any_depth():
    """The questions differ in shape -- neighbour questions nest split/distance/form while
    absolute_position nests only populations. Reading a fixed depth is what let the first
    completeness check inspect nothing while reporting green."""
    obj = {"a": {"b": {"c": _leaf(desk=0.4, no_change=0.1)},
                 "d": _leaf(desk=0.9)},
           "e": _leaf(desk=0.2)}
    rows = L.tidy(obj, "root")
    paths = {r["path"] for r in rows}
    assert len(paths) == 3, paths
    assert {r["predictor"] for r in rows} == {"desk", "no_change"}


def test_tidy_keeps_an_unavailable_predictor_as_a_row_with_its_reason():
    """An absent row and an unavailable row are different things: one is a gap in the report, the
    other is a stated finding about the data. Collapsing them loses the distinction the whole
    `unavailable` vocabulary exists to make."""
    leaf = _leaf(desk=0.4)
    leaf["unavailable"] = {"esk_oracle_independent": "cell has too few surveys in this era"}
    rows = L.tidy({"q": leaf}, "root")
    bad = [r for r in rows if r["predictor"] == "esk_oracle_independent"]
    assert len(bad) == 1 and bad[0]["unavailable"].startswith("cell has too few")
    assert L.short_reason(bad[0]["unavailable"]) == "no split-half"


def test_distance_curves_read_the_predictor_table_under_overall():
    """Each by_distance entry is a whole baseline_panel result, so its predictors sit under
    `overall`. Reading the top level yields an empty dict and an empty line plot -- which looks
    identical to 'this bin had no data'."""
    rep = {"baseline_ladder": {"temporal_buckets": {"b": {"by_distance": {
        "d1-10": {"graded_rows": 99, "overall": {
            "n": 99, "predictors": {"desk": {"n": 99, "median_err": 0.41},
                                    "cell_trend": {"n": 0, "median_err": float("nan")}},
            "win_rate_vs": {"desk": 0.7}}},
        "d11-20": {"overall": {"n": 50, "predictors": {"desk": {"n": 50, "median_err": 0.45}},
                               "win_rate_vs": {}}}}}}}}
    curves = L.distance_curves(rep, "b")
    assert [c["bin"] for c in curves] == ["d1-10", "d11-20"]     # sorted by lower edge
    assert curves[0]["predictors"] == {"desk": 0.41}             # the n=0 rung is dropped
    assert curves[0]["n"] == 99


# --- the visual vocabulary -----------------------------------------------------------------------

def test_every_ordered_predictor_has_a_colour_and_the_order_is_information_order():
    """A colour that means `desk` in one panel and the bar in the next makes every cross-figure
    comparison a re-lookup, which is the thing a twelve-figure suite cannot afford."""
    assert set(S.PREDICTOR_ORDER) <= set(S.PREDICTOR_COLORS)
    got = S.ordered({"desk", "no_change", "spacetime_idw", "esk_oracle_independent"})
    assert got == ["no_change", "spacetime_idw", "desk", "esk_oracle_independent"]


def test_the_two_non_competitors_say_so_in_their_own_label():
    """A reader who looks only at the legend must still not mistake truncation fidelity for a
    ceiling, or the frozen-modern null for a rival predictor."""
    assert "not a ceiling" in S.label("esk_truncation")
    assert "null" in S.label("no_change")
    assert set(S.NOT_A_COMPETITOR) == {"no_change", "esk_truncation"}


# --- end to end ----------------------------------------------------------------------------------

_REAL = os.path.join(os.path.dirname(__file__), "..", "results", "crossed", "processed",
                     "encoder", "desk_tempho_1995")


@pytest.mark.skipif(not os.path.exists(os.path.join(_REAL, "validate_report.json")),
                    reason="archived sweep outputs are gitignored")
def test_the_shipped_run_reproduces_its_own_numbers():
    """Guards the join against the file, not against a fixture: `pair_convergence` room must come
    back near 0.66 on the independent ceiling and near 0.96 on the noise-sharing one, and
    `cross_cell_same_era_modern` must be exactly 0 -- the harness's own zero check."""
    run = L.load_run(_REAL)
    honest = L.honest_room(run["epochs"], "pair_convergence")
    shipped = L.shipped_room(run["epochs"], "pair_convergence")
    assert honest["room"] == pytest.approx(0.656, abs=0.02)
    assert shipped["ceiling"] == "esk_truncation"
    assert shipped["room"] == pytest.approx(0.964, abs=0.02)
    zero = run["epochs"]["types"]["cross_cell_same_era_modern"]["heldout"]["all_distances"]["dot"]
    assert zero["skill_vs"]["desk"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not os.path.exists(os.path.join(_REAL, "validate_report.json")),
                    reason="archived sweep outputs are gitignored")
def test_every_figure_renders_from_a_shipped_run(tmp_path):
    run = L.load_run(_REAL)
    made = [fn(run, str(tmp_path)) for fn in V.FIGURES]
    made.append(V.fig09_spectrum(run, str(tmp_path), comp=None))
    drawn = [p for p in made if p]
    assert len(drawn) >= 7, f"only {len(drawn)} figures rendered"
    assert all(os.path.getsize(p) > 5000 for p in drawn)
    html = V.build_html(str(tmp_path), [(p, "c") for p in drawn], "meta")
    assert os.path.getsize(html) > sum(os.path.getsize(p) for p in drawn)


def test_a_run_directory_with_no_json_is_a_stated_absence_not_a_crash(tmp_path):
    """A run that has not had bbs-route-validate run yet is a normal state; each figure must skip
    with its inputs named rather than the loader refusing the whole directory."""
    run = L.load_run(str(tmp_path))
    assert run["report"] is None and run["routes"] is None and run["epochs"] is None
    assert all(fn(run, str(tmp_path)) is None for fn in V.FIGURES)


# --- the map, and the arrays it needed the validator to start saving -----------------------------

def _write_npz(path, with_direction=True, H=8, W=10):
    rng = np.random.default_rng(0)
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r, c = r.ravel(), c.ravel()
    arrays = dict(
        turn_rows=r, turn_cols=c, turnover_pred=rng.random(r.size),
        turnover_obs=rng.random(r.size),
        d_pred=np.zeros((0, 2)), d_obs=np.zeros((0, 2)), xy_hist=np.zeros((0, 2)),
        analog_hist_year=np.array([]),
        recon_rows=r, recon_cols=c,
        recon_err_desk=rng.random(r.size), recon_err_nochange=rng.random(r.size),
        ref_raster=np.array("/nonexistent/ref.tif"))
    if with_direction:
        arrays.update(dirchg_rows=r, dirchg_cols=c, dirchg_hist_year=np.full(r.size, 1975),
                      dir_cos=rng.uniform(-1, 1, r.size))
    np.savez_compressed(path, **arrays)


def test_the_map_draws_the_direction_field_the_npz_now_carries(tmp_path):
    """`directional_change_agreement` has always returned a per-cell dir_cos and nothing carried it
    to disk, so the one map that says WHERE the model gets direction right could not be drawn from
    any artifact a run wrote. The fourth panel is the point of adding those keys."""
    _write_npz(os.path.join(str(tmp_path), "validate_spacetime.npz"))
    run = L.load_run(str(tmp_path))
    made = V.fig12_maps(run, str(tmp_path))
    assert made and os.path.getsize(made) > 5000
    import matplotlib.image as mpimg
    assert mpimg.imread(made).shape[1] > 1500          # four panels, not three


def test_the_map_degrades_to_three_panels_on_an_npz_written_before_the_change(tmp_path):
    """An older npz has no dir_cos. The figure must still draw rather than raise -- a run made
    before the keys existed is a normal state, not a broken one."""
    _write_npz(os.path.join(str(tmp_path), "validate_spacetime.npz"), with_direction=False)
    run = L.load_run(str(tmp_path))
    made = V.fig12_maps(run, str(tmp_path))
    assert made and os.path.getsize(made) > 5000


def test_the_map_is_skipped_rather_than_faked_when_the_npz_is_absent(tmp_path):
    assert V.fig12_maps(L.load_run(str(tmp_path)), str(tmp_path)) is None


def test_the_validator_writes_the_direction_arrays_into_the_npz():
    """Pins the savez call itself: the keys the map reads must be the keys the validator writes,
    and a rename on either side has to fail here rather than in a figure six weeks later."""
    src = os.path.join(os.path.dirname(__file__), "..", "src", "community_encoder", "train_DESK",
                       "validate_spacetime.py")
    text = open(src, encoding="utf-8").read()
    call = text[text.index("np.savez_compressed("):text.index("ref_raster=np.array(ref_raster)")]
    for key in ("dirchg_rows=", "dirchg_cols=", "dirchg_hist_year=", "dir_cos="):
        assert key in call, f"{key} missing from validate_spacetime's npz"


# --- the figure must read a pre-fix and a post-fix run differently -------------------------------

def _recon_run(zspace_on_withheld):
    """A minimal `absolute_position` shaped report. `zspace_on_withheld=None` is the post-fix
    state: the same-year bar has no admissible source in a withheld year and is absent."""
    def _pop(n, **errs):
        return {"n": n, "reference": "no_change",
                "predictors": {k: {"n_scored": n, "median_err": v}
                               for k, v in errs.items() if v is not None},
                "unavailable": ({} if errs.get("zspace_idw") is not None
                                else {"zspace_idw": "reaches only 0 of %d rows" % n})}
    return {"report": {
        "baseline_ladder": {"common_holdout_years": [1966, 1967],
                            "temporal_buckets": {"unseen_year_unseen_cell": {"overall": {
                                "n": 100, "predictors": {"desk": {"n": 100, "median_err": 0.48},
                                                         "no_change": {"n": 100,
                                                                       "median_err": 0.54}}}}}},
        "zspace_reconstruction": {"absolute_position": {"populations": {
            "train": _pop(600, desk=0.41, no_change=0.47, zspace_idw=0.41, spacetime_idw=0.33),
            "heldout": _pop(110, desk=0.45, no_change=0.47, zspace_idw=0.43, spacetime_idw=0.54),
            "withheld_years": _pop(270, desk=0.45, no_change=0.55,
                                   zspace_idw=zspace_on_withheld, spacetime_idw=0.43)}}}},
        "routes": None, "epochs": None, "run_dir": ".", "label": "t"}


def test_the_figure_flags_a_pre_fix_run_by_detecting_the_impossible_number(tmp_path):
    """A finite same-year bar on a population that is ENTIRELY withheld years can only come from
    reading those years, so it is proof the run predates the fix. Detected from the file rather
    than from a config flag, so one figure reads archived and fresh runs correctly."""
    made = V.fig02_axes(_recon_run(0.4081), str(tmp_path))
    assert made
    cap = V.summary(V.fig02_axes)
    assert "HATCHED" not in cap                      # the flag belongs to the caption, not the doc
    import matplotlib.image as mpimg
    assert mpimg.imread(made).size > 0


def test_a_post_fix_run_is_not_flagged_and_says_why_the_bar_is_absent(tmp_path):
    """After the fix the bar is legitimately missing on withheld years. Rendering that as a
    contamination warning would be as wrong as the contamination was."""
    assert V.fig02_axes(_recon_run(None), str(tmp_path))


# --- the vocabulary the captions assume ----------------------------------------------------------

def test_the_glossary_covers_every_predictor_the_figures_can_draw():
    """A caption that says `spacetime_idw` and an output that never defines it is a poor trade,
    since both registries are already imported. Any predictor the style module can colour must be
    defined by one of them, so adding a bar without a role fails here."""
    defined = {name for name, _defs, _shared in L.glossary()}
    drawable = set(S.PREDICTOR_ORDER)
    assert not (drawable - defined), f"drawable but undefined: {sorted(drawable - defined)}"


def test_the_glossary_keeps_BOTH_definitions_where_a_name_means_two_things():
    """`no_change` is the cell's own OBSERVED community at the recent year in the z-space stream
    and DESK's OWN z frozen at the modern year in the similarity stream. Different objects, one
    name -- and the similarity stream needs the model-frozen form on purpose, because an
    observed-space null is scored in truth's own metric while DESK is not (worth -0.28 on a
    temporally-neutral model). Merging the two entries would hide exactly that."""
    entry = {n: (d, sh) for n, d, sh in L.glossary()}["no_change"]
    defs, shared = entry
    assert shared and len(defs) == 2
    streams = {st for st, _ in defs}
    assert any("z-space" in s for s in streams) and any("similarity" in s for s in streams)


def test_the_two_idw_bars_are_distinguished_by_what_each_may_borrow():
    """The distinction that decides how to read half these figures: one works WITHIN a year, the
    other across years. If the glossary ever stops saying so, the second panel is unreadable."""
    g = {n: " ".join(t for _s, t in d) for n, d, _sh in L.glossary()}
    assert "SAME year" in g["spatial_idw"] and "within a year" in g["spatial_idw"]
    assert "space AND time" in g["spacetime_idw"] or "JOINT space-time" in g["spacetime_idw"]


def test_the_html_carries_the_glossary_before_the_figures(tmp_path):
    """First, not an appendix: a reader without the zspace/spacetime distinction cannot read the
    extrapolation panel correctly, and by then it is too late."""
    png = os.path.join(str(tmp_path), "x.png")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.plot([0, 1]); fig.savefig(png); plt.close(fig)
    html = open(V.build_html(str(tmp_path), [(png, "cap")], "meta"), encoding="utf-8").read()
    assert "spatial_idw" in html and "spacetime_idw" in html
    assert html.index("Vocabulary") < html.index("<figure>")


# --- the rename ----------------------------------------------------------------------------------

def test_the_two_bars_are_named_on_the_axis_that_separates_them():
    """`zspace_idw` named the space its VALUES live in; `spacetime_idw` names the domain it
    INTERPOLATES OVER. Both bars interpolate observed z, so the first half of that pair carried no
    information and the two read as unrelated things side by side. They differ in exactly one
    respect -- whether time is a dimension they may borrow along -- and the names now say so."""
    from src.community_encoder.train_DESK.validate_baselines import LADDER_ROLES
    assert "spatial_idw" in LADDER_ROLES and "spacetime_idw" in LADDER_ROLES
    assert "zspace_idw" not in LADDER_ROLES
    assert "same year only" in S.label("spatial_idw")
    assert "across YEARS" in S.label("spacetime_idw")


def test_an_archived_report_still_reads_under_the_old_key(tmp_path):
    """34 archived run directories carry `zspace_idw`, and they are the only record of runs that
    cannot be reproduced without the data tree. Canonicalising on READ means one name in every
    figure without rewriting any of them."""
    assert L.canonical_predictor("zspace_idw") == "spatial_idw"
    assert L.canonical_predictor("spacetime_idw") == "spacetime_idw"
    legacy = _recon_run(0.4081)
    pops = legacy["report"]["zspace_reconstruction"]["absolute_position"]["populations"]
    for pop in pops.values():                       # rewrite the fixture back to the OLD key
        pop["predictors"] = {("zspace_idw" if k == "spatial_idw" else k): v
                             for k, v in pop["predictors"].items()}
    assert "zspace_idw" in pops["heldout"]["predictors"]
    assert V.fig02_axes(legacy, str(tmp_path)), "a pre-rename report must still render"


def test_the_alias_survives_into_the_flattened_rows():
    """`tidy` is what any future table or CSV would be built from, so the rename has to reach it
    -- otherwise old and new runs produce two different predictor names in one dataframe."""
    leaf = {"n": 10, "reference": "no_change",
            "predictors": {"zspace_idw": {"median_err": 0.4}, "desk": {"median_err": 0.3}},
            "skill_vs": {}, "unavailable": {}}
    got = {r["predictor"] for r in L.tidy({"q": leaf}, "root")}
    assert got == {"spatial_idw", "desk"}
