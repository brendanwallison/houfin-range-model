"""One visual vocabulary for the whole validation figure suite.

WHY THIS EXISTS. The suite draws the same ~10 predictors across a dozen figures. If a colour
means ``desk`` in one panel and ``spacetime_idw`` in the next, every cross-figure reading is a
re-lookup, and the comparisons the report is built around -- DESK against the honest bar, the bar
against the null -- stop being visible at a glance. So predictor identity is a CONSTANT here and
nothing downstream picks its own colour.

The ordering is not alphabetical either. ``PREDICTOR_ORDER`` follows the ladder's own ordering
principle (``validate_baselines.py:714``): each rung is handed strictly more information than the
one above it, so a row's position in a legend or a heatmap already says what it knows.

Two roles are drawn DIFFERENTLY rather than just differently coloured, because the report's own
docstrings say they are not competitors and a reader who treats them as one gets a wrong answer:

* ``no_change`` is a decomposition device -- DESK's own z frozen at the modern year. It shares
  DESK's functional so their difference isolates temporal variation. Drawn as a reference line.
* ``esk_truncation`` is TRUNCATION FIDELITY, not a ceiling: it is the target passed through a
  rank-64 filter, noise included, which is how its pearson of 0.995 came to be quoted as a bound
  DESK "fell short of". Drawn dotted and always outside a room bar, never as its endpoint.
"""
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

#: Information order, weakest to strongest, then the model, then the ceilings. Matches the ladder
#: comment in validate_baselines.py so a legend reads top-to-bottom as "knows less" -> "knows more".
PREDICTOR_ORDER = (
    "no_change",
    "distance_only",
    "cell_nearest_year",
    "cell_trend",
    "borrowed_delta",
    "spatial_idw",
    "spacetime_idw",
    "desk",
    "esk_oracle_independent",
    "esk_truncation",
)

#: Fixed across every figure. Warm = the model; cool = interpolation bars; grey = the null;
#: green = ceilings. Chosen so the two comparisons that matter (desk vs spacetime_idw, and either
#: against a ceiling) are the highest-contrast pairs on the page.
PREDICTOR_COLORS = {
    "desk": "#c1440e",                    # the model under test
    "no_change": "#8a8a8a",               # the null -- deliberately unsaturated
    "spacetime_idw": "#1f6fb4",           # THE honest bar
    "spatial_idw": "#6ba3d6",             # same-year spatial bar, a lighter sibling
    "zspace_idw": "#6ba3d6",              # its pre-rename name, still on 34 archived reports
    "spatial_interp": "#6ba3d6",          # the trainer-side printout's name for the same thing
    "borrowed_delta": "#7b5ea7",
    "cell_trend": "#2e8b74",
    "cell_nearest_year": "#7fbcab",
    "distance_only": "#b0a08c",           # the spatial floor
    "esk_oracle_independent": "#1a7f37",  # THE ceiling
    "esk_truncation": "#9dc183",          # fidelity, NOT a ceiling
}

#: Predictors that are not competitors, and how they are drawn instead of as a bar.
NOT_A_COMPETITOR = {
    "no_change": "reference line",
    "esk_truncation": "dotted tick",
}

#: The seed-to-seed spread measured on this trainer. Any cross-run or cross-arm figure draws it,
#: because a difference inside this band is not a result.
SEED_NOISE = 0.066

#: room < this -> the comparison cannot rank predictors (resolving_room's own threshold).
NARROW_ROOM = 0.15

DPI = 150


def color(name):
    """Colour for a predictor name; a stable grey for anything unregistered."""
    return PREDICTOR_COLORS.get(name, "#cccccc")


def ordered(names):
    """``names`` sorted into PREDICTOR_ORDER, with unknowns appended alphabetically."""
    known = [p for p in PREDICTOR_ORDER if p in names]
    rest = sorted(n for n in names if n not in PREDICTOR_COLORS)
    return known + rest


def label(name):
    """Display label. Marks the two non-competitors in the label itself, so a reader who looks
    only at the legend still cannot mistake them for bars."""
    if name == "no_change":
        return "no-change null"
    if name == "esk_truncation":
        return "esk truncation (fidelity, not a ceiling)"
    if name == "esk_oracle_independent":
        return "esk oracle (independent) — the ceiling"
    if name == "spacetime_idw":
        return "spacetime IDW — borrows across YEARS too"
    if name in ("spatial_idw", "zspace_idw"):
        return "spatial IDW — same year only"
    return name.replace("_", " ")


def hatch_unavailable(ax, x, y, w, h, reason, fontsize=6):
    """Draw a cross-hatched cell carrying its own REASON.

    A structurally-unavailable rung is not a zero and not a tie. Under a spatial holdout a
    held-out cell has no training years of its own, so ``cell_trend`` cannot run; under a temporal
    holdout there are no training points in the withheld years, so ``borrowed_delta`` cannot.
    Rendering either as a blank invites reading it as a failure, and as 0 inverts it -- so the
    cell is hatched and the reason is printed inside it.
    """
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="none", edgecolor="#bbbbbb",
                               hatch="////", linewidth=0.5, zorder=2))
    ax.text(x + w / 2, y + h / 2, reason, ha="center", va="center",
            fontsize=fontsize, color="#777777", zorder=3)


def caption(fig, text, y=0.005, fontsize=7.5):
    """Attach the report's OWN prose under a figure.

    These strings (`_note`, `why`, `read_this_way`, `_scale_warning`, `caveat`) were written into
    the validators specifically to stop a misreading, and today they are visible only inside a
    50 KB console log. Carrying them onto the figure is the cheapest way to keep the guard
    attached to the number it guards.
    """
    # Wrapped by hand rather than with `wrap=True`: matplotlib's wrapping measures against the
    # figure width in points and silently clips on wide figures, which is how a caption written to
    # prevent a misreading ends up half-visible.
    width = max(60, int(fig.get_figwidth() * 15))
    fig.text(0.5, y, textwrap.fill(" ".join(text.split()), width),
             ha="center", va="bottom", fontsize=fontsize, color="#555555")


def _layout(fig, rect):
    """tight_layout warns on twinned axes and then lays them out correctly anyway. Since every
    figure here passes an explicit rect, the warning is noise that would train a reader to ignore
    warnings from this module -- so it is silenced HERE and nowhere wider."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        fig.tight_layout(rect=rect)


def finish(fig, path, caption_text=None, tight_rect=None):
    """Save with the suite's defaults. Returns ``path`` so callers can collect it."""
    if caption_text:
        n_lines = len(textwrap.fill(" ".join(caption_text.split()),
                                    max(60, int(fig.get_figwidth() * 15))).splitlines())
        pad = min(0.30, 0.020 * n_lines + 0.035 * (6.0 / max(fig.get_figheight(), 1.0)))
        caption(fig, caption_text, y=0.004)
        _layout(fig, tight_rect or (0, pad, 1, 1))
    else:
        _layout(fig, tight_rect)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path
