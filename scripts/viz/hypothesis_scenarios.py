#!/usr/bin/env python3
"""Schematic 2x2 illustrations of the pre-/post-invasion niche-shift hypotheses.

Three named zones -- West (native range), East (introduced range), Great
Plains (the historical gap between them) -- each get one of 4 categories:

    0  not niche, unoccupied   (background -- everywhere else)
    1  niche,     unoccupied
    2  niche,     occupied
    3  not niche, occupied     (the diagnostic category: occupied territory
                                the niche model says shouldn't be suitable --
                                evidence of a niche SHIFT rather than lagged
                                colonization of already-suitable habitat)

Four scenarios (2 hypotheses x pre/post invasion):
    H0 (niche conservatism): the Great Plains was suitable niche all along;
        post-invasion occupancy just catches up to it (category 2).
    H1 (niche shift): the Great Plains is occupied despite NOT being niche
        (category 3) -- the species expanded its realized niche.
    Both hypotheses share the same pre-invasion state (West niche+occupied,
    East niche+unoccupied, Great Plains not-niche+unoccupied) -- they only
    diverge in what "post-invasion Great Plains" looks like.

Zone geometries reuse the Great Plains polygon + row-wise west/east split
from overlay_great_plains_ebird.py (same cleaned GP polygon, same "west of
the band's edge at this latitude" logic), so these schematics line up with
the abundance/range figures already produced for this project.

Two additional constraints on the schematic, both enforced in `_load_zone_geoms`
/ `_zone_pieces`:
  - Zones are clipped to land (Natural Earth ne_10m_land, config
    coastline.land_source) so oceans read as empty, not a false "not niche"
    category, and the land boundary is drawn as a thin outline on every panel
    for geographic orientation.
  - Both "occupied" AND "niche" are capped at the real eBird range polygon
    (the SAME ``houfin_range_smooth_27km_2023.gpkg`` used in
    overlay_great_plains_ebird.py): it is the most expansive record of where
    the species is/has been, so anywhere outside it is assumed not niche,
    full stop, regardless of what a scenario's zone-level category would
    otherwise claim. A scenario can mark a zone niche and/or occupied, but
    only the part of that zone inside the observed range keeps that
    category; the rest drops to category 0 (not niche, unoccupied). Since
    the Great Plains zone is frequently entirely outside the range in the
    pre-invasion panels (and partially outside it even post-invasion), it
    would otherwise be visually indistinguishable from the background --
    every panel therefore also draws the Great Plains zone's own boundary
    as a thin dashed outline, independent of fill color.

Writes four candidate color/symbol encodings of the same 2x2 grid, so a
scheme can be picked by comparison rather than by guessing:
  docs/img/hypothesis_scheme_bivariate.png     (hue=niche, lightness=occupied)
  docs/img/hypothesis_scheme_flat.png          (4 unrelated qualitative colors)
  docs/img/hypothesis_scheme_color_hatch.png   (color=niche, hatch=occupied)
  docs/img/hypothesis_scheme_color_outline.png (color=occupied, border=niche)

Also writes a second layout (same 4 schemes) that draws the shared
pre-invasion state ONCE and branches into the two post-invasion outcomes via
arrows labeled H0/H1, since H0 and H1 are identical until the Great Plains is
colonized and showing that state twice side by side is redundant:
  docs/img/hypothesis_branch_bivariate.png
  docs/img/hypothesis_branch_flat.png
  docs/img/hypothesis_branch_color_hatch.png
  docs/img/hypothesis_branch_color_outline.png
"""
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIZ_DIR = os.path.join(REPO_ROOT, "scripts", "viz")
for p in (REPO_ROOT, VIZ_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import overlay_great_plains_ebird as base  # noqa: E402  (path set up above)

OUT_DIR = os.path.join(REPO_ROOT, "docs", "img")

CATEGORY_LABELS = {
    0: "not niche, unoccupied",
    1: "niche, unoccupied",
    2: "niche, occupied",
    3: "not niche, occupied",
}

SCENARIOS = [
    ("H0", "Pre-invasion", {"west": 2, "east": 1, "gp": 0}),
    ("H0", "Post-invasion", {"west": 2, "east": 2, "gp": 2}),
    ("H1", "Pre-invasion", {"west": 2, "east": 1, "gp": 0}),
    ("H1", "Post-invasion", {"west": 2, "east": 2, "gp": 3}),
]


def _west_of_gp_geom(west_edge, transform, ny, box_minx):
    """Mirror of base._east_of_gp_geom, toward box_minx instead of box_maxx."""
    boxes = []
    for i in range(ny):
        y_top = transform.f + i * transform.e
        y_bot = y_top + transform.e
        x1 = west_edge[i]
        if x1 > box_minx:
            boxes.append(shapely_box(box_minx, min(y_top, y_bot), x1, max(y_top, y_bot)))
    return unary_union(boxes)


def _load_zone_geoms():
    cfg = base.load_data_config()
    ref = base.regrid.load_ref(cfg)
    box_minx, box_miny, box_maxx, box_maxy = cfg["grid"]["box_bounds"]
    project_crs = cfg["grid"]["box_crs"]
    box_bounds = (box_minx, box_miny, box_maxx, box_maxy)
    box_geom = shapely_box(box_minx, box_miny, box_maxx, box_maxy)

    transform = ref.rio.transform()
    ny, nx = ref.shape[-2:]

    ecoregions = base.gpd.read_file(base.ECOREGION_SHP).to_crs(project_crs)
    great_plains_raw = ecoregions[ecoregions["NA_L1NAME"] == "GREAT PLAINS"]
    gp_clean_tol = 2 * cfg["grid"]["target_res_m"]
    gp_geom = base._clean_great_plains_geom(
        unary_union(great_plains_raw.geometry), gp_clean_tol
    )

    _, _, _, west_edge, east_edge = base._row_wise_gp_zones(
        gp_geom, transform, ny, nx, box_bounds
    )
    west_geom = _west_of_gp_geom(west_edge, transform, ny, box_minx)
    east_geom = base._east_of_gp_geom(east_edge, transform, ny, box_maxx)
    background_geom = box_geom.difference(unary_union([west_geom, gp_geom, east_geom]))

    # Ocean cutout: clip every zone to land (same Natural Earth source the
    # project's own land mask uses), and keep the land boundary to draw as an
    # outline for orientation.
    land_path = os.path.join(cfg["datasets_root"], cfg["coastline"]["land_source"])
    land_gdf = base.gpd.read_file(land_path).to_crs(project_crs)
    # Reprojecting Natural Earth's global land polygons can leave tiny
    # self-intersections that make unary_union raise; buffer(0) is the
    # standard shapely fix-up (re-noding without changing the shape).
    land_geom = unary_union(land_gdf.geometry.buffer(0))
    land_geom = land_geom.intersection(box_geom)

    zones = {
        "west": west_geom.intersection(land_geom),
        "gp": gp_geom.intersection(land_geom),
        "east": east_geom.intersection(land_geom),
        "background": background_geom.intersection(land_geom),
    }

    # Real observed occupancy -- the ceiling every scenario's "occupied"
    # claim gets capped against (see _zone_pieces).
    houfin_range = base.gpd.read_file(base.RANGE_GPKG, layer="range").to_crs(project_crs)
    range_geom = unary_union(houfin_range.geometry)

    return zones, project_crs, box_bounds, land_geom, range_geom


def _draw_zone(ax, geom, facecolor, edgecolor="none", hatch=None, linestyle="solid", linewidth=1.0):
    if geom is None or geom.is_empty:
        return
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        patch = mpatches.PathPatch(
            _shapely_to_path(poly), facecolor=facecolor, edgecolor=edgecolor,
            hatch=hatch, linestyle=linestyle, linewidth=linewidth,
        )
        ax.add_patch(patch)


def _shapely_to_path(poly):
    from matplotlib.path import Path
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        coords = list(ring.coords)
        verts += coords
        codes += [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


# A scenario's category claim (niche and/or occupied) only holds inside the
# real eBird range polygon -- the most expansive record of where the species
# is/has been. Outside it, both axes collapse to "not niche, unoccupied"
# (category 0): niche claims aren't validated there, and occupancy obviously
# isn't either. Category 0 itself needs no capping (it's already the "outside
# the range" default).
def _zone_pieces(zone_geom, cat, range_geom):
    if cat == 0 or zone_geom.is_empty:
        return [(zone_geom, cat)]
    inside = zone_geom.intersection(range_geom)
    outside = zone_geom.difference(range_geom)
    pieces = []
    if not inside.is_empty:
        pieces.append((inside, cat))
    if not outside.is_empty:
        pieces.append((outside, 0))
    return pieces


def _panel(ax, zones, categories, style_fn, box_bounds, land_geom, range_geom):
    for zone_name in ("background", "west", "gp", "east"):
        cat = 0 if zone_name == "background" else categories[zone_name]
        for piece_geom, piece_cat in _zone_pieces(zones[zone_name], cat, range_geom):
            facecolor, edgecolor, hatch, linestyle, linewidth = style_fn(piece_cat)
            _draw_zone(
                ax, piece_geom, facecolor, edgecolor=edgecolor, hatch=hatch,
                linestyle=linestyle, linewidth=linewidth,
            )
    # Great Plains outline, independent of fill color -- when GP falls
    # outside the observed range it collapses to the same category-0 color
    # as the background and would otherwise disappear entirely.
    _draw_zone(
        ax, zones["gp"], facecolor="none", edgecolor="dimgray",
        linestyle=(0, (3, 2)), linewidth=1.0,
    )
    # Coastline outline for orientation -- drawn last so it sits on top.
    _draw_zone(ax, land_geom, facecolor="none", edgecolor="black", linewidth=0.6)
    box_minx, box_miny, box_maxx, box_maxy = box_bounds
    ax.set_xlim(box_minx, box_maxx)
    ax.set_ylim(box_miny, box_maxy)
    # Unlike imshow/geopandas .plot() (which default to equal aspect), raw
    # PathPatch axes default to "auto" and stretch to fill whatever box
    # they're given -- without this the continent gets visibly distorted
    # whenever a panel's width:height isn't the map's true aspect ratio.
    ax.set_aspect("equal")
    ax.set_axis_off()


def _render_scheme(zones, box_bounds, land_geom, range_geom, style_fn, out_png, legend_style_fn):
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 5.8))
    for ax, (hyp, stage, categories) in zip(axes.flat, SCENARIOS):
        _panel(ax, zones, categories, style_fn, box_bounds, land_geom, range_geom)
        ax.set_title(f"{hyp} -- {stage}", fontsize=11)

    handles = []
    for cat in range(4):
        facecolor, edgecolor, hatch, linestyle, linewidth = legend_style_fn(cat)
        handles.append(mpatches.Patch(
            facecolor=facecolor, edgecolor=edgecolor if edgecolor != "none" else "black",
            hatch=hatch, linestyle=linestyle, linewidth=max(linewidth, 1.0),
            label=CATEGORY_LABELS[cat],
        ))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    # Extra column gap (vs. the branching layout's default) so a row arrow +
    # action caption fits between the pre/post columns without crowding them.
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.subplots_adjust(wspace=0.55)

    # Row arrows: pre-invasion (left column) -> post-invasion (right column),
    # each labeled with the actual ecological change on that row (see
    # ARROW_ACTIONS) -- what turns the shared pre-invasion state into H0's or
    # H1's post-invasion outcome, not just a repeat of the row's H0/H1 tag.
    for row, hyp in enumerate(("H0", "H1")):
        pos_left = axes[row, 0].get_position()
        pos_right = axes[row, 1].get_position()
        y = (pos_left.y0 + pos_left.y1) / 2
        x0, x1 = pos_left.x1 + 0.01, pos_right.x0 - 0.01
        fig.add_artist(mpatches.FancyArrowPatch(
            (x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=16, linewidth=1.4, color="black",
        ))
        fig.text((x0 + x1) / 2, y + 0.028, ARROW_ACTIONS[hyp],
                  fontsize=8, ha="center", va="center", linespacing=1.3)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


# Scheme A: bivariate -- hue encodes niche (blue family) vs not-niche (warm
# family), lightness/saturation encodes occupied (dark/saturated) vs
# unoccupied (light/pale). The 2x2 color relationship visually mirrors the
# 2x2 category structure.
def _style_bivariate(cat):
    return {
        0: ("#e6e6e6", "none", None, "solid", 1.0),   # not niche, unoccupied: pale gray
        1: ("#a6cee3", "none", None, "solid", 1.0),   # niche, unoccupied: pale blue
        2: ("#08519c", "none", None, "solid", 1.0),   # niche, occupied: dark blue
        3: ("#d7301f", "none", None, "solid", 1.0),   # not niche, occupied: dark red
    }[cat]


# Scheme B: flat qualitative (Okabe-Ito colorblind-safe palette). Simple to
# read but the 4 colors don't visually encode the 2x2 axis structure.
def _style_flat(cat):
    return {
        0: ("#999999", "none", None, "solid", 1.0),
        1: ("#56B4E9", "none", None, "solid", 1.0),
        2: ("#0072B2", "none", None, "solid", 1.0),
        3: ("#D55E00", "none", None, "solid", 1.0),
    }[cat]


# Scheme C: color + symbol mixture #1 -- color encodes niche (gray=not
# niche, green=niche); hatching encodes occupied (none=unoccupied,
# diagonal=occupied). The diagnostic cell (not niche, occupied) reads as
# "hatching on the wrong color."
def _style_color_hatch(cat):
    return {
        0: ("#e6e6e6", "#999999", None, "solid", 0.5),
        1: ("#b2e2c8", "#2b8a55", None, "solid", 0.5),
        2: ("#b2e2c8", "#2b8a55", "///", "solid", 0.5),
        3: ("#e6e6e6", "#999999", "///", "solid", 0.5),
    }[cat]


# Scheme D: color + symbol mixture #2 -- color encodes occupied (white=
# unoccupied, purple=occupied); border style encodes niche (dashed=not
# niche, solid=niche). The diagnostic cell (not niche, occupied) reads as
# "purple fill with a dashed outline."
def _style_color_outline(cat):
    return {
        0: ("#ffffff", "black", None, "dashed", 1.2),
        1: ("#ffffff", "black", None, "solid", 1.2),
        2: ("#8856a7", "black", None, "solid", 1.2),
        3: ("#8856a7", "black", None, "dashed", 1.2),
    }[cat]


# H0 and H1 share the identical pre-invasion state (that's the whole point --
# they're indistinguishable until the Great Plains is colonized), so instead
# of drawing it twice side by side, this layout draws it ONCE and branches
# into the two post-invasion outcomes via arrows labeled with what actually
# happens on that branch: H0's Great Plains flips from sink to source (it was
# already niche, just uncolonized) on top of the invasion itself; H1 has no
# such conversion -- occupancy expands into the Great Plains without it ever
# becoming niche, i.e. invasion alone.
PRE_CATEGORIES = SCENARIOS[0][2]
H0_POST_CATEGORIES = SCENARIOS[1][2]
H1_POST_CATEGORIES = SCENARIOS[3][2]
ARROW_ACTIONS = {
    "H0": "invasion +\nsink -> source conversion",
    "H1": "invasion only",
}


def _render_scheme_branching(zones, box_bounds, land_geom, range_geom, style_fn, out_png, legend_style_fn):
    # Size every axes box from the map's OWN aspect ratio (width:height of
    # box_bounds), rather than picking fractions and letting set_aspect
    # ("equal", forced in _panel) shrink the content to fit -- mismatched
    # boxes leave dead whitespace and strand titles far from the visible map.
    box_minx, box_miny, box_maxx, box_maxy = box_bounds
    map_aspect = (box_maxx - box_minx) / (box_maxy - box_miny)

    pre_w_in = 5.0
    pre_h_in = pre_w_in / map_aspect
    post_w_in = 5.4
    post_h_in = post_w_in / map_aspect

    left_margin, right_margin = 0.15, 0.15
    gap_pre_post = 2.1          # room for the H0/H1 arrows + action labels
    inter_post_gap = 0.9        # room for each post panel's own title
    top_margin, bottom_margin = 0.5, 0.7  # room for titles / legend

    fig_w = left_margin + pre_w_in + gap_pre_post + post_w_in + right_margin
    posts_total_h = 2 * post_h_in + inter_post_gap
    fig_h = max(pre_h_in, posts_total_h) + top_margin + bottom_margin

    fig = plt.figure(figsize=(fig_w, fig_h))

    def rect_in(x_in, y_in, w_in, h_in):
        return (x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h)

    pre_y_in = bottom_margin + (fig_h - bottom_margin - top_margin - pre_h_in) / 2
    ax_pre = fig.add_axes(rect_in(left_margin, pre_y_in, pre_w_in, pre_h_in))

    posts_y0_in = bottom_margin + (fig_h - bottom_margin - top_margin - posts_total_h) / 2
    x_post_in = left_margin + pre_w_in + gap_pre_post
    ax_h1 = fig.add_axes(rect_in(x_post_in, posts_y0_in, post_w_in, post_h_in))
    ax_h0 = fig.add_axes(rect_in(
        x_post_in, posts_y0_in + post_h_in + inter_post_gap, post_w_in, post_h_in
    ))

    _panel(ax_pre, zones, PRE_CATEGORIES, style_fn, box_bounds, land_geom, range_geom)
    ax_pre.set_title("Pre-invasion (shared by H0, H1)", fontsize=11)

    _panel(ax_h0, zones, H0_POST_CATEGORIES, style_fn, box_bounds, land_geom, range_geom)
    ax_h0.set_title("H0 -- Post-invasion", fontsize=11)

    _panel(ax_h1, zones, H1_POST_CATEGORIES, style_fn, box_bounds, land_geom, range_geom)
    ax_h1.set_title("H1 -- Post-invasion", fontsize=11)

    # Arrow endpoints/labels from the SAME computed geometry (figure-fraction
    # coordinates), so they track the panels exactly regardless of figure size.
    pre_right_x = (left_margin + pre_w_in) / fig_w
    pre_mid_y = (pre_y_in + pre_h_in / 2) / fig_h
    post_left_x = x_post_in / fig_w
    h0_mid_y = (posts_y0_in + post_h_in + inter_post_gap + post_h_in / 2) / fig_h
    h1_mid_y = (posts_y0_in + post_h_in / 2) / fig_h

    # Straight arrows (no arc bulge) so the H0/H1 action labels can sit at a
    # simple, predictable offset above/below the line without the curve
    # wandering into the text -- a curved arc's peak height isn't constant,
    # which is what made the earlier version collide with the caption text.
    arrow_kwargs = dict(transform=fig.transFigure, arrowstyle="-|>",
                         mutation_scale=22, linewidth=1.8, color="black")
    fig.add_artist(mpatches.FancyArrowPatch(
        (pre_right_x + 0.015, pre_mid_y), (post_left_x - 0.015, h0_mid_y), **arrow_kwargs
    ))
    fig.add_artist(mpatches.FancyArrowPatch(
        (pre_right_x + 0.015, pre_mid_y), (post_left_x - 0.015, h1_mid_y), **arrow_kwargs
    ))
    # Each arrow gets a bold H0/H1 tag plus a smaller caption naming the
    # actual ecological change on that branch (see ARROW_ACTIONS) -- "H0"/
    # "H1" alone name the hypothesis, not what changed to produce it.
    arrow_label_x = (pre_right_x + post_left_x) / 2
    h0_line_y = (pre_mid_y + h0_mid_y) / 2
    h1_line_y = (pre_mid_y + h1_mid_y) / 2
    fig.text(arrow_label_x, h0_line_y + 0.075, "H0",
              fontsize=14, fontweight="bold", ha="center", va="center")
    fig.text(arrow_label_x, h0_line_y + 0.035, ARROW_ACTIONS["H0"],
              fontsize=9.5, ha="center", va="center", linespacing=1.4)
    fig.text(arrow_label_x, h1_line_y - 0.035, "H1",
              fontsize=14, fontweight="bold", ha="center", va="center")
    fig.text(arrow_label_x, h1_line_y - 0.075, ARROW_ACTIONS["H1"],
              fontsize=9.5, ha="center", va="center", linespacing=1.4)

    handles = []
    for cat in range(4):
        facecolor, edgecolor, hatch, linestyle, linewidth = legend_style_fn(cat)
        handles.append(mpatches.Patch(
            facecolor=facecolor, edgecolor=edgecolor if edgecolor != "none" else "black",
            hatch=hatch, linestyle=linestyle, linewidth=max(linewidth, 1.0),
            label=CATEGORY_LABELS[cat],
        ))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    zones, project_crs, box_bounds, land_geom, range_geom = _load_zone_geoms()
    os.makedirs(OUT_DIR, exist_ok=True)

    _render_scheme(
        zones, box_bounds, land_geom, range_geom, _style_bivariate,
        os.path.join(OUT_DIR, "hypothesis_scheme_bivariate.png"), _style_bivariate,
    )
    _render_scheme(
        zones, box_bounds, land_geom, range_geom, _style_flat,
        os.path.join(OUT_DIR, "hypothesis_scheme_flat.png"), _style_flat,
    )
    _render_scheme(
        zones, box_bounds, land_geom, range_geom, _style_color_hatch,
        os.path.join(OUT_DIR, "hypothesis_scheme_color_hatch.png"), _style_color_hatch,
    )
    _render_scheme(
        zones, box_bounds, land_geom, range_geom, _style_color_outline,
        os.path.join(OUT_DIR, "hypothesis_scheme_color_outline.png"), _style_color_outline,
    )

    # Additional: shared-pre-invasion branching layout, same 4 color schemes.
    _render_scheme_branching(
        zones, box_bounds, land_geom, range_geom, _style_bivariate,
        os.path.join(OUT_DIR, "hypothesis_branch_bivariate.png"), _style_bivariate,
    )
    _render_scheme_branching(
        zones, box_bounds, land_geom, range_geom, _style_flat,
        os.path.join(OUT_DIR, "hypothesis_branch_flat.png"), _style_flat,
    )
    _render_scheme_branching(
        zones, box_bounds, land_geom, range_geom, _style_color_hatch,
        os.path.join(OUT_DIR, "hypothesis_branch_color_hatch.png"), _style_color_hatch,
    )
    _render_scheme_branching(
        zones, box_bounds, land_geom, range_geom, _style_color_outline,
        os.path.join(OUT_DIR, "hypothesis_branch_color_outline.png"), _style_color_outline,
    )


if __name__ == "__main__":
    main()
