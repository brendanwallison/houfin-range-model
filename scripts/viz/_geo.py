"""A North America the reader recognises, on the model grid. Shared by the validation maps.

WHY THIS EXISTS. The repo has two map idioms that never meet. `map_diagnostics.py` draws ~18
panels as a bare ``imshow`` of a 133x224 array with the axes off -- correct pixels, no geography,
and unreadable to anyone who does not already know the grid. The docs figures
(`hypothesis_r_field.py`, `hypothesis_scenarios.py`) do it properly: a projected extent in
ESRI:102003 metres with the Natural Earth coastline drawn over it. That second idiom was never
factored out, so it exists only inside two scripts that reach each other through
``sys.path``-inserted private names.

Validation maps need the second one. A map whose whole job is "is the error at the expansion
front, or in the Great Plains, or on the coast" cannot be a pixel block -- the answer is a place,
and the reader has to be able to see which place.

WHAT THIS DELIBERATELY DOES NOT DO. It never loads the 44 MB CEC ecoregion shapefile: that file is
gitignored, hand-downloaded, and absent on HPC. Great Plains geometry comes from the RASTER via
``read_zone_raster``, the path that was written to be HPC-safe, and every optional layer degrades
to absent rather than raising -- a missing basemap should cost a coastline, not a figure.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib                                             # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

from src.config_utils import load_data_config                 # noqa: E402

#: Land that is drawn but carries no data. Deliberately not white: white reads as "zero" next to a
#: sequential colormap, and the distinction between "no finch data here" and "no change here" is
#: one this project has already had to relearn on a figure.
LAND_FILL = "#efece7"
COAST = "#5a5a5a"
GP_EDGE = "#8a5a2b"


def base_grid_spec():
    """The COMMITTED grid, bypassing any ``local_data_config.json`` override.

    Every validation artifact is produced on TACC at 27 km. This machine's dev override points at
    25 km, so a figure that resolved its extent through ``load_data_config`` would place a 27 km
    array on a 25 km box and be silently wrong by ~8% -- shifted, not obviously broken.
    ``covariate_samples._base_grid_spec`` exists for exactly this reason; this is that function,
    in the shared place, so the next map does not have to rediscover it.
    """
    with open(os.path.join(_REPO, "config", "data_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    minx, miny, maxx, maxy = cfg["grid"]["box_bounds"]
    return {"box_bounds": (minx, miny, maxx, maxy),
            # matplotlib's imshow extent order, with origin="upper"
            "extent": (minx, maxx, miny, maxy),
            "res_m": float(cfg["grid"]["target_res_m"]),
            "crs": cfg["grid"]["box_crs"]}


class GeoContext:
    """Everything a map needs to place an array on the continent, loaded once.

    ``ocean``/``land`` are the model's own terrestrial mask, not a redrawn one, so a cell that the
    model treats as water is water here too. ``land_geom`` and ``gp_geom`` are optional: they need
    geopandas and the Natural Earth / zone-raster files, and their absence costs an outline.
    """

    def __init__(self, shape=None, want_vectors=True):
        spec = base_grid_spec()
        self.extent = spec["extent"]
        self.box_bounds = spec["box_bounds"]
        self.res_m = spec["res_m"]
        self.crs = spec["crs"]
        self.land = self._load_land(shape)
        self.shape = self.land.shape if self.land is not None else shape
        self.gp_zones = self._load_gp_zones(self.shape)
        self.front = self._load_front(self.shape)
        self.land_geom = self._load_land_geom() if want_vectors else None
        self.notes = []
        if self.land is None:
            self.notes.append("land mask unavailable")
        if self.land_geom is None:
            self.notes.append("no coastline (Natural Earth land shapefile absent)")
        if self.gp_zones is None:
            self.notes.append("no Great Plains outline (zone raster absent)")
        if self.front is None:
            self.notes.append("no colonization front (run scripts/build_colonization_front.py)")

    # --- loaders, each independently optional ---------------------------------------------
    def _load_land(self, shape):
        try:
            import rasterio
            from src.config_utils import load_age_model_config
            path = load_age_model_config()["ocean_mask"]
            with rasterio.open(path) as src:
                ocean = src.read(1)
            land = ocean == 0
            return land if (shape is None or land.shape == shape) else None
        except Exception:
            return None

    def _load_gp_zones(self, shape):
        try:
            from src.data.preprocess.great_plains import read_zone_raster
            path = load_data_config()["regions"]["great_plains_zones"]
            return read_zone_raster(path, shape) if shape else None
        except Exception:
            return None

    def _load_front(self, shape):
        """Observed first-detection year, EASTERN EXPANSION ONLY -- the native hull is nodata.

        Optional like every other overlay. Its absence is worth a note rather than a failure,
        because it is a derived product that has to be built where the BBS npz lives.
        """
        try:
            import rasterio
            cfg = load_data_config()
            path = os.path.join(cfg["processed_root"], "regions",
                                "colonization_front_27km.tif")
            if not os.path.exists(path):
                return None
            with rasterio.open(path) as src:
                a = src.read(1).astype("float64")
                nod = src.nodata
            a = np.where((a == nod) if nod is not None else False, np.nan, a)
            return a if (shape is None or a.shape == shape) else None
        except Exception:
            return None

    def _load_land_geom(self):
        try:
            import geopandas as gpd
            from shapely.ops import unary_union
            cfg = load_data_config()
            src = os.path.join(cfg["datasets_root"], cfg["coastline"]["land_source"])
            if not os.path.exists(src):
                return None
            gdf = gpd.read_file(src).to_crs(self.crs)
            # buffer(0) is REQUIRED, not defensive: reprojected Natural Earth polygons carry
            # self-intersections and unary_union raises on them. Recorded the same way in
            # hypothesis_scenarios.py, which hit it first.
            return unary_union(list(gdf.geometry.buffer(0)))
        except Exception:
            return None

    # --- drawing --------------------------------------------------------------------------
    def imshow(self, ax, grid, mask_to_land=True, **kw):
        """Place a grid array on the continent. Returns the mappable."""
        g = np.asarray(grid, dtype="float64")
        if mask_to_land and self.land is not None and self.land.shape == g.shape:
            g = np.where(self.land, g, np.nan)
        cmap = kw.pop("cmap", "viridis")
        cm = plt.get_cmap(cmap).copy()
        # Missing data is transparent so the land fill beneath shows through: a NaN cell then
        # reads as "land, no measurement" instead of as the low end of the colour ramp.
        cm.set_bad(alpha=0.0)
        return ax.imshow(g, extent=self.extent, origin="upper", cmap=cm, **kw)

    def basemap(self, ax, fill_land=True):
        """Land fill under the data, coastline over it, extent and aspect pinned."""
        if fill_land and self.land_geom is not None:
            self._plot_geom(ax, self.land_geom, facecolor=LAND_FILL, edgecolor="none", zorder=0)
        elif fill_land and self.land is not None:
            ax.imshow(np.where(self.land, 1.0, np.nan), extent=self.extent, origin="upper",
                      cmap=matplotlib.colors.ListedColormap([LAND_FILL]), zorder=0)
        minx, miny, maxx, maxy = self.box_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        # Mandatory: a PathPatch axis defaults to aspect "auto" and stretches the continent.
        ax.set_aspect("equal")
        ax.set_axis_off()

    def coastline(self, ax, lw=0.5, color=COAST):
        if self.land_geom is not None:
            self._plot_geom(ax, self.land_geom, facecolor="none", edgecolor=color, linewidth=lw,
                            zorder=5)

    def great_plains(self, ax, lw=1.1, color=GP_EDGE, ls=(0, (4, 2))):
        """Outline the barrier corridor. Drawn from the raster, so no shapefile is needed."""
        if self.gp_zones is None:
            return False
        ax.contour(self.gp_zones["barrier"].astype(float), levels=[0.5], colors=[color],
                   linewidths=lw, linestyles=[ls], extent=self.extent, origin="upper", zorder=6)
        return True

    def _plot_geom(self, ax, geom, **kw):
        import geopandas as gpd
        gs = gpd.GeoSeries([geom], crs=self.crs)
        if kw.get("facecolor") in (None, "none"):
            gs.boundary.plot(ax=ax, **{k: v for k, v in kw.items() if k != "facecolor"})
        else:
            gs.plot(ax=ax, **kw)

    def scalebar(self, ax, km=500, loc=(0.06, 0.06)):
        """A scale bar. The repo had none, and a projected map without one invites the reader to
        judge distance by pixels -- which is the whole question when the subject is a 918 km
        barrier and a 162 km holdout block."""
        minx, miny, maxx, maxy = self.box_bounds
        x0 = minx + loc[0] * (maxx - minx)
        y0 = miny + loc[1] * (maxy - miny)
        ax.plot([x0, x0 + km * 1000], [y0, y0], color="#333333", lw=2, solid_capstyle="butt",
                zorder=8)
        ax.text(x0 + km * 500, y0 + 0.012 * (maxy - miny), f"{km} km", ha="center", va="bottom",
                fontsize=6.5, color="#333333", zorder=8)


def to_grid(rows, cols, values, shape, reduce="mean"):
    """Scatter per-cell values onto the grid. Repeated cells are averaged (or counted).

    Not another private clone: four already exist and this one differs in taking a reduction,
    because a validation map is usually many cell-YEARS collapsing onto one cell and the reader
    must know whether they are looking at a mean or a count.
    """
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    vals = np.asarray(values, dtype="float64")
    ok = np.isfinite(vals)
    lin = rows[ok] * shape[1] + cols[ok]
    cnt = np.bincount(lin, minlength=shape[0] * shape[1]).astype("float64")
    if reduce == "count":
        return cnt.reshape(shape)
    tot = np.bincount(lin, weights=vals[ok], minlength=shape[0] * shape[1])
    out = np.full(shape[0] * shape[1], np.nan)
    nz = cnt > 0
    out[nz] = tot[nz] / cnt[nz]
    return out.reshape(shape)


def gate(grid, support, min_support, label="cells"):
    """NaN out cells with too little support, and say how many survived.

    The house precedent is `trend_diagnostics._gate_mask`: gate hard to NaN, name the threshold,
    print the retained count. Chosen over alpha-by-n or stippling because neither exists anywhere
    in this repo and a new convention should be the simplest defensible one -- a faded cell still
    invites reading its colour, a NaN cell does not.
    """
    keep = np.asarray(support) >= min_support
    out = np.where(keep, grid, np.nan)
    n = int(np.isfinite(out).sum())
    return out, f"{n:,} {label} at >= {min_support}"


def desaturate(grid, room, floor=0.15):
    """Return an alpha field that fades a value toward the background as its room -> 0.

    The scalar suite refuses to rank predictors where the floor-to-ceiling distance is under
    ~0.15, because differences there are not evidence of skill. A map has no way to refuse a cell,
    so it fades it: a cell nobody can resolve should recede rather than shout a colour.
    """
    r = np.asarray(room, dtype="float64")
    a = np.clip(r / max(floor, 1e-9), 0.0, 1.0)
    return np.where(np.isfinite(r) & np.isfinite(grid), a, 0.0)
