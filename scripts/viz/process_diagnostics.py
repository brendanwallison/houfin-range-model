"""Process-level validation diagnostics (added 2026-11-06; called from map_diagnostics.main).

Why: the count likelihood cannot see the low-density dynamics that decide establishment -- a zero costs about mu, and
establishment (half of breeders mated) happens at densities ~25x below route detectability. These figures check the
dynamics against evidence that does not depend on the count -> bird conversion (research record: SANDBOX_INVASION_SHAPE
ZERO-COST, ESTABLISHMENT-LAG LIT, PRESENCE, FIG04-SPLIT, RUNGE).

  19_establishment_timing.png   model 'mating common' year (mated fraction >= .25 / .5 in a state's BBS-route cells; first
                                and 5-yr sustained) vs the documented window [first occurrence, first breeding]
                                (config/validation/establishment_windows.json) + the data's first BBS detection; map of
                                each cell's sustained establishment year.
  20_premature_establishment.png cells mating-common before their state's documented first occurrence, and before the
                                cell's own first BBS detection (cells first detected after 1970); years early.
  21_invisible_band.png         share of occupied cell-years below the mating threshold / between mating and route
                                detectability ('invisible' to the likelihood) / above detectability, by region over time.
  22_cbc_presence.png           early-CBC presences (1902-65; absences ignored) under the model: P(detect) per presence,
                                by region and era (needs population_model.cbc_likelihood.file).
  23_runge_contribution.png     Runge et al. 2006 breeding contribution C by era (needs <run_dir>/contribution_audit/
                                annual_budget.npz from research/exp/run_contribution_audit.py).
  04b_corrected_fitted_mean.png fig 04's fitted mean is the MEDIAN-route expectation; here the model's MEAN count (Jensen
                                correction for fitted route effects + unseen routes at the mean route), all and by region,
                                plus the data's within-observer x route paired trends.
Every figure is optional: a missing input returns {"skipped": reason}.
"""
import csv, json, os
from pathlib import Path
import numpy as np

# Regions are CEC Level I ecoregions (user rule 2026-11-07: never lat/lon boxes; the old west/plains/east longitude bands put
# the prairie-Midwest edge in "plains"). Raster on the model lattice: <ref_raster dir>/regions/cec_level1_54km.tif, or
# data_config regions.cec_level1. Ordered roughly west -> east; Level I codes absent from the sample are dropped.
CEC_L1 = {7: "Marine West Coast Forest", 11: "Mediterranean California", 6: "NW Forested Mountains", 10: "N. Am. Deserts",
          13: "Temperate Sierras", 12: "S. Semiarid Highlands", 9: "Great Plains", 3: "Taiga", 5: "Northern Forests",
          8: "Eastern Temperate Forests", 14: "Tropical Dry Forests", 15: "Tropical Humid Forests", 2: "Tundra", 4: "Hudson Plain"}


def _expand(p):
    return os.path.expandvars(str(p))


def _eco(dcfg, rows, cols):
    """CEC Level I code of each (row, col) on the model grid (0 = none). Raises if the raster is missing: a regional panel
    must never silently fall back to longitude bands."""
    import rasterio
    p = (dcfg.get("regions") or {}).get("cec_level1") or str(Path(_expand(dcfg["grid"]["ref_raster"])).parent / "regions" / "cec_level1_54km.tif")
    p = _expand(p)
    if not Path(p).exists():
        raise FileNotFoundError(f"CEC Level I raster not found at {p} (build: research scratch build_eco_l1.py)")
    with rasterio.open(p) as s, rasterio.open(_expand(dcfg["grid"]["ref_raster"])) as r:
        if (s.height, s.width) != (r.height, r.width) or not np.allclose(tuple(s.transform)[:6], tuple(r.transform)[:6]):
            raise ValueError(f"{p} is not on the model lattice of {dcfg['grid']['ref_raster']}")
        L1 = s.read(1)
    return np.asarray(L1)[np.asarray(rows), np.asarray(cols)].astype(int)


def _eco_regions(codes, weight=None, min_n=1):
    """[(name, mask)] for each CEC Level I region with at least min_n members (or weight) in `codes`."""
    w = np.ones(codes.shape) if weight is None else weight
    return [(nm, codes == c) for c, nm in CEC_L1.items() if w[codes == c].sum() >= min_n]


def _panel_grid(n, ncol=5, w=4.2, h=3.6):
    import matplotlib.pyplot as plt
    nr = int(np.ceil(n / ncol)); nc = min(n, ncol)
    fig, ax = plt.subplots(nr, nc, figsize=(w * nc, h * nr), squeeze=False)
    for a_ in ax.ravel()[n:]:
        a_.axis("off")
    return fig, ax.ravel()[:n]


def _lonlat(ref_raster, rows, cols):
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(_expand(ref_raster)) as s:
        xx, yy = rasterio.transform.xy(s.transform, rows, cols)
        lon, lat = transform(s.crs, "EPSG:4326", xx, yy)
        return np.asarray(lon), np.asarray(lat), s


def _cell_of(src, land, lat, lon):
    from rasterio.warp import transform
    xs, ys = transform("EPSG:4326", src.crs, [lon], [lat]); r, c = src.index(xs[0], ys[0])
    if 0 <= r < land.shape[0] and 0 <= c < land.shape[1] and land[r, c]:
        return r, c
    lr, lc = np.nonzero(land); i = int(np.argmin((lr - r) ** 2 + (lc - c) ** 2))
    return int(lr[i]), int(lc[i])


def _thresholds(sim):
    """Mating-.5 density and route-detection-.5 mean on the route-count scale (both in the model's own units)."""
    from src.model.age_forward import allee_mated_fraction
    lat = sim["latents"]
    gamma = float(np.ravel(np.asarray(sim["allee_gamma"]))[0])
    xs = np.logspace(-4, 2, 4000)
    mated = np.asarray(allee_mated_fraction(xs))
    n_mate50_rel = float(xs[np.argmax(mated >= .5)] / gamma)
    conc = float(np.ravel(np.asarray(lat.get("concentration", 1.)))[0]); kap = float(np.ravel(np.asarray(lat.get("obs_conc_kappa", 0.)))[0])
    mus = np.logspace(-4, 3, 4000); ph = conc * np.maximum(mus, 1e-3) ** kap
    pdet = 1. - (1. + mus / ph) ** (-ph)
    return gamma, n_mate50_rel, float(mus[np.argmax(pdet >= .5)]), conc, kap


def _mated(sim, N_rel):
    from src.model.age_forward import allee_mated_fraction
    gamma = float(np.ravel(np.asarray(sim["allee_gamma"]))[0])
    return np.asarray(allee_mated_fraction(gamma * N_rel))


def establishment_figures(sim, data, dcfg, years, rows, cols, shape, out_dir, windows_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    spec = json.load(open(windows_path)); bbs = Path(_expand(spec["bbs_dir"]))
    if not (bbs / "Routes.csv").exists():
        return {"skipped": f"no BBS release at {bbs}"}
    N = np.asarray(sim["Na_grid"]) + np.asarray(sim["Nj_grid"])                     # (T, Ny, Nx), relative
    land = np.asarray(data["land_mask"]).astype(bool); yrs = np.asarray(years).astype(int)
    M = _mated(sim, N)                                                                 # mated fraction per cell-year
    def sustained_year(series, th, k=5):
        above = series >= th; ok = [i for i in range(len(yrs) - k + 1) if above[i:i + k].all()]
        return int(yrs[ok[0]]) if ok else None
    routes = {}
    with open(bbs / "Routes.csv", newline="", encoding="latin-1") as fh:
        for r in csv.DictReader(fh):
            try:
                routes[(int(r["CountryNum"]), int(r["StateNum"]), int(r["Route"]))] = (float(r["Latitude"]), float(r["Longitude"]))
            except ValueError:
                pass
    _, _, src = _lonlat(dcfg["grid"]["ref_raster"], rows[:1], cols[:1])
    import rasterio
    rows_out, metrics = [], {}
    with rasterio.open(_expand(dcfg["grid"]["ref_raster"])) as s:
        for st, w in spec["states"].items():
            f = bbs / "States" / f"{st}.csv"
            if not f.exists():
                continue
            keys, first = set(), None
            with open(f, newline="", encoding="latin-1") as fh:
                for r in csv.DictReader(fh):
                    k = (int(r["CountryNum"]), int(r["StateNum"]), int(r["Route"])); keys.add(k)
                    if r["AOU"].strip() == "05190" and r["RPID"].strip() == "101" and int(r["SpeciesTotal"]) > 0:
                        first = min(first or 9999, int(r["Year"]))
            cells = sorted({_cell_of(s, land, *routes[k]) for k in keys if k in routes})
            if not cells:
                continue
            cr = np.array([c[0] for c in cells]); cc = np.array([c[1] for c in cells])
            series = M[:, cr, cc].max(1)
            first_cross = {th: (int(yrs[np.argmax(series >= th)]) if (series >= th).any() else None) for th in (.25, .5)}
            sus = {th: sustained_year(series, th) for th in (.25, .5)}
            crossings = int(np.sum((series[1:] >= .5) & (series[:-1] < .5)))
            rows_out.append((st, w["occ"], w["breed"], first, first_cross, sus, crossings))
    if not rows_out:
        return {"skipped": "no documented states found in the BBS release"}
    off = np.array([r[5][.5] - r[2] for r in rows_out if r[5][.5]])
    inside = np.array([((r[1] or -9999) <= (r[5][.5] or 0) <= r[2]) for r in rows_out])
    metrics = dict(n_states=len(rows_out), sustained50_minus_first_breeding_median=float(np.median(off)) if off.size else None,
                   sustained50_minus_first_breeding_mean_abs=float(np.mean(np.abs(off))) if off.size else None,
                   share_within_window=float(inside.mean()), upward_crossings_gt1=int(sum(r[6] > 1 for r in rows_out)),
                   states={r[0]: dict(occ=r[1], breed=r[2], first_bbs=r[3], first25=r[4][.25], first50=r[4][.5],
                                      sustained25=r[5][.25], sustained50=r[5][.5], crossings50=r[6]) for r in rows_out})
    # cell map of sustained establishment year (theta .5)
    est = np.full(shape, np.nan)
    for r_, c_ in zip(rows, cols):
        y = sustained_year(M[:, r_, c_], .5)
        if y is not None:
            est[r_, c_] = y
    fig, ax = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw=dict(width_ratios=[1, 1.3]))
    order = sorted(range(len(rows_out)), key=lambda i: rows_out[i][2])
    for j, i in enumerate(order):
        st, occ, brd, fb, fc, sus, cr_ = rows_out[i]
        ax[0].plot([occ or brd - 15, brd], [j, j], color="0.75", lw=6, solid_capstyle="butt")
        if fc[.25]: ax[0].plot(fc[.25], j, "o", mfc="none", mec="#2166ac")
        if sus[.5]: ax[0].plot(sus[.5], j, "o", color="#2166ac")
        if fb: ax[0].plot(fb, j, "x", color="#b2182b")
        if cr_ > 1: ax[0].text(1902, j, "!", color="red")
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([rows_out[i][0] for i in order], fontsize=8)
    ax[0].set(xlim=(1930, 2000), xlabel="year",
              title=f"Establishment vs documented window (grey: first occurrence -> first breeding)\n"
                    f"filled: model sustained mated>=.5; open: first mated>=.25; x: data first BBS detection\n"
                    f"median offset vs first breeding {metrics['sustained50_minus_first_breeding_median']:+.0f} yr; "
                    f"within window {metrics['share_within_window']:.0%}")
    im = ax[1].imshow(est, cmap="viridis", vmin=1900, vmax=2025); ax[1].axis("off")
    ax[1].set_title("Cell establishment year (mated fraction >= .5 for 5 yr)"); fig.colorbar(im, ax=ax[1], fraction=.035)
    fig.tight_layout(); fig.savefig(out_dir / "19_establishment_timing.png", dpi=130); plt.close(fig)
    # premature establishment map
    obs_r, obs_c = np.asarray(data["obs_rows"]), np.asarray(data["obs_cols"])
    ridx = np.asarray(data["obs_obsroute_idx"]) if "obs_obsroute_idx" in data else np.zeros(obs_r.size, int)
    yobs = np.asarray(data["observed_results"]); tobs = yrs[np.asarray(data["obs_time_indices"])]
    pos = (ridx >= 0) & (yobs > 0)
    firstdet = np.full(shape, np.nan)
    for r_, c_, y in zip(obs_r[pos], obs_c[pos], tobs[pos]):
        if np.isnan(firstdet[r_, c_]) or y < firstdet[r_, c_]:
            firstdet[r_, c_] = y
    early = np.where(np.isfinite(est) & np.isfinite(firstdet) & (firstdet > 1970), firstdet - est, np.nan)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    im = ax.imshow(np.where(early > 0, early, np.nan), cmap="Reds", vmin=0, vmax=40); ax.axis("off")
    ax.imshow(np.where(land & ~np.isfinite(early), 0., np.nan), cmap="Greys", vmin=0, vmax=4)
    ax.set_title("Premature establishment: years the model is mating-common BEFORE the cell's first BBS detection\n"
                 "(cells first detected after 1970)"); fig.colorbar(im, ax=ax, fraction=.035, label="years early")
    fig.tight_layout(); fig.savefig(out_dir / "20_premature_establishment.png", dpi=130); plt.close(fig)
    e = early[np.isfinite(early)]
    metrics["premature_cells"] = dict(n=int(e.size), share_gt10yr=float((e > 10).mean()) if e.size else None,
                                      median_years_early=float(np.median(e)) if e.size else None)
    return metrics


def invisible_band(sim, data, dcfg, years, rows, cols, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    gamma, n_mate50, mu_det50, conc, kap = _thresholds(sim)
    scale = float(np.asarray(data["pop_scalar"]))
    N = (np.asarray(sim["Na_grid"]) + np.asarray(sim["Nj_grid"]))[:, rows, cols]
    eco = _eco(dcfg, rows, cols)
    occ = N * scale > 1e-4
    lo = N < n_mate50; hi = N * scale >= mu_det50; mid = ~lo & ~hi
    yrs = np.asarray(years).astype(int)
    regs = _eco_regions(eco, occ.any(0).astype(float), min_n=5)
    fig, ax = _panel_grid(len(regs)); out = {}
    for a_, (reg, m) in zip(ax, regs):
        m = m[None, :]
        tot = np.maximum((occ & m).sum(1), 1)
        shares = [((occ & b) & m).sum(1) / tot for b in (lo, mid, hi)]
        a_.stackplot(yrs, *shares, labels=["below mating threshold", "invisible band (mated, undetectable)", "detectable"],
                     colors=["#d9d9d9", "#fdae61", "#2c7bb6"])
        a_.set(title=f"{reg} (CEC L1)", xlabel="year", ylim=(0, 1)); out[reg] = dict(invisible_share_1960=float(shares[1][yrs == 1960][0]),
                                                                            invisible_share_2010=float(shares[1][yrs == 2010][0]))
    ax[0].set_ylabel("share of occupied cells"); ax[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Density bands (route-count scale): mating .5 at {n_mate50 * scale:.3f}, route detection .5 at {mu_det50:.2f} "
                 "-- the likelihood barely sees the orange band (a zero costs ~mu)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "21_invisible_band.png", dpi=130); plt.close(fig)
    out.update(mating50_route_counts=n_mate50 * scale, detect50_route_counts=mu_det50)
    return out


def cbc_presence(sim, data, cfg, dcfg, years, rows, cols, shape, out_dir, k=0.43724203275856455):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f = ((cfg.get("population_model", {}) or {}).get("cbc_likelihood") or {}).get("file")
    if not f or not Path(_expand(f)).exists():
        return {"skipped": "no population_model.cbc_likelihood.file"}
    c = np.load(_expand(f)); scale = float(np.asarray(data["pop_scalar"]))
    N = (np.asarray(sim["Na_grid"]) + np.asarray(sim["Nj_grid"]))[:, rows, cols] * scale
    yrs = np.asarray(years).astype(int)
    pos = c["cbc_count"] > 0
    src, wl, ws = c["cbc_source_idx"], c["cbc_w_local"], c["cbc_w_seasonal"]
    eco_land = _eco(dcfg, rows, cols)
    j = np.flatnonzero(pos); t = c["cbc_t"][j]; ci = c["cbc_circle"][j]
    s = np.maximum(src[ci], 0); w = np.where(src[ci] >= 0, np.maximum(wl[ci], ws[ci]), 0.)
    A = np.sum(N[t[:, None], s] * w, axis=1)
    p = -np.expm1(-k * np.log1p(A / k))
    ceco = eco_land[s[np.arange(len(s)), np.argmax(wl[ci], 1)]]       # ecoregion of the circle's main local source cell
    out = {}
    for reg, mreg in _eco_regions(ceco):
        for lo_, hi_ in ((1902, 1939), (1940, 1959), (1960, 1965)):
            m = mreg & (yrs[t] >= lo_) & (yrs[t] <= hi_)
            if m.any():
                out[f"{reg}_{lo_}_{hi_}"] = dict(n=int(m.sum()), cost=float(-np.log(np.maximum(p[m], 1e-300)).sum()),
                                                share_p_lt_05=float((p[m] < .05).mean()), share_p_lt_001=float((p[m] < .001).mean()))
    grid = np.full(shape, np.nan)
    cell_r, cell_c = rows[s[np.arange(len(s)), np.argmax(wl[ci], 1)]], cols[s[np.arange(len(s)), np.argmax(wl[ci], 1)]]
    for r_, c_, pv in zip(cell_r, cell_c, p):
        grid[r_, c_] = pv if np.isnan(grid[r_, c_]) else min(grid[r_, c_], pv)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=.035, label="min P(detect)")
    ax.set_title("Early-CBC presences (1902-65) under the model: lowest P(detect) per cell (red = presence the model calls impossible)")
    fig.tight_layout(); fig.savefig(out_dir / "22_cbc_presence.png", dpi=130); plt.close(fig)
    return out


def runge_figure(run_dir, dcfg, shape, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    f = Path(run_dir) / "contribution_audit" / "annual_budget.npz"
    if not f.exists():
        return {"skipped": f"no {f} (run research/exp/run_contribution_audit.py)"}
    z = np.load(f); yrs = z["years"]; lr, lc = z["land_rows"], z["land_cols"]
    eras = [(1966, 1993), (1994, 2009), (2010, 2025)]
    # Which cells to SHOW: mean breeders >= 1e-4 x the densest cell (was 1e-2, which hid the moderately occupied Great Plains
    # behind coastal California's densities -- user, 2026-11-08). The breeder-weighted summaries are insensitive to this
    # (the extra cells carry little weight); per-region (CEC GP / ETF) summaries are added so the Plains get their own number.
    try:
        eco = _eco(dcfg, lr, lc)
    except Exception:
        eco = None
    fig, ax = plt.subplots(1, 3, figsize=(18, 5)); out = {}
    for a_, (lo, hi) in zip(ax, eras):
        t = (yrs >= lo) & (yrs <= hi) & (z["pulse"].max(1) == 0)
        num, den = z["breeder_contribution"][t].sum(0), z["input_adult"][t].sum(0)
        C = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 1e-20)
        occ = den / max(t.sum(), 1) >= 1e-4 * np.nanmax(den / max(t.sum(), 1))
        g = np.full(shape, np.nan); g[lr, lc] = np.where(occ, C, np.nan)
        im = a_.imshow(g, cmap="RdBu", norm=TwoSlopeNorm(1., .3, 2.)); a_.axis("off")
        out[f"{lo}_{hi}"] = dict(breeder_weighted_C=float(np.nansum(C[occ] * den[occ]) / den[occ].sum()),
                                 source_breeder_share=float(den[occ & (C > 1)].sum() / den[occ].sum()), cells_shown=int(occ.sum()))
        reg_txt = ""
        if eco is not None:
            for code, nm in ((9, "GP"), (8, "ETF")):
                m = occ & (eco == code)
                if m.any() and den[m].sum() > 0:
                    out[f"{lo}_{hi}"][f"C_{nm}"] = float(np.nansum(C[m] * den[m]) / den[m].sum())
                    out[f"{lo}_{hi}"][f"cells_{nm}"] = int(m.sum())
                    reg_txt += f"; {nm} C {out[f'{lo}_{hi}'][f'C_{nm}']:.2f}"
        a_.set_title(f"Runge C {lo}-{hi}; source share {out[f'{lo}_{hi}']['source_breeder_share']:.2f}{reg_txt}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.02, label="C (>1 source)")
    fig.savefig(out_dir / "23_runge_contribution.png", dpi=130); plt.close(fig)
    return out


def _eco_level(dcfg, rows, cols, level):
    """CEC Level `level` (1 or 2) code per (row, col) + {code: label}; None if the raster is absent (level 2 is optional)."""
    import rasterio
    d = Path(_expand(dcfg["grid"]["ref_raster"])).parent / "regions"
    tif, js = d / f"cec_level{level}_54km.tif", d / f"cec_level{level}_54km.json"
    if not tif.exists():
        return None, None
    with rasterio.open(tif) as s:
        codes = np.asarray(s.read(1))[np.asarray(rows), np.asarray(cols)].astype(int)
    meta = json.loads(js.read_text())["classes"] if js.exists() else {}
    if level == 1:
        names = {int(k): str(v).title() for k, v in meta.items()}
    else:
        names = {int(k): f"{v['code']} {str(v['name']).title()}" for k, v in meta.items()}
    return codes, names


def runge_regions_figure(run_dir, dcfg, out_dir, min_share=.005, min_share_l2=.002):
    """Breeder-weighted Runge C by CEC Level I and Level II ecoregion x era (user, 2026-11-08), as annotated heatmaps.
    Rows sorted by breeder share; regions holding < min_share of an era's breeders are omitted (listed in the json)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    f = Path(run_dir) / "contribution_audit" / "annual_budget.npz"
    if not f.exists():
        return {"skipped": f"no {f}"}
    z = np.load(f); yrs = z["years"]; lr, lc = z["land_rows"], z["land_cols"]
    eras = [(1966, 1993), (1994, 2009), (2010, 2025)]
    per_era = []
    for lo, hi in eras:
        t = (yrs >= lo) & (yrs <= hi) & (z["pulse"].max(1) == 0)
        per_era.append((z["breeder_contribution"][t].sum(0), z["input_adult"][t].sum(0)))
    levels = [(lv,) + _eco_level(dcfg, lr, lc, lv) for lv in (1, 2)]
    levels = [(lv, c, n) for lv, c, n in levels if c is not None]
    out = {}
    fig, axes = plt.subplots(1, len(levels), figsize=(9.5 * len(levels), 12), gridspec_kw=dict(width_ratios=[1] * len(levels)))
    fig.subplots_adjust(wspace=.95, left=.2)
    axes = np.atleast_1d(axes)
    norm = TwoSlopeNorm(1., .7, 1.3)
    for ax_, (lv, codes, names) in zip(axes, levels):
        tot = sum(den.sum() for _, den in per_era)
        rows_ = []
        for code in np.unique(codes[codes > 0]):
            m = codes == code
            share = sum(den[m].sum() for _, den in per_era) / max(tot, 1e-20)
            Cs = [float(num[m].sum() / den[m].sum()) if den[m].sum() > 0 else np.nan for num, den in per_era]
            rows_.append((share, names.get(int(code), str(code)), Cs))
        rows_.sort(key=lambda r: -r[0])
        thr = min_share if lv == 1 else min_share_l2
        keep = [r for r in rows_ if r[0] >= thr]
        out[f"level{lv}"] = {r[1]: dict(breeder_share=r[0], **{f"C_{lo}_{hi}": c for (lo, hi), c in zip(eras, r[2])}) for r in rows_}
        M = np.array([r[2] for r in keep])
        im = ax_.imshow(M, cmap="RdBu", norm=norm, aspect="auto")
        for i, r in enumerate(keep):
            for j, c in enumerate(r[2]):
                if np.isfinite(c):
                    ax_.text(j, i, f"{c:.2f}", ha="center", va="center", fontsize=8,
                             color="white" if abs(c - 1.) > .2 else "#1a1a1a")
        ax_.set_yticks(range(len(keep)), [f"{r[1][:38]}  ({100 * r[0]:.0f}%)" for r in keep], fontsize=8)
        ax_.set_xticks(range(len(eras)), [f"{lo}-{hi}" for lo, hi in eras], fontsize=9)
        ax_.set_title(f"CEC Level {'I' if lv == 1 else 'II'} (rows: share of breeders >= {100 * thr:.1f}%)", fontsize=10)
        ax_.tick_params(length=0)
        for s in ax_.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, fraction=.025, label="breeder-weighted Runge C (>1 net source; colour clipped .7-1.3)")
    fig.suptitle("Runge contribution by ecoregion and era", fontsize=12)
    fig.savefig(out_dir / "23b_runge_by_ecoregion.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    (out_dir / "23b_runge_by_ecoregion.json").write_text(json.dumps(out, indent=1) + "\n")
    return {"levels": [lv for lv, _, _ in levels]}


def model_mean_count(sim, data, cfg, run_dir=None):
    """Per-row expected count as the model's MEAN (not median-route) count: rows on fitted routes get the Laplace
    posterior-variance Jensen factor exp(var/2); rows on routes with no training data are drawn at the MEAN route,
    mu0 exp(sigma^2 / 2). Returns (mean_count, sigma) or (None, None) when the model has no observer effect.
    Shared by fig 04 (map_diagnostics.plot_fit_diagnostics) and 04b."""
    if "expected_obs_before_observer" not in sim or "obs_obsroute_idx" not in data:
        return None, None
    spec = ((cfg.get("population_model", {}) or {}).get("observation", {}) or {}).get("observer_effect", {}) or {}
    sigma = float(np.ravel(np.asarray(sim["latents"].get("observer_route_sigma", spec.get("sigma_u", 0.))))[0])
    if run_dir is not None and (Path(run_dir) / "laplace_route_sigma.json").exists():
        sigma = float(json.load(open(Path(run_dir) / "laplace_route_sigma.json"))["sigma"])
    y = np.asarray(data["observed_results"], float); w = np.asarray(data.get("obs_weight", np.ones(y.size)), float)
    mu = np.asarray(sim["expected_obs"], float); mu0 = np.asarray(sim["expected_obs_before_observer"], float)
    phi = np.asarray(sim["obs_concentration"], float); idx = np.asarray(data["obs_obsroute_idx"])
    train = (w > 0) & np.isfinite(y)
    n = int(idx.max()) + 1; info = np.zeros(n)
    on = (idx >= 0) & train
    np.add.at(info, idx[on], (w * (np.where(train, y, 0.) + phi) * mu * phi / (mu + phi) ** 2)[on])
    seen = np.zeros(n, bool); seen[idx[on]] = True
    var = sigma ** 2 / (1. + sigma ** 2 * info)
    corr = mu.copy()
    s = (idx >= 0) & seen[np.maximum(idx, 0)]
    corr[s] = mu[s] * np.exp(.5 * var[idx[s]])
    u = (idx >= 0) & ~seen[np.maximum(idx, 0)]
    corr[u] = mu0[u] * np.exp(.5 * sigma ** 2)
    return corr, sigma


def corrected_fitted_mean(sim, data, cfg, dcfg, years, out_dir, run_dir=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if "expected_obs_before_observer" not in sim or "obs_obsroute_idx" not in data:
        return {"skipped": "no observer effect in this model"}
    # Laplace-marginal fits estimate sigma outside the model's sites (laplace_route_sigma.json beside the checkpoint).
    # Per-route information is the OBSERVED NB2 information on the log-mean scale, (y + phi) mu phi / (mu + phi)^2: the
    # expected (Fisher) form understates it on high-count routes and over-corrects (checked vs the exact Laplace curvature
    # of research/exp/invasion_sandbox/fitted_mean_split.py).
    corr, sigma = model_mean_count(sim, data, cfg, run_dir)
    y = np.asarray(data["observed_results"], float); w = np.asarray(data.get("obs_weight", np.ones(y.size)), float)
    mu = np.asarray(sim["expected_obs"], float); idx = np.asarray(data["obs_obsroute_idx"])
    yrs = np.asarray(years).astype(int)[np.asarray(data["obs_time_indices"])]
    train = (w > 0) & np.isfinite(y)
    rows, cols = np.asarray(data["obs_rows"]), np.asarray(data["obs_cols"])
    eco = _eco(dcfg, rows, cols)
    ok = np.isfinite(y) & (idx >= 0)
    regs = [("all", np.ones(y.size, bool))] + _eco_regions(eco, (ok & train & (yrs >= 1966)).astype(float), min_n=300)
    fig, ax = _panel_grid(len(regs)); out = {}
    for a_, (reg, mreg) in zip(ax, regs):
        m = ok & mreg & (yrs >= 1966)
        ys = np.unique(yrs[m])
        f = lambda v: np.array([v[m & (yrs == t)].mean() for t in ys])
        a_.plot(ys, f(y), color="k", label="observed"); a_.plot(ys, f(mu), color="#d95f0e", ls="--", label="fig 04 (median route)")
        a_.plot(ys, f(corr), color="#2166ac", label="model mean count (corrected)"); a_.set_title(reg); a_.set_xlabel("year")
        out[reg] = dict(obs_2019=float(f(y)[ys == 2019][0]) if (ys == 2019).any() else None,
                        model_2019=float(f(corr)[ys == 2019][0]) if (ys == 2019).any() else None,
                        obs_2025=float(f(y)[ys == ys.max()][0]), model_2025=float(f(corr)[ys == ys.max()][0]))
    ax[0].legend(fontsize=7); ax[0].set_ylabel("mean BBS count per route-year")
    # data-only: within observer x route paired trends by region (ratio of means, 2016-19 -> 2021-25)
    txt = []
    for reg, mreg in regs[1:]:
        m = ok & mreg
        A_, B_ = m & (yrs >= 2016) & (yrs <= 2019), m & (yrs >= 2021) & (yrs <= 2025)
        both = np.intersect1d(np.unique(idx[A_]), np.unique(idx[B_]))
        if both.size:
            ma = np.array([y[A_ & (idx == k)].mean() for k in both]); mb = np.array([y[B_ & (idx == k)].mean() for k in both])
            out[reg]["within_observer_ratio_2016_19_to_2021_25"] = float(mb.sum() / max(ma.sum(), 1e-9)); txt.append(f"{reg} x{mb.sum() / max(ma.sum(), 1e-9):.2f}")
    fig.suptitle("Fitted mean as the model's MEAN count (Jensen + unseen routes at the mean route), by CEC Level I ecoregion\n"
                 "within-observer x route 2016-19 -> 2021-25: " + ", ".join(txt), fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "04b_corrected_fitted_mean.png", dpi=130); plt.close(fig)
    out["sigma_used"] = sigma
    return out


def run_process_diagnostics(sim, data, cfg, dcfg, years, rows, cols, shape, out_dir, run_dir=None, repo_root=None):
    """Runs every process diagnostic; each failure or missing input is recorded, never raised."""
    out_dir = Path(out_dir); repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    windows = repo_root / "config" / "validation" / "establishment_windows.json"
    jobs = {"establishment": lambda: establishment_figures(sim, data, dcfg, years, rows, cols, shape, out_dir, windows),
            "invisible_band": lambda: invisible_band(sim, data, dcfg, years, rows, cols, out_dir),
            "cbc_presence": lambda: cbc_presence(sim, data, cfg, dcfg, years, rows, cols, shape, out_dir),
            "runge": lambda: runge_figure(run_dir, dcfg, shape, out_dir) if run_dir else {"skipped": "no run_dir"},
            "runge_regions": lambda: runge_regions_figure(run_dir, dcfg, out_dir) if run_dir else {"skipped": "no run_dir"},
            "corrected_fitted_mean": lambda: corrected_fitted_mean(sim, data, cfg, dcfg, years, out_dir, run_dir)}
    res = {}
    for k, f in jobs.items():
        try:
            res[k] = f()
        except Exception as e:  # diagnostics never block the suite
            res[k] = {"error": repr(e)}
        print(f"[process-diag] {k}: {'skipped' if 'skipped' in res[k] else ('ERROR ' + res[k]['error'] if 'error' in res[k] else 'ok')}", flush=True)
    return res
