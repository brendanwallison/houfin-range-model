"""E073: should DESK be a persistent spatial field PLUS a covariate-driven within-place deviation, z = zbar(x) + delta,
delta = g(c(x,t) - cbar(x); place), g(0) = 0, with no time or position-by-time input -- and would that EXTRAPOLATE?

    python research/exp/anchored_delta.py novelty  --cache <dir> --out <dir>          # part A (no bird data)
    python research/exp/anchored_delta.py score    --cache <tempho dir> --out <dir> --similarity <csv>   # part B
    python research/exp/anchored_delta.py holdout  --cache <base dir> --out <dir> --similarity <csv>     # part C
    python research/exp/anchored_delta.py score ... --planted            # verification (a): a planted response

Covariates come from the yearly states (1902-2025; streams climate 240, landuse 33 (LUH2), hyde 3, bui 7, static 19),
transformed as DESK does, standardized on training rows. Causal EMAs at half-lives 0/3/10/25 years start in 1902.
Dynamic streams are grouped: climate, landuse, human (hyde + bui). Within-place deviation dc = c - cbar(cell), cbar
over the cell's trained-year rows. Features W = principal components of dc per stream, fitted on training rows
(climate 24, landuse 12, human 6; "all" = their concatenation, 42).

Estimand (parts B, C): the standard one (species_readout_probe / fair_rivals --target dev): held-out cells, early
epoch (withheld 1966-86 in tempho caches; trained years in the base cache) vs 2005-25, the 273 out-of-community species
(log1p of ABBA half means; noise-free pooled corr and k), 6x6-block bootstrap, paired arm differences; strata by AVONET
migration (sedentary primary) and hofi_like; ESK r24 community change as the secondary target. Every map is a per-
species BLR (lib.blr, block amplitudes by marginal likelihood) fitted on training cells x trained years: within maps on
within-cell deviations; the between map on cell means.

Part A (exposure and novelty): within-place share of each stream's variance; novelty of the 1920-39 state relative to
each cell's 1966-2025 mean, against the distribution of 6-year-window deviations in 1966-2025 (Mahalanobis in the
stream's within-PC space; share beyond the reference 99th percentile), by Great Plains zone; analog support of the
1920-39 state among all 1966-2025 cell-windows of OTHER cells (nearest-neighbour distance vs the modern reference);
direction of the main land-use channels' 1920-39 minus 1996-2025 difference by zone.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

STATES = os.path.expanduser("~/houfin/work/houfin/processed/encoder/states")
ZONES_TIF = os.path.expanduser("~/houfin/work/houfin/processed/regions/great_plains_zones_27km.tif")
ZONES = {1: "west", 2: "great_plains", 3: "east"}
# Amendment 2026-10-14 (after part A, before any bird-data number): land use split into COVER (the 12 LUH2 state
# fractions) and the MANAGEMENT CLOCK (E041's 19 management + 2 secondary-land channels: fertilizer, irrigation, harvest,
# flood, plantations, protection, secondary age / biomass) -- flat before ~1950, a continental clock in 1966-2025.
# amendment 2: physical cover -- forest = primf + secdf, natural non-forest = primn + secdn (derived in load_states);
# the primary / secondary accounting channels move to the clock group with management
COVER = ("c3ann", "c3nfx", "c3per", "c4ann", "c4per", "pastr", "range", "urban", "forest", "natnonforest")
GROUPS = {"climate": ("climate",), "cover": ("landuse:cover",), "mgmt": ("landuse:mgmt",), "human": ("hyde", "bui")}
NPC = {"climate": 24, "cover": 8, "mgmt": 8, "human": 6}


def group_cols(stream, names, g):
    """Column indices of stream group g ("landuse:cover" / "landuse:mgmt" split the land-use stream by channel name)."""
    sel = np.zeros(len(stream), bool)
    for s in GROUPS[g]:
        if s == "landuse:cover":
            sel |= (stream == "landuse") & np.isin(names, COVER)
        elif s == "landuse:mgmt":
            sel |= (stream == "landuse") & ~np.isin(names, COVER)
        else:
            sel |= stream == s
    return np.where(sel)[0]
HALF_LIVES = (0, 3, 10, 25)
Y0 = 1902


# ------------------------------------------------------------------------------------------------ covariates
def load_states(cells, y1=2025):
    """(n_years, n_cells, C) transformed states at the given grid cells from 1902, plus channel->stream map."""
    from src.community_encoder.train_DESK import covariate_io as cio
    schema = cio.load_schema(STATES)
    ydir = os.path.join(STATES, "yearly_states")
    years = list(range(Y0, y1 + 1))
    st = None
    for t, y in enumerate(years):
        s = cio.load_state_stack(y, ydir, schema)
        if st is None:
            st = np.full((len(years), len(cells), s.shape[-1]), np.nan, "float32")
        st[t] = s[cells[:, 0], cells[:, 1]]
    stream = np.empty(st.shape[-1], dtype=object)
    names = []
    for s in schema["streams"]:
        stream[int(s["start"]):int(s["end"])] = s["name"]
        v = s.get("variables") or [f"{s['name']}_{i}" for i in range(int(s["dim"]))]
        names += list(v) if len(v) == int(s["dim"]) else [f"{s['name']}_{i}" for i in range(int(s["dim"]))]
    names = np.array(names)
    # amendment 2: LUH2's primary / secondary split is land-use ACCOUNTING (land becomes secondary once harvested;
    # primf falls 0.36 -> 0.09 while primf + secdf stays ~0.38, 1902-2025) -- a clock. Physical cover = the totals.
    ix = {n: i for i, n in enumerate(names) if True}
    lu = lambda n: np.where((stream == "landuse") & (names == n))[0][0]
    forest = st[..., lu("primf")] + st[..., lu("secdf")]
    natnf = st[..., lu("primn")] + st[..., lu("secdn")]
    st = np.concatenate([st, forest[..., None], natnf[..., None]], axis=-1)
    stream = np.concatenate([stream, np.array(["landuse", "landuse"], dtype=object)])
    names = np.concatenate([names, ["forest", "natnonforest"]])
    del ix
    return st, stream, names, years


def ema(st, hl):
    if hl <= 0:
        return st
    a = 1.0 - 0.5 ** (1.0 / hl)
    out = np.empty_like(st)
    prev = st[0].copy()
    for t in range(len(st)):
        x = st[t]
        prev = np.where(np.isfinite(x), a * x + (1 - a) * np.where(np.isfinite(prev), prev, x), prev)
        out[t] = prev
    return out


def cell_mean(keys, X, mask):
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    uc, inv = np.unique(cid, return_inverse=True)
    w = np.asarray(mask, float)
    sums = np.zeros((len(uc), X.shape[1]))
    np.add.at(sums, inv, np.where(mask[:, None], X, 0.0))
    cnt = np.bincount(inv, weights=w, minlength=len(uc))
    return (sums / np.maximum(cnt, 1)[:, None])[inv]


class Cov:
    """Covariates at the cache's keys for each half-life: standardized, cell means, within PCs per stream group."""

    def __init__(self, keys, trn_mask, ty_mask):
        cells, cidx = np.unique(keys[:, :2], axis=0, return_inverse=True)
        self.cells, self.cidx = cells, cidx
        st, stream, names, years = load_states(cells)
        self.stream, self.names = stream, names
        self.dyn = np.array([s not in ("soil", "elevation") for s in stream])
        ti = keys[:, 2] - Y0
        raw = {}
        for hl in HALF_LIVES:
            S = ema(st, hl)
            raw[hl] = np.asarray(S[ti, cidx], "float64")
        fin = np.isfinite(raw[0]).all(1)
        self.fin = fin
        trn = trn_mask & fin
        self.mu = {hl: np.nanmean(raw[hl][trn], 0) for hl in HALF_LIVES}
        self.sd = {hl: np.nanstd(raw[hl][trn], 0) + 1e-9 for hl in HALF_LIVES}
        self.X, self.Xbar, self.W, self.V = {}, {}, {}, {}
        for hl in HALF_LIVES:
            Xs = np.where(np.isfinite(raw[hl]), (raw[hl] - self.mu[hl]) / self.sd[hl], 0.0)
            Xb = cell_mean(keys, Xs, ty_mask & fin)
            self.X[hl], self.Xbar[hl] = Xs, Xb
            D = Xs - Xb
            for g in GROUPS:
                cols = group_cols(stream, names, g)
                Dg = D[trn][:, cols]
                _, s_, Vt = np.linalg.svd(Dg[:: max(1, len(Dg) // 40000)], full_matrices=False)
                V = Vt[: NPC[g]].T
                sc = np.sqrt((Dg @ V).var(0)) + 1e-12                       # whiten: unit within-SD per PC
                self.V[(hl, g)] = (cols, V / sc)
                self.W[(hl, g)] = D[:, cols] @ (V / sc)
        st_cols = np.where(~self.dyn)[0]
        Sb = self.Xbar[0][:, st_cols]
        _, _, Vt = np.linalg.svd(Sb[trn][:: max(1, trn.sum() // 40000)] - Sb[trn].mean(0), full_matrices=False)
        self.static_pcs = (Sb - Sb[trn].mean(0)) @ Vt[:8].T
        self.years, self.st = years, st

    def w(self, hl, g="all"):
        if g == "all":
            return np.hstack([self.W[(hl, k)] for k in GROUPS])
        if g == "no_clock":
            return np.hstack([self.W[(hl, k)] for k in ("climate", "cover", "human")])
        return self.W[(hl, g)]

    def wbar(self, hl, g="all"):
        """Cell means projected on the SAME within loadings (for the between / space-for-time map)."""
        gs = list(GROUPS) if g == "all" else (["climate", "cover", "human"] if g == "no_clock" else [g])
        out = []
        for k in gs:
            cols, V = self.V[(hl, k)]
            out.append(self.Xbar[hl][:, cols] @ V)
        return np.hstack(out)


# ------------------------------------------------------------------------------------------------ estimand
class Estimand:
    def __init__(self, cache, n_boot, similarity):
        import pandas as pd
        from change_oracle import group_rows
        from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
        from src.data.preprocess.bbs_community import log1p_community
        c = lambda f: os.path.join(cache, f)
        self.cache = cache
        keys = np.load(c("keys.npy"))
        split = np.load(c("split.npz"))
        withheld = [int(y) for y in split["withheld"]]
        E, full, half_a, half_b = group_rows(cache, keys, withheld)
        idx = {}
        for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
            if ep == 0 and (bool(wh) or not withheld):
                idx.setdefault(cell, [None, None])[0] = j
            elif ep == 1 and not wh:
                idx.setdefault(cell, [None, None])[1] = j
        pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
        cells = E["cells"][pairs[:, 0]]
        ev = np.where(split["holdout"][cells[:, 0], cells[:, 1]])[0]
        self.keys, self.split, self.withheld = keys, split, withheld
        self.ho = split["holdout"][keys[:, 0], keys[:, 1]]
        self.bf = split["buffer"][keys[:, 0], keys[:, 1]]
        self.ty = ~np.isin(keys[:, 2], withheld)
        meta = json.load(open(c("meta.json"), encoding="utf-8"))
        self.species = list(meta["dev_species"])
        Xdev = np.asarray(np.load(c("X_dev.npy")), "float64")
        self.Y = np.asarray(log1p_community(Xdev), "float64")
        self.Zesk = np.asarray(np.load(c("esk_annual.npy"), mmap_mode="r")[:, :24], "float64")
        je, jm = pairs[ev, 0], pairs[ev, 1]
        self.early = [np.asarray(full[j]) for j in je]
        self.modern = [np.asarray(full[j]) for j in jm]
        self.cells_ev = cells[ev]
        T = {}
        T["dev"] = [epoch_mean_observed(Xdev, [h[j] for j in jm]) - epoch_mean_observed(Xdev, [h[j] for j in je])
                    for h in (half_a, half_b)]
        T["esk"] = [np.asarray(E[f"z_{h}"][jm, :24] - E[f"z_{h}"][je, :24], "float64") for h in ("a", "b")]
        self.T = T
        blk = (self.cells_ev[:, 0] // 6) * 100000 + self.cells_ev[:, 1] // 6
        ub, binv = np.unique(blk, return_inverse=True)
        rng = np.random.default_rng(11)
        self.Wb = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)),
                                                                size=n_boot)[:, binv]]).astype("float64")
        sim = pd.read_csv(similarity)
        d = sim[sim["group"] == "dev"].set_index("species_code").reindex(self.species)
        q = lambda col, p: d[col].quantile(p)
        self.strata = {"all": np.ones(len(d), bool), "sedentary": (d.migration == 1).values,
                       "resident": (d.migration <= 2).values, "migratory": (d.migration == 3).values,
                       "hofi_like": ((d.phylo_distance <= q("phylo_distance", .5))
                                     & (d.trait_distance <= q("trait_distance", .5)) & (d.migration <= 2)).values}
        self.strata = {k: np.asarray(v, bool) for k, v in self.strata.items()}

    def delta(self, F):
        """Each held-out cell's modern-minus-early mean of feature rows F."""
        return np.stack([F[m].mean(0) for m in self.modern]) - np.stack([F[e].mean(0) for e in self.early])

    def draws(self, P, target="dev", m=None, cells=None):
        """Per bootstrap draw: pooled noise-free corr and k over the species (columns) in m and the cells in ``cells``."""
        Ta, Tb = self.T[target]
        if m is not None:
            P, Ta, Tb = P[:, m], Ta[:, m], Tb[:, m]
        W = self.Wb if cells is None else self.Wb[:, cells]
        if cells is not None:
            P, Ta, Tb = P[cells], Ta[cells], Tb[cells]
        out = np.empty((len(W), 2))
        for i, w in enumerate(W):
            sw = w.sum()
            mu = lambda X: (w[:, None] * X).sum(0) / sw
            Pc, Ac, Bc = P - mu(P), Ta - mu(Ta), Tb - mu(Tb)
            vp = (w[:, None] * Pc * Pc).sum()
            vt = (w[:, None] * Ac * Bc).sum()
            cv = 0.5 * ((w[:, None] * Pc * Ac).sum() + (w[:, None] * Pc * Bc).sum())
            out[i] = (cv / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan, cv / vp if vp > 0 else np.nan)
        return out


def ci(v):
    return [float(np.nanpercentile(v[1:], 2.5)), float(np.nanpercentile(v[1:], 97.5))]


def fit_blocks(blocks, Y, rows):
    """Per-species BLR on concatenated feature blocks (each its own amplitude)."""
    from lib import blr
    X = np.hstack(blocks)
    b, s = [], 0
    for B in blocks:
        b.append((s, s + B.shape[1]))
        s += B.shape[1]
    return blr.fit(X[rows], Y[rows], b)


# ------------------------------------------------------------------------------------------------ part B
def arms_within(est, cov, target, Ytgt, extra=None):
    """All part-B arms for one target. Returns {arm: P (n_ev, S)} and side info."""
    keys = est.keys
    fin = cov.fin
    trn = np.where(~est.ho & ~est.bf & est.ty & fin)[0]
    Yw = Ytgt - cell_mean(keys, Ytgt, est.ty & fin)
    P, info = {}, {}

    def within(name, blocks, eval_blocks=None):
        m = fit_blocks(blocks, Yw, trn)
        Fe = np.hstack([est.delta(B) for B in (eval_blocks or blocks)])
        P[name] = Fe @ m["coef"].T
        info[name] = {"median_log_amp": np.median(np.log(m["a"]), 0).round(3).tolist()}
        return m

    # B1 streams (h = 10 as the default history; B2 varies it)
    for g in ("climate", "cover", "mgmt", "human", "all", "no_clock"):
        within(f"within_{g}_h10", [cov.w(10, g)])
    # B2 lag
    for hl in (0, 3, 25):
        within(f"within_all_h{hl}", [cov.w(hl)])
    within("within_all_h0+h10", [cov.w(0), cov.w(10)])
    # B3 space-for-time: between map on cell means (training cells, one row per cell), static PCs as controls
    Wb_ = cov.wbar(10)
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    _, first = np.unique(cid, return_index=True)
    trn_cells = np.intersect1d(first, np.where(~est.ho & ~est.bf & fin)[0])
    Ybar = cell_mean(keys, Ytgt, est.ty & fin)
    mb = fit_blocks([Wb_, cov.static_pcs], Ybar, trn_cells)
    beta_b = mb["coef"][:, : Wb_.shape[1]]                                       # (S, k): dynamic part only
    W10 = cov.w(10)
    dW = est.delta(W10)
    P["s4t_pure"] = dW @ beta_b.T
    off = 0
    for g in GROUPS:                                         # per-group space-for-time (coefficients from the joint fit)
        k = NPC[g]
        P[f"s4t_{g}"] = dW[:, off:off + k] @ beta_b[:, off:off + k].T
        off += k
    s4t_rows = np.einsum("nk,sk->ns", W10, beta_b)                               # per species scalar feature
    # anchored: y_w = kappa * (W beta_b) + W gamma; per species its own scalar column -> fit species one at a time
    # in a batched way: BLR needs a shared X, so use the prior-centred form (residual on W) and report kappa by LS
    Yres = Yw - s4t_rows
    m = fit_blocks([W10], Yres, trn)
    P["anchored_prior_s4t"] = dW @ (beta_b + m["coef"]).T
    num = (s4t_rows[trn] * Yw[trn]).sum(0)
    den = (s4t_rows[trn] ** 2).sum(0)
    info["kappa_ls"] = {"median": float(np.median(num / np.maximum(den, 1e-12))),
                        "quartiles": np.percentile(num / np.maximum(den, 1e-12), [25, 75]).tolist()}
    # B4 context (varying coefficient)
    ctx_cov = cov.static_pcs[:, :2]
    ctx = np.hstack([ctx_cov, cov.wbar(10)[:, :2]])
    ctx = (ctx - ctx[trn].mean(0)) / (ctx[trn].std(0) + 1e-9)
    VC = np.hstack([W10] + [W10 * ctx[:, [j]] for j in range(ctx.shape[1])])
    within("vc_cov_h10", [W10, VC[:, W10.shape[1]:]])
    if extra is not None and "zbar" in extra:
        zc = extra["zbar"][:, :4]
        zc = (zc - zc[trn].mean(0)) / (zc[trn].std(0) + 1e-9)
        VD = np.hstack([W10 * zc[:, [j]] for j in range(zc.shape[1])])
        within("vc_desk_h10", [W10, VD])
    # B5 nonlinear (g(0) = 0): RFF(W) - RFF(0)
    rng = np.random.default_rng(3)
    for ell in (2.0, 6.0):
        om = rng.normal(size=(W10.shape[1], 256)) / ell
        ph = rng.uniform(0, 2 * np.pi, 256)
        R = np.sqrt(2 / 256) * (np.cos(W10 @ om + ph) - np.cos(ph))
        within(f"rff{ell:g}_h10", [R])
        within(f"lin+rff{ell:g}_h10", [W10, R])
    # J: delta learned from the COMMUNITY (ESK r24 within deviations ~ W, ridge), then the target's own within readout
    # on that shared 24-dim delta -- what a two-part DESK would do (one representation, species read out by BLR)
    if extra is not None and "Zesk" in extra:
        Zw = extra["Zesk"] - cell_mean(keys, extra["Zesk"], est.ty & fin)
        for nm, Wf in (("all", W10), ("no_clock", cov.w(10, "no_clock"))):
            A = Wf[trn]
            lam = 1e-1 * np.trace(A.T @ A) / A.shape[1]
            Bc = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Zw[trn])
            dlt = Wf @ Bc                                                       # (N, 24) community-learned delta
            within(f"comm_delta_{nm}_h10", [dlt])
    # B6 DESK
    if extra is not None and "z" in extra:
        z, zbar = extra["z"], extra["zbar"]
        zw = z - zbar
        within("desk_within", [zw])
        within("within_all_h10+desk", [W10, zw])
        within("within_no_clock_h10+desk", [cov.w(10, "no_clock"), zw])
        # production split readout (levels), its within coefficients applied to DESK's change
        ms = fit_blocks([zbar, zw], Ytgt, trn)
        P["desk_split_level"] = est.delta(z) @ ms["coef"][:, zbar.shape[1]:].T
        # how much of DESK's within change is linear in covariate change?
        A = W10[trn]
        coef, *_ = np.linalg.lstsq(np.hstack([A, np.ones((len(A), 1))]), zw[trn], rcond=None)
        fit_ = np.hstack([W10, np.ones((len(W10), 1))]) @ coef
        r2 = 1 - ((zw[trn] - fit_[trn]) ** 2).sum() / ((zw[trn] - zw[trn].mean(0)) ** 2).sum()
        info["desk_dz_r2_on_W10"] = float(r2)
        within("desk_within_covpart", [fit_], [fit_])
        within("desk_within_residpart", [zw - fit_])
    # B7 placebo (position RFF x time) and placebo + covariates
    t = keys[:, 2].astype(float)
    tbar = cell_mean(keys, t[:, None], est.ty & fin)[:, 0]
    dt = (t - tbar) / 10.0
    pls = []
    for sd in (1, 2):
        rg = np.random.default_rng(sd)
        om, ph = rg.normal(size=(2, 96)) / 300.0, rg.uniform(0, 2 * np.pi, 96)
        Rp = np.sqrt(2 / 96) * np.cos(keys[:, :2].astype(float) * 27.0 @ om + ph) * dt[:, None]
        pls.append(Rp)
    for i, Rp in enumerate(pls):
        within(f"placebo#{i}", [Rp])
        within(f"placebo+within_all_h10#{i}", [Rp, W10])
    for nm in ("placebo", "placebo+within_all_h10"):
        P[nm] = 0.5 * (P.pop(f"{nm}#0") + P.pop(f"{nm}#1"))
    # B8 attribution of within_all_h10's predicted change by stream (columns of W10 by group)
    m = fit_blocks([W10], Yw, trn)
    off, contrib = 0, {}
    for g in GROUPS:
        k = NPC[g]
        contrib[g] = est.delta(W10)[:, off:off + k] @ m["coef"][:, off:off + k].T
        off += k
    info["_contrib"] = contrib
    return P, info


def report(est, P, target, strata, order_ref="within_all_h10"):
    res = {}
    for sname in strata:
        m = est.strata[sname] if target == "dev" else None
        if m is not None and m.sum() < 5:
            continue
        D = {k: est.draws(v, target, m) for k, v in P.items()}
        o = {"n_species": int(m.sum()) if m is not None else 24,
             "arms": {k: {"corr": float(v[0, 0]), "ci": ci(v[:, 0]), "k": float(v[0, 1])} for k, v in D.items()}}
        pairs_ = [("comm_delta_all_h10", "desk_within"), ("comm_delta_no_clock_h10", "desk_within"),
                  ("comm_delta_all_h10", "within_all_h10"), ("comm_delta_no_clock_h10", "within_no_clock_h10"),
                  ("comm_delta_no_clock_h10", "desk_split_level"),
                  ("within_all_h10", "desk_within"), ("within_all_h10", "desk_split_level"),
                  ("within_all_h10+desk", "within_all_h10"), ("within_all_h10+desk", "desk_within"),
                  ("anchored_prior_s4t", "within_all_h10"), ("s4t_pure", "within_all_h10"),
                  ("vc_cov_h10", "within_all_h10"), ("vc_desk_h10", "within_all_h10"),
                  ("rff2_h10", "within_all_h10"), ("rff6_h10", "within_all_h10"),
                  ("lin+rff2_h10", "within_all_h10"), ("within_all_h0+h10", "within_all_h10"),
                  ("placebo+within_all_h10", "placebo"), ("placebo", "within_all_h10"),
                  ("desk_within_covpart", "desk_within"), ("desk_within_residpart", "desk_within")]
        for g in ("climate", "cover", "mgmt", "human", "no_clock"):
            pairs_.append((f"within_{g}_h10", "within_all_h10"))
        pairs_ += [("within_no_clock_h10", "desk_within"), ("within_no_clock_h10", "desk_split_level"),
                   ("within_no_clock_h10+desk", "within_no_clock_h10"), ("within_no_clock_h10+desk", "desk_within")]
        for hl in (0, 3, 25):
            pairs_.append((f"within_all_h{hl}", "within_all_h10"))
        o["diffs"] = {}
        for x, y in pairs_:
            if x in D and y in D:
                d = D[x][:, 0] - D[y][:, 0]
                o["diffs"][f"{x} - {y}"] = {"diff": float(d[0]), "ci": ci(d)}
        # F: by Great Plains zone (held-out cells), key arms only
        if getattr(est, "zone", None) is not None:
            o["by_zone"] = {}
            for zn in ("west", "great_plains", "east"):
                cells = np.where(est.zone == zn)[0]
                if len(cells) < 15:
                    continue
                o["by_zone"][zn] = {"n_cells": int(len(cells))}
                for k in ("within_all_h10", "within_no_clock_h10", "comm_delta_no_clock_h10", "desk_within",
                          "desk_split_level", "placebo", "s4t_pure", "anchored_prior_s4t"):
                    if k in P:
                        Dz = est.draws(P[k], target, m, cells)
                        o["by_zone"][zn][k] = {"corr": float(Dz[0, 0]), "ci": ci(Dz[:, 0]), "k": float(Dz[0, 1])}
        res[sname] = o
    return res


def attach_zones(est):
    try:
        import rasterio
        with rasterio.open(ZONES_TIF) as src:
            zg = src.read(1)
        est.zone = np.array([ZONES.get(int(zg[r, c]), "other") for r, c in est.cells_ev])
    except Exception as e:                                                    # zones are a report, not a requirement
        print(f"[E073] zones unavailable: {e}")
        est.zone = None


def run_score(a):
    est = Estimand(a.cache, a.n_boot, a.similarity)
    attach_zones(est)
    trn_mask = ~est.ho & ~est.bf & est.ty
    cov = Cov(est.keys, trn_mask, est.ty)
    z = np.asarray(np.load(os.path.join(a.cache, "z_raw.npy"), mmap_mode="r")[:, :24], "float64")
    z = np.where(np.isfinite(z), z, 0.0)
    zbar = cell_mean(est.keys, z, est.ty & cov.fin)
    extra = {"z": z, "zbar": zbar, "Zesk": est.Zesk}
    out = {"cache": a.cache, "n_eval_cells": int(len(est.early)), "planted": bool(a.planted), "targets": {}}
    targets = [("dev", est.Y), ("esk", est.Zesk)]
    if a.planted:
        targets = [("dev", planted_species(est, cov))]
        est.T["dev"] = planted_truth(est, targets[0][1])
    for target, Ytgt in targets:
        P, info = arms_within(est, cov, target, Ytgt, extra)
        contrib = info.pop("_contrib")
        strata = list(est.strata) if target == "dev" else ["all"]
        out["targets"][target] = {"scores": report(est, P, target, strata), "info": info}
        if target == "dev":
            # B8: share of within_all_h10's predicted change variance by stream, per stratum (pooled over species)
            sh = {}
            for sname in ("all", "sedentary"):
                if sname not in est.strata:
                    continue
                m = est.strata[sname]
                tot = sum(((v[:, m] - v[:, m].mean(0)) ** 2).sum() for v in contrib.values())
                sh[sname] = {g: float(((v[:, m] - v[:, m].mean(0)) ** 2).sum() / tot) for g, v in contrib.items()}
            out["targets"][target]["stream_share_of_predicted_change"] = sh
        np.savez_compressed(os.path.join(a.out, f"preds_{target}.npz"), **{k.replace("+", "_PLUS_"): v.astype("float32")
                                                                          for k, v in P.items()})
        summarize_target(out["targets"][target], target)
    json.dump(out, open(os.path.join(a.out, "e073_score.json"), "w"), indent=1)


def summarize_target(o, target):
    print(f"== target {target}")
    for sname, r in o["scores"].items():
        arms = sorted(r["arms"].items(), key=lambda kv: -np.nan_to_num(kv[1]["corr"], nan=-9))
        print(f"  {sname} (n {r['n_species']}): " + "  ".join(f"{k} {v['corr']:.3f}(k{v['k']:.2f})" for k, v in arms))
        for k, v in r["diffs"].items():
            print(f"     {k:44s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    print(f"  info {json.dumps(o['info'])[:1200]}")
    if "stream_share_of_predicted_change" in o:
        print(f"  stream share {o['stream_share_of_predicted_change']}")


# ------------------------------------------------------------------------------------------------ verification (a)
def planted_species(est, cov):
    """60 planted species: a cell effect + a LINEAR response to the h=10 within features of ONE stream group (20 each)
    + annual noise. The arm matching the planted stream / history must win; s4t / placebo must not."""
    rng = np.random.default_rng(42)
    n = len(est.keys)
    cid = est.keys[:, 0].astype(np.int64) * 100000 + est.keys[:, 1]
    _, inv = np.unique(cid, return_inverse=True)
    cols = []
    for g in ("climate", "cover", "human"):
        Wg = cov.w(10, g)
        for _ in range(20):
            b = rng.normal(size=Wg.shape[1]) / np.sqrt(Wg.shape[1])
            sig = Wg @ b
            cell = rng.normal(size=inv.max() + 1)[inv]
            cols.append(cell + 0.6 * sig / (sig.std() + 1e-9))
    S = np.stack(cols, 1)
    est._planted_signal = S
    est.strata = {"all": np.ones(60, bool), "planted_climate": np.arange(60) < 20,
                  "planted_cover": (np.arange(60) >= 20) & (np.arange(60) < 40), "planted_human": np.arange(60) >= 40}
    est.species = [f"planted{i}" for i in range(60)]
    return S + rng.normal(size=S.shape) * 1.0


def planted_truth(est, Y):
    rng = np.random.default_rng(43)
    S = est._planted_signal
    out = []
    for _ in range(2):
        T = np.stack([S[m].mean(0) for m in est.modern]) - np.stack([S[e].mean(0) for e in est.early])
        out.append(T + rng.normal(size=T.shape) * 0.3)
    return out


# ------------------------------------------------------------------------------------------------ part C
def run_holdout(a):
    """Covariate-extrapolation holdouts on a cache whose readout rows span all years (base): training CELLS are
    restricted by their covariate trend along a direction u (leading within-PC of landuse / climate); held-out cells
    are scored by where their own trend falls. Penalty = restricted-fit score - full-fit score (paired draws)."""
    est = Estimand(a.cache, a.n_boot, a.similarity)
    keys = est.keys
    trn_mask = ~est.ho & ~est.bf & est.ty
    cov = Cov(keys, trn_mask, est.ty)
    fin = cov.fin
    z = np.asarray(np.load(os.path.join(a.cache, "z_raw.npy"), mmap_mode="r")[:, :24], "float64")
    z = np.where(np.isfinite(z), z, 0.0)
    zbar = cell_mean(keys, z, est.ty & fin)
    W10 = cov.w(10)
    Yw = est.Y - cell_mean(keys, est.Y, est.ty & fin)
    t = keys[:, 2].astype(float)
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    uc, inv = np.unique(cid, return_inverse=True)
    ev_cid = est.cells_ev[:, 0].astype(np.int64) * 100000 + est.cells_ev[:, 1]
    ev_ci = np.searchsorted(uc, ev_cid)
    rng = np.random.default_rng(3)
    om = rng.normal(size=(W10.shape[1], 256)) / 2.0
    ph = rng.uniform(0, 2 * np.pi, 256)
    R = np.sqrt(2 / 256) * (np.cos(W10 @ om + ph) - np.cos(ph))
    arm_blocks = {"within_all_h10": [W10], "within_no_clock_h10": [cov.w(10, "no_clock")],
                  "within_cover_h10": [cov.w(10, "cover")], "within_mgmt_h10": [cov.w(10, "mgmt")],
                  "within_climate_h10": [cov.w(10, "climate")], "rff2_h10": [R], "desk_within": [z - zbar]}
    out = {"cache": a.cache, "splits": {}}
    for g in ("cover", "mgmt", "climate"):
        u_proj = cov.W[(10, g)][:, 0]                                            # leading within PC of the group
        # per-cell trend (slope per decade) of the projection over the cell's rows
        tt = t - cell_mean(keys, t[:, None], fin)[:, 0]
        num = np.bincount(inv, weights=np.where(fin, tt * u_proj, 0), minlength=len(uc))
        den = np.bincount(inv, weights=np.where(fin, tt * tt, 0), minlength=len(uc))
        slope = 10 * num / np.maximum(den, 1e-9)
        trn_cells_mask = np.zeros(len(uc), bool)
        trn_cells_mask[np.unique(inv[np.where(trn_mask & fin)[0]])] = True
        tr_sl = slope[trn_cells_mask]
        thr = np.percentile(np.abs(tr_sl), 70)
        sev = slope[ev_ci]
        designs = {
            "magnitude": (np.abs(slope) <= thr, {"beyond": np.abs(sev) > thr, "inside": np.abs(sev) <= thr}),
            "sign_pos_only": (slope > 0, {"opposite": sev < 0, "same": sev > 0}),
            "sign_neg_only": (slope < 0, {"opposite": sev > 0, "same": sev < 0})}
        for dname, (keep_cells, groups) in designs.items():
            rows_full = np.where(trn_mask & fin)[0]
            rows_rest = rows_full[keep_cells[inv[rows_full]]]
            res = {"n_train_rows_full": int(len(rows_full)), "n_train_rows_restricted": int(len(rows_rest)),
                   "n_eval": {k: int(v.sum()) for k, v in groups.items()}, "arms": {}}
            for arm, blocks in list(arm_blocks.items()) + [("comm_delta_no_clock_h10", None)]:
                if blocks is None:                       # community-learned delta, learned on the same rows as the fit
                    Wf = cov.w(10, "no_clock")
                    Zw = est.Zesk - cell_mean(keys, est.Zesk, est.ty & fin)

                    def cd(rows):
                        A = Wf[rows]
                        lam = 1e-1 * np.trace(A.T @ A) / A.shape[1]
                        return Wf @ np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Zw[rows])
                    df, dr = cd(rows_full), cd(rows_rest)
                    Pf = est.delta(df) @ fit_blocks([df], Yw, rows_full)["coef"].T
                    Pr = est.delta(dr) @ fit_blocks([dr], Yw, rows_rest)["coef"].T
                else:
                    Fe = np.hstack([est.delta(B) for B in blocks])
                    Pf = Fe @ fit_blocks(blocks, Yw, rows_full)["coef"].T
                    Pr = Fe @ fit_blocks(blocks, Yw, rows_rest)["coef"].T
                ar = {}
                for gname, cm in groups.items():
                    cells = np.where(cm)[0]
                    if len(cells) < 15:
                        continue
                    for sname in ("all", "sedentary"):
                        m = est.strata[sname]
                        Df, Dr = est.draws(Pf, "dev", m, cells), est.draws(Pr, "dev", m, cells)
                        ar[f"{gname}|{sname}"] = {"full": float(Df[0, 0]), "restricted": float(Dr[0, 0]),
                                                  "penalty": float(Dr[0, 0] - Df[0, 0]),
                                                  "penalty_ci": ci(Dr[:, 0] - Df[:, 0]),
                                                  "k_full": float(Df[0, 1]), "k_restricted": float(Dr[0, 1])}
                res["arms"][arm] = ar
                print(f"[E073 C] {g} {dname} {arm}: " + "; ".join(f"{k} {v['full']:.3f}->{v['restricted']:.3f} "
                      f"({v['penalty']:+.3f} [{v['penalty_ci'][0]:+.3f},{v['penalty_ci'][1]:+.3f}]) k {v['k_full']:.2f}->"
                      f"{v['k_restricted']:.2f}" for k, v in ar.items()), flush=True)
            out["splits"][f"{g}|{dname}"] = res
    json.dump(out, open(os.path.join(a.out, "e073_holdout.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ part A
def run_novelty(a):
    import rasterio
    keys = np.load(os.path.join(a.cache, "keys.npy"))
    split = np.load(os.path.join(a.cache, "split.npz"))
    cells = np.unique(keys[:, :2], axis=0)
    st, stream, names, years = load_states(cells)
    with rasterio.open(ZONES_TIF) as src:
        zg = src.read(1)
    zone = np.array([ZONES.get(int(zg[r, c]), "other") for r, c in cells])
    yi = lambda y0, y1: slice(y0 - Y0, y1 - Y0 + 1)
    ref = st[yi(1966, 2025)]
    mu = np.nanmean(ref.reshape(-1, ref.shape[-1]), 0)
    sd = np.nanstd(ref.reshape(-1, ref.shape[-1]), 0) + 1e-9
    S = (st - mu) / sd
    cm = np.nanmean(S[yi(1966, 2025)], 0)                                  # each cell's 1966-2025 mean
    # 6-year windows in 1966-2025 (10 windows) and the 1920-39 mean
    wins = [(y, y + 5) for y in range(1966, 2026, 6)]
    Wn = np.stack([np.nanmean(S[yi(*w)], 0) for w in wins])               # (10, cells, C)
    early = np.nanmean(S[yi(1920, 1939)], 0)
    out = {"n_cells": int(len(cells)), "zones": {z: int((zone == z).sum()) for z in ZONES.values()}, "streams": {}}
    for g in list(GROUPS) + ["all_dynamic", "no_clock"]:
        if g == "all_dynamic":
            cols = np.where(np.isin(stream, ("climate", "landuse", "hyde", "bui")))[0]
        elif g == "no_clock":
            cols = np.concatenate([group_cols(stream, names, k) for k in ("climate", "cover", "human")])
        else:
            cols = group_cols(stream, names, g)
        Dref = (Wn[:, :, cols] - cm[None, :, cols]).reshape(-1, len(cols))
        Dref = Dref[np.isfinite(Dref).all(1)]
        _, s_, Vt = np.linalg.svd(Dref - Dref.mean(0), full_matrices=False)
        k = min(NPC.get(g, 30), len(cols))
        V = Vt[:k].T
        sc = (Dref @ V).std(0) + 1e-12
        m_ref = (((Dref @ V) / sc) ** 2).mean(1)
        p99 = np.percentile(m_ref, 99)
        De = early[:, cols] - cm[:, cols]
        ok = np.isfinite(De).all(1)
        m_e = np.full(len(cells), np.nan)
        m_e[ok] = (((De[ok] @ V) / sc) ** 2).mean(1)
        # within-place share of variance, 1966-2025 and 1902-2025 (annual)
        def wshare(sl):
            A = S[sl][:, :, cols]
            tot = np.nanvar(A.reshape(-1, len(cols)), 0).sum()
            wi = np.nanvar((A - np.nanmean(A, 0)[None]).reshape(-1, len(cols)), 0).sum()
            return float(wi / tot)
        o = {"within_share_1966_2025": wshare(yi(1966, 2025)), "within_share_1902_2025": wshare(yi(1902, 2025)),
             "ref_p99_mahal": float(p99), "by_zone": {}}
        for zname in list(ZONES.values()) + ["all"]:
            zm = (zone == zname) if zname != "all" else np.ones(len(cells), bool)
            v = m_e[zm & ok]
            if len(v) < 10:
                continue
            o["by_zone"][zname] = {"median_mahal_1920_39": float(np.median(v)),
                                   "median_ratio_to_ref_median": float(np.median(v) / np.median(m_ref)),
                                   "share_beyond_ref_p99": float((v > p99).mean()),
                                   "pc1_sd_units_median": float(np.median(((De[zm & ok] @ V) / sc)[:, 0]))}
        out["streams"][g] = o
        print(f"[E073 A] {g}: within share {o['within_share_1966_2025']:.3f} (1966-2025) "
              f"{o['within_share_1902_2025']:.3f} (1902-2025); 1920-39 novelty by zone "
              + "; ".join(f"{z} med {v['median_ratio_to_ref_median']:.1f}x ref, beyond p99 {v['share_beyond_ref_p99']:.2f}"
                          for z, v in o["by_zone"].items()), flush=True)
    # analog support: 1920-39 full state (dynamic + static) vs 1966-2025 windows of OTHER cells, top-20 PCs
    cols = np.arange(S.shape[-1])
    # each stream weighted equally (otherwise the 240 climate channels dominate the distance)
    sw = np.array([1.0 / np.sqrt((stream == s).sum()) for s in stream])
    Wn, early = Wn * sw, early * sw
    R_ = Wn[:, :, cols].reshape(-1, len(cols))
    owner = np.tile(np.arange(len(cells)), len(wins))
    okr = np.isfinite(R_).all(1)
    R_, owner = R_[okr], owner[okr]
    _, _, Vt = np.linalg.svd(R_[:: max(1, len(R_) // 20000)] - R_.mean(0), full_matrices=False)
    V = Vt[:20].T
    Rp, Ep = (R_ - R_.mean(0)) @ V, (early - R_.mean(0)) @ V
    from scipy.spatial import cKDTree
    tree = cKDTree(Rp)

    def nn_other(Q, q_owner, k=40):
        d, i = tree.query(Q, k=k)
        o = owner[i]
        d = np.where(o == q_owner[:, None], np.inf, d)
        return d.min(1)
    rng = np.random.default_rng(0)
    samp = rng.choice(len(Rp), size=min(8000, len(Rp)), replace=False)
    d_ref = nn_other(Rp[samp], owner[samp])
    oke = np.isfinite(Ep).all(1)
    d_e = np.full(len(cells), np.nan)
    d_e[oke] = nn_other(Ep[oke], np.arange(len(cells))[oke])
    p95 = np.percentile(d_ref[np.isfinite(d_ref)], 95)
    out["analog_support"] = {"ref_median_nn": float(np.nanmedian(d_ref)), "ref_p95_nn": float(p95), "by_zone": {}}
    for zname in list(ZONES.values()) + ["all"]:
        zm = ((zone == zname) if zname != "all" else np.ones(len(cells), bool)) & oke
        if zm.sum() < 10:
            continue
        out["analog_support"]["by_zone"][zname] = {"median_nn_ratio": float(np.median(d_e[zm]) / np.nanmedian(d_ref)),
                                                   "share_beyond_ref_p95": float((d_e[zm] > p95).mean())}
    print(f"[E073 A] analog support of the 1920-39 state among other cells' 1966-2025 windows: "
          f"{out['analog_support']['by_zone']}", flush=True)
    Wn, early = Wn / sw, early / sw
    # direction: land-use channel differences (1920-39 minus 1996-2025), raw fractions, by zone
    lu = np.where(stream == "landuse")[0]
    d_lu = np.nanmean(st[yi(1920, 1939)][:, :, lu], 0) - np.nanmean(st[yi(1996, 2025)][:, :, lu], 0)
    out["landuse_direction"] = {}
    for zname in list(ZONES.values()) + ["all"]:
        zm = (zone == zname) if zname != "all" else np.ones(len(cells), bool)
        v = np.nanmean(d_lu[zm], 0)
        top = np.argsort(-np.abs(v))[:8]
        out["landuse_direction"][zname] = {str(names[lu][j]): float(v[j]) for j in top}
    print(f"[E073 A] land use 1920-39 minus 1996-2025 (mean raw units): {json.dumps(out['landuse_direction'])}")
    # per-channel novelty for land COVER and human channels: 1920-39 deviation from the cell's 1966-2025 mean in units of
    # the channel's SD of 1966-2025 six-year-window deviations (pooled over cells), median by zone
    out["channel_novelty_sd"] = {}
    chans = np.concatenate([group_cols(stream, names, "cover"), group_cols(stream, names, "human")])
    Dw = (Wn[:, :, chans] - cm[None, :, chans]).reshape(-1, len(chans))
    sdw = np.nanstd(Dw, 0) + 1e-12
    De = (early[:, chans] - cm[:, chans]) / sdw
    for zname in list(ZONES.values()) + ["all"]:
        zm = (zone == zname) if zname != "all" else np.ones(len(cells), bool)
        out["channel_novelty_sd"][zname] = {str(names[j]): float(np.nanmedian(De[zm, i])) for i, j in enumerate(chans)}
    print(f"[E073 A] cover/human channel novelty (median SD units of 1966-2025 window deviations): "
          f"{json.dumps({z: {k: round(v, 1) for k, v in d.items()} for z, d in out['channel_novelty_sd'].items()})}")
    # does BBS 1966-2025 within-place change already contain the 1920->2000 DIRECTION? cosine of each cell's
    # 1920-39 deviation with its own 1966-71 deviation, per stream group (positive: the BBS era starts partway along it)
    out["direction_overlap"] = {}
    for g in GROUPS:
        cols = group_cols(stream, names, g)
        A_ = early[:, cols] - cm[:, cols]
        B_ = Wn[0][:, cols] - cm[:, cols]
        ok_ = np.isfinite(A_).all(1) & np.isfinite(B_).all(1)
        cos = (A_[ok_] * B_[ok_]).sum(1) / (np.linalg.norm(A_[ok_], axis=1) * np.linalg.norm(B_[ok_], axis=1) + 1e-12)
        ratio = np.linalg.norm(A_[ok_], axis=1) / (np.linalg.norm(B_[ok_], axis=1) + 1e-12)
        out["direction_overlap"][g] = {"median_cos_1920_vs_1966": float(np.median(cos)),
                                       "median_norm_ratio_1920_over_1966": float(np.median(ratio))}
    print(f"[E073 A] direction overlap: {out['direction_overlap']}")
    json.dump(out, open(os.path.join(a.out, "e073_novelty.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=("novelty", "score", "holdout"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--similarity", default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--planted", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"novelty": run_novelty, "score": run_score, "holdout": run_holdout}[a.part](a)


if __name__ == "__main__":
    main()
