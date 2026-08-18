"""Per-axis top-N community selection with a migration rank penalty.

The reference community is the UNION of four explicit top-N cuts, one per axis, taken
over a pool already gated on trend-product availability. A union rather than a composite
ranking because the axes measure incommensurable things -- two reward *proximity to the
focal*, two reward position on the urban gradient -- so averaging their ranks yields a
scalar nothing is actually ranked on. The union also preserves WHICH axis chose each
species (``category``), which a mean rank destroys.

Migrants are downweighted by an additive rank penalty, because Z is meant to encode
local, year-round habitat suitability inferred from year-round covariates against a June
breeding count. A long-distance migrant's June abundance is jointly governed by
wintering-ground and flyway dynamics the model has no covariates for, so migrant
abundance injects variance the kernel attributes to local habitat.

The penalty is not cosmetic on the morphology axis in particular: ``Wing.Length``,
``Kipps.Distance``, ``Secondary1`` and ``Hand-Wing.Index`` are near-collinear, so wing
shape carries ~4 of 11 traits and the axis is effectively a wing-shape axis -- and wing
shape is a migration proxy. Left alone that axis ENRICHES long-distance migrants relative
to the candidate pool.

Reading the penalty
-------------------
The penalty is added to a species' rank *within the whole gated pool* to form a sort key;
each axis then takes the N smallest keys. The trap is to read the cut as "key <= N": it is
not. The Nth-smallest KEY is well above N, and it RISES with the penalty, because penalised
species vacate slots that unpenalised species fill from deeper down the raw ranking. One
measured axis at N=30, p=30 admits a class-3 species whose raw rank is 3 (key 63) at
position 26 of 30, with a threshold key of 73.

So the legible quantity is not the penalty but the **admission bar in raw-rank units** --
what raw rank each migration class needs to place. See ``admission_bars``. Raising p never
removes a sedentary species; class 1 is never penalised, so its bar only loosens (measured:
30 -> 73 -> 104 as p goes 0 -> 30 -> 60) while the migratory bar tightens (30 -> 13 -> 0).

Because pool-rank units are an arbitrary scale -- if the gate changes or the pool grows, a
fixed p silently means something different -- callers state the target composition and
``solve_penalty`` recovers the smallest p that achieves it.

Pure and numpy-only on purpose: the whole selection rule is unit-testable without pandas,
rasterio, the network, or the BBS/eBird gate artifacts.
"""
import numpy as np

# AVONET ``Migration``: 1 = sedentary, 2 = partial, 3 = migratory.
SEDENTARY, PARTIAL, MIGRATORY = 1, 2, 3
MIGRATION_LABELS = {SEDENTARY: "sedentary", PARTIAL: "partial", MIGRATORY: "migratory"}

# The four selection axes. ``column`` names a key in the pool dict; ``ascending``
# says whether a SMALL value is better. The two urban axes read the same SIGNED
# column from opposite ends, which is what makes both tails reachable. Ranking one
# absolute deviation-from-median column descending does NOT achieve the same thing:
# the tolerance distribution is asymmetric (min -1.79, median -0.09, max 5.00), so a
# single extremeness cut lands ~9:1 on urban-lovers.
AXES = (
    {"name": "phylo", "column": "phylo_distance", "ascending": True,
     "note": "nearest the focal on the patristic tree"},
    {"name": "morph", "column": "trait_distance", "ascending": True,
     "note": "nearest the focal in AVONET morphospace"},
    {"name": "urban_loving", "column": "urban_tolerance", "ascending": False,
     "note": "highest urban tolerance"},
    {"name": "urban_avoiding", "column": "urban_tolerance", "ascending": True,
     "note": "lowest urban tolerance"},
)


def rank_average(values, ascending=True):
    """Ranks 1..n with ties averaged (matches ``pandas.Series.rank``).

    Rank here rather than reusing AVONET's precomputed ``*.Rank`` columns: those are
    ranked over the FULL table, while the penalty has to act on ranks within the
    *gated* pool, or p means a different thing on every axis. Ranking here also puts
    the two urban axes -- which arrive as signed values, not ranks -- on the same
    scale as the two distance axes, so one p is commensurable across all four.
    """
    v = np.asarray(values, dtype="float64")
    if v.ndim != 1:
        raise ValueError(f"rank_average expects 1-D values, got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError("rank_average requires finite values; drop or impute "
                         "non-finite rows when the pool is built, so the loss is "
                         "reported rather than silently ranked last")
    if not ascending:
        v = -v
    order = np.argsort(v, kind="stable")
    positions = np.arange(1, v.size + 1, dtype="float64")
    ranked = np.empty(v.size, dtype="float64")
    sorted_v = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranked[i:j + 1] = positions[i:j + 1].mean()          # average over the tie block
        i = j + 1
    out = np.empty(v.size, dtype="float64")
    out[order] = ranked
    return out


def migration_penalty(migration, p):
    """Rank penalty per species: 0 / p / 2p for sedentary / partial / migratory.

    Linear in the AVONET class so one number controls the whole gradient, and
    zero for class 1 so raising p can never displace a sedentary species.
    """
    m = np.asarray(migration)
    if not np.isin(m, (SEDENTARY, PARTIAL, MIGRATORY)).all():
        bad = sorted(set(np.asarray(m).ravel().tolist()) - {SEDENTARY, PARTIAL, MIGRATORY})
        raise ValueError(f"migration must be 1/2/3 (AVONET Migration); got {bad}")
    return (m.astype("float64") - 1.0) * float(p)


def admission_bars(threshold_key, p):
    """Per-class admission bar in RAW-RANK units, given a realized threshold key.

    A class-c species places iff its raw pool rank is at most
    ``threshold_key - (c-1)*p``. This is the interpretable form of the rule: at
    p=30 one measured axis admitted sedentary species on raw rank <= 73, partial
    on <= 43 and migratory on <= 13. Exact except at the threshold itself, where
    the tie-break in ``select_axis`` decides.

    A bar <= 0 means the class cannot place on this axis at all.
    """
    return {c: float(threshold_key) - (c - 1) * float(p)
            for c in (SEDENTARY, PARTIAL, MIGRATORY)}


def select_axis(codes, values, migration, ascending, n, p):
    """Top-``n`` on one axis after the migration penalty.

    Returns a dict with the selected ``codes`` (best first), the realized
    ``threshold_key``, the per-class ``bars``, and the per-species ``raw_rank`` /
    ``key`` so callers can log or test the mechanism.

    Ties on the key are broken toward the SMALLER penalty, then by species code.
    Breaking toward the better raw rank instead would partly undo the penalty --
    at equal key the migrant is by construction the one with the better raw rank
    -- and an unspecified tie-break would make selection non-reproducible.
    """
    codes = list(codes)
    if not (len(codes) == len(values) == len(migration)):
        raise ValueError(f"ragged pool: {len(codes)} codes, {len(values)} values, "
                         f"{len(migration)} migration classes")
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    raw = rank_average(values, ascending=ascending)
    pen = migration_penalty(migration, p)
    key = raw + pen
    order = sorted(range(len(codes)), key=lambda i: (key[i], pen[i], codes[i]))
    take = order[:min(n, len(order))]
    threshold_key = float(key[take[-1]]) if take else float("nan")
    return {
        "codes": [codes[i] for i in take],
        "indices": take,
        "threshold_key": threshold_key,
        "bars": admission_bars(threshold_key, p),
        "raw_rank": raw,
        "key": key,
        "penalty": pen,
        "n_requested": n,
        "n_available": len(codes),
    }


def composition(migration):
    """Counts and fractions by migration class (fractions are 0 for an empty set)."""
    m = np.asarray(list(migration))
    total = int(m.size)
    counts = {c: int((m == c).sum()) for c in (SEDENTARY, PARTIAL, MIGRATORY)}
    fracs = {c: (counts[c] / total if total else 0.0) for c in counts}
    return {"n": total, "counts": counts, "fracs": fracs}


def interleaved_order(axis_codes, axes=AXES):
    """A total order over the union: round-robin, each axis's next-best in turn.

    The union of four top-N sets has no natural total order, but three readers
    sort the community on ``mean_rank``, so one has to be synthesized. Round-robin
    keeps every axis represented at every depth, so any downstream truncation
    stays balanced across axes instead of exhausting one axis first.
    """
    names = [a["name"] for a in axes]
    queues = {nm: list(axis_codes.get(nm, [])) for nm in names}
    out, seen = [], set()
    while any(queues[nm] for nm in names):
        for nm in names:
            while queues[nm] and queues[nm][0] in seen:
                queues[nm].pop(0)
            if queues[nm]:
                code = queues[nm].pop(0)
                seen.add(code)
                out.append(code)
    return out


def select_union(pool, n_per_axis, p, axes=AXES):
    """Union of the per-axis top-N selections, with the audit trail.

    ``pool`` is a dict of equal-length sequences: ``species_code``, ``migration``,
    and one entry per axis ``column``. Returns the ordered ``codes``, the
    ``category`` map (which axes chose each species -- the union's selection
    basis, which a mean rank destroys), per-axis diagnostics, and the realized
    composition.
    """
    codes = [str(c) for c in pool["species_code"]]
    migration = np.asarray(pool["migration"])
    per_axis, axis_codes = {}, {}
    for ax in axes:
        if ax["column"] not in pool:
            raise KeyError(f"axis {ax['name']!r} needs pool column {ax['column']!r}; "
                           f"pool has {sorted(pool)}")
        sel = select_axis(codes, pool[ax["column"]], migration,
                          ax["ascending"], n_per_axis, p)
        per_axis[ax["name"]] = sel
        axis_codes[ax["name"]] = sel["codes"]

    ordered = interleaved_order(axis_codes, axes)
    category = {code: [nm for nm in axis_codes if code in set(axis_codes[nm])]
                for code in ordered}
    by_code = {c: i for i, c in enumerate(codes)}
    comp = composition([migration[by_code[c]] for c in ordered])
    return {
        "codes": ordered,
        "category": category,
        "axes": per_axis,
        "composition": comp,
        "penalty": float(p),
        "n_per_axis": int(n_per_axis),
    }


def solve_penalty(pool, n_per_axis, max_migratory_frac, axes=AXES,
                  p_max=120, step=1):
    """Smallest integer penalty whose union meets ``max_migratory_frac``.

    Configuring the target rather than the penalty is what keeps the rule stable:
    pool-rank units are arbitrary, so a fixed p drifts in meaning whenever the
    gate or the pool changes, while a stated composition does not.

    Scans upward rather than bisecting. The composition is monotone in p in
    practice but not guaranteed to be -- the union is a set union of four
    independently-cut axes -- and a scan returns the true smallest qualifying p
    under any shape, at trivial cost (a few hundred rows x ~100 candidates).

    Returns ``(selection, trace)``. If no p in range qualifies, the selection is
    the one at ``p_max`` and ``selection['target_met']`` is False -- refusing to
    silently pass off a community that misses the target.
    """
    trace, best = [], None
    for p in range(0, int(p_max) + 1, int(step)):
        sel = select_union(pool, n_per_axis, p, axes=axes)
        frac = sel["composition"]["fracs"][MIGRATORY]
        trace.append({"p": p, "n": sel["composition"]["n"],
                      "migratory_frac": frac,
                      "fracs": dict(sel["composition"]["fracs"])})
        best = sel
        if frac <= max_migratory_frac:
            sel["target_met"] = True
            sel["max_migratory_frac"] = float(max_migratory_frac)
            return sel, trace
    best["target_met"] = False
    best["max_migratory_frac"] = float(max_migratory_frac)
    return best, trace


def format_selection(sel):
    """Human-readable summary: composition, solved penalty, per-axis bars."""
    comp = sel["composition"]
    lines = [
        f"[axes] {comp['n']} species from {len(sel['axes'])} axes x "
        f"{sel['n_per_axis']} at penalty p={sel['penalty']:g}"
        + ("" if sel.get("target_met", True) else "  TARGET NOT MET"),
        "[axes] composition: " + ", ".join(
            f"{MIGRATION_LABELS[c]} {comp['counts'][c]} ({comp['fracs'][c]:.0%})"
            for c in (SEDENTARY, PARTIAL, MIGRATORY)),
    ]
    if "max_migratory_frac" in sel:
        lines.append(f"[axes] target migratory <= {sel['max_migratory_frac']:.0%}, "
                     f"achieved {comp['fracs'][MIGRATORY]:.0%}")
    lines.append("[axes] admission bar in RAW POOL RANK (a class places iff its raw "
                 "rank is at or below its bar):")
    lines.append(f"[axes]   {'axis':<15}{'thresh':>8}{'sed':>8}{'part':>8}{'migr':>8}")
    for name, a in sel["axes"].items():
        b = a["bars"]
        lines.append(f"[axes]   {name:<15}{a['threshold_key']:>8.0f}"
                     f"{b[SEDENTARY]:>8.0f}{b[PARTIAL]:>8.0f}{b[MIGRATORY]:>8.0f}")
    return "\n".join(lines)
