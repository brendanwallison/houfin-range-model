"""CPU-only adult BBL reconnaissance. No production model or GPU dependencies.

prepare builds monthly release/recovery m-arrays including never-recovered birds.
fit estimates constant regional adult survival and era-specific conditional dead
reporting. Release occurs at month midpoint (explicit approximation). This is a
diagnostic, not an annual population-model likelihood or a juvenile-survival fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, xlogy
from scipy.signal import lfilter

REGIONS = ("west", "plains", "central_east", "east")


def marked_transition(departure, resident_survival, movement, journey_survival):
    """Origin x destination subprobability matrix; Q REPLACES Sj for movers.

    movement is the land/edge-corrected pre-survival transition. Rows sum to at
    most one, preserving any finite-epsilon loss in the simulator. For cohort-specific Q, sum this expression over weighted cohorts
    before adding the stayer diagonal. This reference does not construct FFT rows.
    """
    p, sj, move, q = map(np.asarray, (departure, resident_survival, movement, journey_survival))
    n = len(p)
    if sj.shape != (n,) or move.shape != (n, n) or q.shape not in ((n,), (n, n)):
        raise ValueError("incompatible origin/destination dimensions")
    if any(np.any(~np.isfinite(v)) or np.any((v < 0) | (v > 1)) for v in (p, sj, move, q)):
        raise ValueError("probabilities must be finite and in [0,1]")
    if np.any(move.sum(1) > 1+1e-10):
        raise ValueError("movement must include land conditioning and sum to at most one per origin")
    return np.diag((1-p)*sj) + p[:, None]*move*q


def marked_recovery_distribution(initial, transitions, reporting):
    """Propagate a marked cohort without births, retaining location uncertainty.

    transitions[t,origin,destination] includes movement AND survival; reporting[t]
    is conditional on death in that interval, constant across locations for this
    cohort. The last outcome means never reported dead before administrative
    cutoff, including unreported deaths. Partial intervals require explicit
    transitions supplied by the caller; no additional Sj or Q factor is applied.
    This NumPy reference is not wired into NumPyro.
    """
    alive=np.asarray(initial,dtype=float).copy()
    ts=np.asarray(transitions,dtype=float)
    r=np.asarray(reporting,dtype=float)
    n=len(alive)
    if ts.ndim!=3 or ts.shape[1:]!=(n,n) or r.shape!=(len(ts),):
        raise ValueError("incompatible marked-cohort arrays")
    if (np.any(~np.isfinite(alive)) or np.any(alive<0)
            or not np.isclose(alive.sum(),1,atol=1e-12,rtol=0)
            or np.any(~np.isfinite(ts)) or np.any(ts<0)
            or np.any(ts.sum(2)>1+1e-12)
            or np.any(~np.isfinite(r)) or np.any((r<0)|(r>1))):
        raise ValueError("invalid initial distribution or transition/reporting probabilities")
    recovered=[]
    for t,report in zip(ts,r):
        following=alive@t
        recovered.append(max(0.,float(alive.sum()-following.sum()))*report)
        alive=following
    return np.r_[recovered,1-sum(recovered)]


def recovery_probabilities(survival, reporting, n_months, initial_fraction=.5):
    """Rows: release month. Columns: recovery month, then never recovered.

    survival is one annual probability; reporting is conditional on death in
    each month. The first interval is half a month; later intervals are full.
    S can later be replaced by propagated marked-bird distributions, not S(E[x]).
    """
    if not 0 < survival < 1:
        raise ValueError("annual survival must be strictly between zero and one")
    if not 0 < initial_fraction <= 1:
        raise ValueError("initial month exposure must be in (0,1]")
    r = np.asarray(reporting, dtype=float)
    if r.shape != (n_months,) or np.any(~np.isfinite(r)) or np.any((r < 0) | (r > 1)):
        raise ValueError("invalid conditional reporting probabilities")
    lag = np.arange(n_months)[None, :] - np.arange(n_months)[:, None]
    exposure_before = np.where(lag>0,lag-1+initial_fraction,0) / 12
    duration = np.where(lag == 0, initial_fraction, 1.) / 12
    log_s = np.log(survival)
    recovered = np.where(lag >= 0, np.exp(log_s*exposure_before)*(-np.expm1(log_s*duration))*r, 0.)
    return np.column_stack((recovered, 1-recovered.sum(1)))


def likelihood_counts(counts):
    """Sparse recovery events plus the full no-recovery denominator."""
    row, col = np.nonzero(counts[:, :-1])
    if np.any(col < row):
        raise ValueError("recovery precedes release")
    return row, col, counts[row,col], counts[:,-1]


def negative_loglik(theta, counts, era, initial_fraction=.5):
    """Multinomial kernel; constants omitted. No informative survival prior."""
    s = expit(theta[0])
    r = expit(np.asarray(theta[1:]))[era]
    row,col,observed,unobserved = likelihood_counts(counts) if isinstance(counts,np.ndarray) else counts
    sm, sh = s**(1/12), s**(initial_fraction/12)
    # Backward recursion: eventual reported death before administrative cutoff.
    future = lfilter([1-sm], [1,-sm], r[::-1])[::-1]
    recovered = (1-sh)*r + sh*np.r_[future[1:],0.]
    lag = col-row
    p = s**(np.where(lag>0,lag-1+initial_fraction,0)/12) * (1-s**(np.where(lag==0,initial_fraction,1)/12)) * r[col]
    return float(-xlogy(observed,p).sum()-xlogy(unobserved,1-recovered).sum())


def prepare(source, output, cutoff="2024-12-31", start_year=1960):
    import pandas as pd

    source, output = Path(source), Path(output)
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    df = df[df.species_id.eq("5190")].copy()
    # Codes from age.csv: 1 AHY, 5 SY, 6 ASY, 7 TY, 8 ATY; 4 is LOCAL,
    # not adult. Status 3 normal wild;
    # condition 03-05 dead. Band-only and banding-mortality reports are excluded.
    for col in ("age_code", "bird_status", "present_condition", "how_obtained_code"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    dates = pd.to_datetime(df.event_date, format="%m/%d/%Y", errors="coerce")
    # Require valid year/month but permit unknown day; analysis is month resolved.
    year = pd.to_numeric(df.event_year, errors="coerce")
    month = pd.to_numeric(df.event_month, errors="coerce")
    ym = pd.to_datetime(dict(year=year, month=month, day=1), errors="coerce")
    end = pd.Timestamp(cutoff)
    if end != end + pd.offsets.MonthEnd(0):
        raise ValueError("cutoff must be a month end")
    df["month_index"] = (year-start_year)*12 + month-1
    df["date"] = dates
    df["ym"] = ym
    valid_time = ym.notna() & (ym.dt.year >= start_year) & (ym <= end)
    # original_band is the chain key. Never silently collapse ambiguous releases.
    df["key"] = df.original_band.where(df.original_band.ne(""), df.band)
    bands = df[df.event_type.eq("B") & valid_time].copy()
    duplicate_keys = bands.key[bands.key.duplicated(keep=False)].unique()
    bands = bands[~bands.key.isin(duplicate_keys) & bands.key.ne("")]
    lon = pd.to_numeric(bands.lon_dd, errors="coerce")
    lat = pd.to_numeric(bands.lat_dd, errors="coerce")
    bands["region"] = np.select([lon < -100, lon < -95, lon < -80], [0, 1, 2], default=3)
    keep = (bands.age_code.isin([1,5,6,7,8]) & bands.bird_status.eq(3)
            & lon.between(-180, -50) & lat.between(10, 85)
            & ((lon < -100) | bands.ym.dt.month.between(4,8)))
    bands = bands[keep].copy()
    handling_keys = df.loc[df.event_type.isin(["E", "R"])
                           & df.how_obtained_code.isin([10,51]), "key"]
    handling_excluded = int(bands.key.isin(handling_keys).sum())
    bands = bands[~bands.key.isin(handling_keys)]
    dead = df[(df.event_type.isin(["E", "R"])) & valid_time
              & df.present_condition.isin([3,4,5])
              & ~df.how_obtained_code.isin([10,51,96,98])].copy()
    dead = dead.sort_values(["ym", "event_date"]).drop_duplicates("key")
    joined = bands.merge(dead[["key", "month_index", "date"]], on="key", how="left", suffixes=("", "_dead"))
    # Ambiguous same-day/death-before-release records cannot be ordinary recoveries.
    bad = joined.month_index_dead.lt(joined.month_index) | (
        joined.date.notna() & joined.date_dead.notna() & joined.date_dead.le(joined.date))
    excluded_bad = int(bad.sum())
    joined = joined[~bad]
    n_months = (end.year-start_year)*12+end.month
    counts = np.zeros((4, n_months, n_months+1), dtype=np.int64)
    release = joined.month_index.to_numpy(dtype=int)
    recovery = joined.month_index_dead.fillna(n_months).to_numpy(dtype=int)
    region = joined.region.to_numpy(dtype=int)
    np.add.at(counts, (region, release, recovery), 1)
    years = start_year + np.arange(n_months)//12
    era = np.searchsorted([1980,1995,2006], years, side="right")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"source": str(source), "sha256": digest, "cutoff": cutoff,
                "start_year": start_year, "regions": REGIONS,
                "adult_bands": counts.sum((1,2)).tolist(),
                "dead_recoveries": counts[:,:,:-1].sum((1,2)).tolist(),
                "ambiguous_release_keys_excluded": len(duplicate_keys),
                "bad_chronology_birds_excluded": excluded_bad,
                "handling_mortality_birds_excluded": handling_excluded,
                "assumptions": ["month-midpoint release", "death assigned to encounter month",
                  "constant survival within origin region", "reporting varies by death era",
                  "breeding-season bandings outside West", "no routine live-recapture likelihood"],
                "status": "adult diagnostic only; not population-model likelihood"}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"adult_marray.npz", counts=counts, era=era)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def fit(input_dir, initial_fraction=.5, constant_reporting=False):
    root = Path(input_dir)
    data = np.load(root/"adult_marray.npz", allow_pickle=False)
    era = np.zeros_like(data["era"]) if constant_reporting else data["era"]
    n_eras = int(era.max())+1
    results = {}
    for name, counts in zip(REGIONS, data["counts"]):
        if counts[:,:-1].sum() == 0:
            results[name] = {"status":"no dead recoveries; survival not estimated"}
            continue
        sparse = likelihood_counts(counts)
        starts = [np.r_[a, np.full(n_eras,-5.)] for a in [-1., 0., 1.]]
        fits = [minimize(negative_loglik, x, args=(sparse, era, initial_fraction), method="L-BFGS-B",
                         bounds=[(-5,5)]+[(-12,0)]*n_eras) for x in starts]
        best = min(fits, key=lambda f:f.fun)
        # Optimize reporting at each fixed survival: uncertainty in S, with reporting refit.
        profile = []
        for s in np.linspace(.2,.85,27):
            logit = np.log(s/(1-s))
            f = minimize(lambda x:negative_loglik(np.r_[logit,x],sparse,era,initial_fraction),
                         best.x[1:], method="L-BFGS-B", bounds=[(-12,0)]*n_eras)
            profile.append({"survival": float(s), "delta_nll": float(f.fun-best.fun),
                            "success": bool(f.success)})
        results[name] = {"annual_survival": float(expit(best.x[0])),
                         "conditional_reporting_by_era": expit(best.x[1:]).tolist(),
                         "nll": float(best.fun), "success": bool(best.success),
                         "start_nll_spread": float(np.ptp([f.fun for f in fits])),
                         "initial_fraction": initial_fraction,
                         "constant_reporting": constant_reporting,
                         "profile": profile}
    suffix = "" if initial_fraction==.5 and not constant_reporting else f"_fraction{initial_fraction:g}_constant{int(constant_reporting)}"
    (root/f"adult_fit{suffix}.json").write_text(json.dumps(results, indent=2)+"\n")
    return results


def validate_sampling(input_dir):
    """Two planted adult truths under the actual release-cohort sample sizes."""
    root=Path(input_dir)
    data=np.load(root/"adult_marray.npz",allow_pickle=False)
    real=json.loads((root/"adult_fit.json").read_text())
    output={}
    for seed in [41,42]:
        rng=np.random.default_rng(seed)
        counts=np.zeros_like(data["counts"])
        for g,name in enumerate(REGIONS):
            reporting=np.array(real[name]["conditional_reporting_by_era"])
            p=recovery_probabilities(.55,reporting[data["era"]],len(data["era"]))
            counts[g]=np.array([rng.multinomial(int(n),row)
                               for n,row in zip(data["counts"][g].sum(1),p)])
        target=root/f"synthetic_seed{seed}"; target.mkdir(exist_ok=True)
        np.savez_compressed(target/"adult_marray.npz",counts=counts,era=data["era"])
        fitted=fit(target)
        for name,v in fitted.items():
            if "profile" in v:
                truth_point=min(v["profile"],key=lambda x:abs(x["survival"]-.55))
                v["truth_in_approx95_profile"]=truth_point["delta_nll"] <= 1.920729
                v["truth_delta_nll"]=truth_point["delta_nll"]
        output[str(seed)]=fitted
    (root/"sampling_guards.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "fit", "validate"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", default="2024-12-31")
    parser.add_argument("--initial-fraction", type=float, choices=[.25,.5,.75], default=.5)
    parser.add_argument("--constant-reporting", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        print(json.dumps(prepare(args.source,args.out,args.cutoff), indent=2))
    elif args.mode=="fit":
        print(json.dumps({k:{a:b for a,b in v.items() if a != "profile"}
                          for k,v in fit(args.out,args.initial_fraction,args.constant_reporting).items()}, indent=2))
    else:
        print(json.dumps({s:{k:{a:b for a,b in v.items() if a != "profile"}
                             for k,v in d.items()} for s,d in validate_sampling(args.out).items()},indent=2))
