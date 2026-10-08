"""Headline numbers for the model variants: fit, the K-switch metrics, refine and probe.

Reads only JSON (metrics.json, refine_report.json, probe.json), so it is safe on a login
node. Rows with no output yet print '-'.

    python scripts/diagnostics/headline_numbers.py
"""
import json
import os

R = os.path.join(os.environ.get("HOUFIN_PROCESSED", "data/processed"), "model_results")
RUNS = [  # label, metrics dir, refine-report dir, probe dir
    ("18 MAP", "age_map_float32_run_18_new_z_quick90", None, "probe__age_map_float32_run_18_new_z_quick90"),
    ("18 HMC best", "hmcdraw_best__age_map_float32_run_18_new_z_quick90__float64", None, None),
    ("18 refined", None, "refined_float64__age_map_float32_run_18_new_z_quick90", None),
    ("19 refined", "refined_float64__age_map_float32_run_19_new_z_quick90",
     "refined_float64__age_map_float32_run_19_new_z_quick90",
     "probe__refined_float64__age_map_float32_run_19_new_z_quick90"),
    ("19e refined", "refined_float64__age_map_float32_run_19e_new_z_quick90",
     "refined_float64__age_map_float32_run_19e_new_z_quick90",
     "probe__refined_float64__age_map_float32_run_19e_new_z_quick90"),
]
COLS = [("fit corr", "fit.log1p_correlation"), ("fit rmse", "fit.log1p_rmse"),
        ("K fold", "k_range.modern_fold_range"), ("K@floor", "k_range.fraction_near_floor"),
        ("Allee-dead", "realized_source_sink.allee_dead_fraction_of_suitable"),
        ("rho F-K", "manifold.corr_repro_capacity"), ("rho Sa-F", "manifold.corr_survival_repro"),
        ("mean lam", "modern_mean_lambda"), ("suitable", "modern_suitable_fraction"),
        ("realiz lam", "realized_source_sink.realized_modern_mean_lambda"),
        ("Z-unexpl", "z_attribution_since_invasion.residual_fraction_of_total"),
        ("K median", "k_range.modern_median_K"), ("dis. sev", "disease.disease_severity_median")]


def load(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(v):
    return "-" if v is None else (f"{v:.4g}" if isinstance(v, (int, float)) else str(v))


print(f"{'':12s}" + "".join(f"{c:>11s}" for c, _ in COLS))
for label, mdir, _, _ in RUNS:
    m = load(mdir, "map_diagnostics", "metrics.json") if mdir else None
    print(f"{label:12s}" + "".join(f"{fmt(get(m, p)):>11s}" for _, p in COLS))
print("\nL-BFGS refine (U is NOT comparable across model variants -- different prior terms):")
for label, _, rdir, _ in RUNS:
    r = load(rdir, "refine_report.json") if rdir else None
    if r:
        print(f"  {label:12s} U {r['U_start']:.1f} -> {r['U_end']:.1f}  ({r['U_end'] - r['U_start']:+.1f})  "
              f"{r['iterations']} its, |grad| {r['grad_norm_end']:.3g}; {r['reason']}")
print("\nprobe at the point (Hessian):")
for label, _, _, pdir in RUNS:
    p = load(pdir, "probe.json") if pdir else None
    if p:
        h = p.get("hessian", {})
        print(f"  {label:12s} d={p['d']}  grad {p['grad_seconds_median']:.3f}s  |grad| {p['grad_norm_at_map']:.3g}  "
              f"negative eigs {h.get('n_negative')}  min eig {fmt(h.get('eig_min'))}  "
              f"max eig {fmt(h.get('eig_max'))}")
