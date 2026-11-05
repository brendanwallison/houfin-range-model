"""Matched structural-fit readouts with mortality at its actual forward-stage density.

The diagnostic scan adds outputs only. Its population arrays must reproduce the
unmodified scan before its extra fields can be used. No fitted mechanism changes.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

from harness import Sandbox, capture, to_land, potential_lambda
from src.model import age_forward as af


def diagnostic_forward():
    source = textwrap.dedent(inspect.getsource(af.forward_sim_age_structured))
    changes = [
        ("return_adult_arrivals=settle_dispersers_only or occ,", "return_adult_arrivals=True,"),
        ("if settle_dispersers_only or occ:", "if True:"),
        ("            Sa_g = Sa_g * (1.0 - m_g)",
         "            diagnostic = jnp.stack([Nloc, m_g, N_a, N_j, N_a_post, juvenile_stayers, juvenile_arriving, adult_arriving], axis=-1)[land_rows, land_cols]\n"
         "            Sa_g = Sa_g * (1.0 - m_g)"),
        ("(N_total_new, N_a_new, N_j_new, psi, F_nest)",
         "(N_total_new, N_a_new, N_j_new, psi, F_nest, diagnostic)"),
        ("extra = (_outs[4],) if nest_cells is not None else ()",
         "extra = (_outs[4], _outs[5]) if nest_cells is not None else ()"),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError(f"Diagnostic source anchor must match exactly once: {old!r}")
        source = source.replace(old, new)
    namespace = dict(af.__dict__)
    exec(compile(source, "<diagnostic_forward>", "exec"), namespace)
    return namespace["forward_sim_age_structured"]


def run_diagnostic(x):
    if x.get("disease_m_amp") is None or x.get("occupancy_b") is not None or x.get("strata_u") is not None:
        raise ValueError("Registered readout expects survival disease, no occupancy or strata")
    fn = diagnostic_forward()
    args = dict(x, nest_cells=(x["land_rows"], x["land_cols"]))
    out = jax.tree.map(lambda a: np.asarray(a), fn(**args))
    original = af.forward_sim_age_structured(**x)
    error = max(float(np.max(np.abs(out[i] - np.asarray(original[i])))) for i in range(3))
    if not all(np.allclose(out[i], np.asarray(original[i]), rtol=1e-10, atol=1e-12) for i in range(3)):
        raise ValueError(f"Diagnostic scan changed population states: max error {error}")
    return out, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-controls", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sb = Sandbox(os.environ["SBX_REFINED"])
    with np.load(args.params) as saved:
        sb.lat.update({k: saved[k] for k in saved.files})
    x, _ = capture(sb.data, sb.lat)
    out, error = run_diagnostic(x)
    control = None
    if args.validate_controls:
        null, null_error = run_diagnostic(dict(x, disease_m_amp=jnp.zeros_like(x["disease_m_amp"])))
        mortality = null[-1][..., 1]
        if not np.array_equal(mortality, np.zeros_like(mortality)):
            raise ValueError("Planted disease-off control did not recover exactly zero mortality")
        control = {"disease_off_mortality_max": float(np.abs(mortality).max()),
                   "disease_off_population_max_error": null_error}
    total, na, nj, factual, flows = out
    nloc, m, na_start, nj_start, na_post, j_stay, j_arrive, a_arrive = np.moveaxis(flows, -1, 0)
    t, nx = len(sb.years), sb.Nx
    land = lambda v: to_land(v, sb.rows, sb.cols, t, nx)
    sa, sj, fmax, k = [land(x[name]) for name in ("Sa_flat", "Sj_flat", "Fmax_flat", "K_flat")]
    density = land(total)
    amp = land(x["disease_m_amp"])
    hill_n, half = float(x["disease_hill_n"]), float(x["disease_k_half"])
    hn_stage = (np.maximum(nloc, 0.) + 1e-12) ** hill_n
    reconstructed_mortality = amp * hn_stage / (hn_stage + half ** hill_n)
    mortality_error = float(np.max(np.abs(reconstructed_mortality - m)))
    if not np.allclose(reconstructed_mortality, m, rtol=1e-10, atol=1e-12):
        raise ValueError("Independent NumPy mortality reconstruction disagrees with scan readout")
    hn_end = np.maximum(density, 0.) ** hill_n
    end_density_approximation = amp * hn_end / (hn_end + half ** hill_n)
    lam = potential_lambda(sa, sj, fmax, bool(x.get("yearling_breeding")))
    projection = potential_lambda(sa*(1-m), sj*(1-m), factual, bool(x.get("yearling_breeding")))
    occ = np.isfinite(sb.first_obs)
    summaries = {}
    for region, mask in sb.region.items():
        mask = mask & occ
        values = {}
        for year in (1966, 1990, 1993, 2010, 2020):
            ti = int(np.flatnonzero(sb.years == year)[0])
            q = lambda a: np.quantile(a[ti, mask], [.1, .5, .9]).tolist()
            values[str(year)] = {"potential_lambda_q10_50_90": q(lam),
                                 "resident_stage_projection_q10_50_90": q(projection),
                                 "Sa_q10_50_90": q(sa), "Sj_q10_50_90": q(sj),
                                 "Fmax_q10_50_90": q(fmax), "Factual_q10_50_90": q(factual),
                                 "K_route_count_q10_50_90": q(k*20), "disease_mortality_q10_50_90": q(m)}
        eras = {}
        for y0, y1 in ((1966, 1990), (1993, 2010), (2011, 2025)):
            support = np.ix_((sb.years >= y0) & (sb.years <= y1), mask)
            arr = (a_arrive+j_arrive)[support].sum()
            local = nloc[support].sum()
            eras[f"{y0}-{y1}"] = {"mortality_cell_year_mean": float(m[support].mean()),
                                  "end_density_mortality_approximation_mean": float(end_density_approximation[support].mean()),
                                  "arriving_share_of_pre_disease_pool": float(arr / max(local, 1e-30)),
                                  "potential_lambda_cell_year_mean": float(lam[support].mean()),
                                  "stage_projection_below_one_share": float((projection[support] < 1).mean())}
        summaries[region] = {"fixed_positive_cells": int(mask.sum()), "years": values, "eras": eras}
    a_depart = na_start - na_post + a_arrive
    j_depart = nj_start - j_stay
    movement = {"adult_departures_total": float(a_depart.sum()), "adult_arrivals_total": float(a_arrive.sum()),
                "juvenile_departures_total": float(j_depart.sum()), "successful_juvenile_arrivals_total": float(j_arrive.sum()),
                "scope": "all model years and land cells; model-density sums; juvenile arrivals already include Q"}
    replay = sb.run({"name": "fitted"}, keep_density=True)
    replay_density = replay.pop("_density")
    replay.pop("final_state_sha256", None)  # legacy mask-overwrite bug
    replay["final_land_density_sha256"] = hashlib.sha256(np.ascontiguousarray(replay_density[-1]).tobytes()).hexdigest()
    if not np.allclose(replay_density, density, rtol=1e-10, atol=1e-12):
        raise ValueError("Scoring replay disagrees with instrumented population states")
    result = {"params_sha256": hashlib.sha256(args.params.read_bytes()).hexdigest(),
              "forward_source_sha256": hashlib.sha256(Path(af.__file__).read_bytes()).hexdigest(),
              "readout_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "diagnostic_population_max_error": error, "regions": summaries,
              "mortality_numpy_reconstruction_max_error": mortality_error,
              "planted_control": control,
              "movement": movement, "scored_replay": replay,
              "interpretation": {"potential_lambda": "mate-unlimited, disease-free, crowding-free index; not the N-to-zero limit with Allee",
                                 "resident_stage_projection": "eigenvalue at fitted Factual and actual stage mortality, not feedback Jacobian or realized growth",
                                 "mortality": "fraction applied at actual post-dispersal pre-reproduction Nloc; not computed from end-of-year density",
                                 "relative_front_thresholds": "arm-specific modern density reference; use absolute thresholds for matched comparisons"}}
    (args.out / "readout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"state": "done", "out": str(args.out), "diagnostic_population_max_error": error,
                      "regions": {r: v["years"]["2010"]["potential_lambda_q10_50_90"][1] for r,v in summaries.items()}}), flush=True)


if __name__ == "__main__":
    main()
