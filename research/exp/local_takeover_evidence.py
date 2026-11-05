"""Publish compact aggregate takeover evidence; leave data and fits external."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--r104", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_root
    frozen = read(root / "replay-frozen-gpu-fixedhash/replay.json")
    reconstructed = read(root / "replay-reconstructed-gpu/replay.json")
    trace = read(root / "trace-audit/trace_audit.json")
    gradient = read(root / "gradient-audit/gradient_audit.json")
    r104 = read(args.r104)
    result = {"scope": "aggregate reproducibility, trace accounting, likelihood derivatives; no new fit",
              "replays": {name: {key: value for key, value in replay.items() if key != "sources"}
                          for name, replay in (("frozen", frozen), ("reconstructed", reconstructed))},
              "trace": {key: value for key, value in trace.items()
                        if key not in ("resolved_config", "input_shapes")},
              "gradient": gradient,
              "r104": {key: r104.get(key) for key in ("start_neg_logp", "end_neg_logp", "converged", "message",
                                                      "n_eval", "bbs_nll_end", "n50_raw_end", "never_reached_end")}}
    result["r104"].update(
        source_sha256=hashlib.sha256(args.r104.read_bytes()).hexdigest(),
        occupied_east_1966_median=r104["lambda_split_end"]["1966-2010"]["occupied_east"]["q10_50_90_y0"][1],
        occupied_east_2010_median=r104["lambda_split_end"]["1966-2010"]["occupied_east"]["q10_50_90_2010"][1],
        modern_east_mortality=r104["disease"]["by_region"]["east"]["2011-2025"]["realised_mortality"],
        last_evaluation=json.loads(args.r104.with_name(args.r104.name.replace("_summary.json", ".jsonl")).read_text().splitlines()[-1]))
    if not all(all(replay["checks"].values()) for replay in (frozen, reconstructed)):
        raise ValueError("Cannot publish a passed baseline with failed replay checks")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "r104_objective": result["r104"]["end_neg_logp"],
                      "data_inactive_sites": [row["site"] for row in gradient["sites"] if row["max_abs"] == 0]}))


if __name__ == "__main__":
    main()
