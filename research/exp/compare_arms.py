"""One table across fast-arm runs: change skill (with CI), captured share and attenuation per change set,
for all resolvable species and the common tiers, total and place-specific.

    python research/exp/compare_arms.py <dir containing one fast-arm output per subdirectory> [--tiers 0.1-0.3 0.3-1.01]
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--tiers", nargs="+", default=("0.1-0.3", "0.3-1.01"))
    a = ap.parse_args()
    runs = []
    for d in sorted(os.listdir(a.root)):
        p = os.path.join(a.root, d, "results.json")
        if os.path.exists(p):
            runs.append((d, json.load(open(p, encoding="utf-8"))))
    f = lambda v: "   -  " if v is None else f"{v:+.3f}"

    def cell(o):
        if not o:
            return f"{'':27s}"
        sk = o["skill_vs_no_change"]
        ci = sk.get("median_ci") or [None, None]
        return f"{f(sk.get('median'))} [{f(ci[0])},{f(ci[1])}] {f(o['captured_pooled'])}"
    sets = sorted({s for _, r in runs for s, o in r["change"].items() if "note" not in o})
    for s in sets:
        print(f"change set {s}: skill vs no_change [95% CI] captured  (all resolvable | "
              + " | ".join(f"prev {t}" for t in a.tiers) + ")")
        for kind in ("total", "place"):
            print(f"  {kind}")
            for name, r in runs:
                o = r["change"].get(s)
                if not o or "note" in o:
                    continue
                blk = o if kind == "total" else o.get("place")
                if not blk:
                    continue
                row = [cell(blk)] + [cell((blk.get("tiers") or {}).get(t)) for t in a.tiers]
                desc = (f"{r['kernel']}/{r['features']}/r{r['rank']}"
                        + (f"/lag{r['lag_hl']:g}" if r.get("lag_hl") else "")
                        + (f"/ref{r['ref_window'][0]}" if r["kernel"] == "split" else ""))
                print(f"    {name:4s} {desc:24s} " + " | ".join(row))


if __name__ == "__main__":
    main()
