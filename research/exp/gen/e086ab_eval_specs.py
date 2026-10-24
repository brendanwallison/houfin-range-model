"""Generate the E086ab evaluation: derived sum caches for the sw arms, then fair_rivals dev (score_both) on every fold
cache and the 96-species control, full + row 3% (thinning seeds 1, 2)."""
import json
import os

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = "{python}"
ENV = "export BLR_DEVICE=cuda BLR_MAX_ITER=100 BLR_STARTS=0.5 SCORE_TARGETS=dev"
REG = (("full", ""), ("row03t1", " --thin 0.03 --thin-unit row --thin-seed 1"), ("row03t2", " --thin 0.03 --thin-unit row --thin-seed 2"))
cmd = ["set -e", ENV]
runs = []   # (name, cache, extra)
for f in range(3):
    for st in ("_late", ""):
        tag = "late" if st else "sel"
        s, sw = f"desk_tempho_1995_s1_ab_f{f}_s{st}", f"desk_tempho_1995_s1_ab_f{f}_sw{st}"
        cmd.append(f"[ -f {{cache}}/{sw}_sum/meta.json ] || {PY} research/exp/e086ab_sum_cache.py --cache {{cache}}/{sw} "
                   f"--out {{cache}}/{sw}_sum --summer 24")
        for c in (s, sw, sw + "_sum"):
            cmd.append(f"bash research/bin/fill_esk.sh {{cache}}/desk_tempho_1995_s1 {{cache}}/{c} || true")
        runs += [(f"f{f}_{tag}_s_r24", s, " --rank 24"), (f"f{f}_{tag}_sum", sw + "_sum", " --rank 48 --desk-split 24")]
        if st:
            runs += [(f"f{f}_{tag}_s_r48", s, " --rank 48"), (f"f{f}_{tag}_sw_summer_r24", sw, " --rank 24")]
runs += [("ctl_late_r24", "desk_tempho_1995_s1_late", " --rank 24"), ("ctl_sel_r24", "desk_tempho_1995_s1", " --rank 24"),
         ("ctl_late_r48", "desk_tempho_1995_s1_late", " --rank 48")]
for reg, extra in REG:
    for n, c, x in runs:
        cmd.append(f"[ -f {{out}}/{reg}/{n}/species_components_p96dev.npz ] || bash research/bin/score_both.sh {PY} "
                   f"{{out}}/{reg}/{n} --cache {{cache}}/{c}{x}{extra}")
json.dump({"exp": "E086", "name": "e086ab-eval", "group": "gpu", "overlay": "research/overlays/desk_tempho_1995_seed1.json",
           "cmd": ["bash", "-c", "; ".join(cmd)], "expected_minutes": 480,
           "note": "E086ab eval: fold caches (s, sw sum, sw summer-only), 96-species control; fair_rivals dev full + row 3%"},
          open(os.path.join(R, "research", "specs", "E086ab_eval.json"), "w"), indent=1)
print("ok", len(runs) * 3, "score runs")
