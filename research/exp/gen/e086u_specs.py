"""Generate E086u: (u1) selected-state fair_rivals scoring of ruz / ruzt / control s1-4; (u2) plain Ruzicka with the
withheld-era winters EXCLUDED (ruzx, include_withheld false) at T0 1996 s1-4, trained + scored late like E086s."""
import json
import os

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OV, SP = os.path.join(R, "research", "overlays"), os.path.join(R, "research", "specs")
PY = "{python}"
ENV = "export BLR_DEVICE=cuda BLR_MAX_ITER=100 BLR_STARTS=0.5 SCORE_TARGETS=dev"
REG = (("full", ""), ("row03t1", " --thin 0.03 --thin-unit row --thin-seed 1"), ("row03t2", " --thin 0.03 --thin-unit row --thin-seed 2"))
def score_cmds(models):
    cmd = ["set -e", ENV]
    for name, cache in models:
        cmd.append(f"bash research/bin/fill_esk.sh {{cache}}/desk_tempho_1995_s1 {{cache}}/{cache} || true")
    for reg, extra in REG:
        for name, cache in models:
            cmd.append(f"[ -f {{out}}/{reg}/{name}/species_components_p96dev.npz ] || bash research/bin/score_both.sh {PY} "
                       f"{{out}}/{reg}/{name} --cache {{cache}}/{cache} --rank 24{extra}")
    return ["bash", "-c", "; ".join(cmd)]
# u1: selected state
sel = [(f"{k}_s{s}", f"desk_tempho_1995_s{s}" + ("" if k == "ctl" else f"_{k}")) for k in ("ruz", "ctl", "ruzt") for s in (1, 2, 3, 4)]
json.dump({"exp": "E086", "name": "e086u-selected", "group": "gpu", "overlay": "research/overlays/desk_tempho_1995_seed1.json",
           "cmd": score_cmds(sel), "expected_minutes": 480,
           "note": "E086u1: CBC Ruzicka family at the SELECTED state, fair_rivals dev r24 full + row 3%"},
          open(os.path.join(SP, "E086u_selected.json"), "w"), indent=1)
# u2: withheld winters excluded
jobs = []
for s in (1, 2, 3, 4):
    base = json.load(open(os.path.join(OV, f"desk_tempho_1995_seed{s}_ruz.json" if s > 2 or True else "")))
    ov = json.loads(json.dumps(base))
    ov["desk"]["aux_ruzicka"]["include_withheld"] = False
    ov["desk"]["aux_ruzicka"]["label"] = "cbc_ruzicka_x"
    ov["_comment"] = f"E086u2: desk_tempho_1995_seed{s}_ruz with the withheld-era (1966-1995) CBC winters EXCLUDED (ruzx)."
    ovp = f"research/overlays/desk_tempho_1995_seed{s}_ruzx.json"
    json.dump(ov, open(os.path.join(R, ovp), "w"), indent=1)
    name, cache = f"desk-tempho1995-seed{s}-ruzx", f"desk_tempho_1995_s{s}_ruzx"
    json.dump({"exp": "E086", "name": name, "group": "gpu", "overlay": ovp,
               "cmd": ["bash", "-c", "set -e; export RESEARCH_OUT={out} TMPDIR={out}/tmp MPLCONFIGDIR={out}/tmp/mpl "
                       "WANDB_MODE=offline WANDB_DIR={out}; mkdir -p {out}/desk {out}/tmp; " + PY + " scripts/run_encoder.py desk; "
                       + PY + f" research/exp/cache.py --out {{cache}}/{cache}; " + PY + " research/exp/epoch_variant.py --run {out} "
                       f"--overlay {ovp} --cache-out {{cache}}/{cache}_late --work {{out}}/late"],
               "expected_minutes": 90,
               "summarize": ["python", "-c", "import json; s=json.load(open('{out}/desk/run_summary.json')); print({k: s.get(k) "
                             "for k in ('best_epoch','best_val_kernel','best_val_zmse','epochs_run')})"],
               "note": f"E086u2: ruzx T0 1995 seed {s}"}, open(os.path.join(SP, f"E086u_desk_t1995_s{s}_ruzx.json"), "w"), indent=1)
    jobs.append(f"E086/{name}")
late = [(f"ruzx_s{s}", f"desk_tempho_1995_s{s}_ruzx_late") for s in (1, 2, 3, 4)]
json.dump({"exp": "E086", "name": "e086u-ruzx-eval", "group": "gpu", "overlay": "research/overlays/desk_tempho_1995_seed1.json",
           "after_jobs": jobs, "cmd": score_cmds(late), "expected_minutes": 180,
           "note": "E086u2 eval: ruzx late, fair_rivals dev r24 full + row 3% (ruz / control from E086n / E086s)"},
          open(os.path.join(SP, "E086u_ruzx_eval.json"), "w"), indent=1)
print("ok")
