"""Generate E086v: can PRE-1940-QUALITY CBC winters carry the sparse-data gain? T0 1996 s1-4, three arms (after the
e086v-data job): ruzd = Ruzicka term whose 1966-95 winters are degraded; presd = ruzx + presence term (completeness
slopes) on DEGRADED 1966-95 lists; presc = ruzx + the same presence term on CLEAN 1966-95 lists. Scored late as E086s/u."""
import json
import os

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OV, SP = os.path.join(R, "research", "overlays"), os.path.join(R, "research", "specs")
PY = "{python}"
DD = "/mnt/e/Datasets/cbc_raw/pseudo_early3_w6695/prepared"
ENV = "export BLR_DEVICE=cuda BLR_MAX_ITER=100 BLR_STARTS=0.5 SCORE_TARGETS=dev"
REG = (("full", ""), ("row03t1", " --thin 0.03 --thin-unit row --thin-seed 1"), ("row03t2", " --thin 0.03 --thin-unit row --thin-seed 2"))
PRES = lambda path, label: {"path": path, "weight": 0.01, "label": label, "include_withheld": True, "completeness_slopes": True}
jobs, models = [], []
for s in (1, 2, 3, 4):
    for kind in ("ruzd", "presd", "presc"):
        src = f"desk_tempho_1995_seed{s}_{'ruz' if kind == 'ruzd' else 'ruzx'}.json"
        ov = json.loads(json.dumps(json.load(open(os.path.join(OV, src)))))
        if kind == "ruzd":
            ov["desk"]["aux_ruzicka"].update({"path": f"{DD}/cbc_ruzicka_res_deg.npz", "label": "cbc_ruzicka_deg"})
        else:
            ov["desk"]["aux_presence"] = PRES(f"{DD}/cbc_presence_res_comp_w6695_{'deg' if kind == 'presd' else 'clean'}.npz",
                                              f"cbc_presence_{kind}")
        ov["_comment"] = f"E086v: {kind} T0 1996 seed {s} (from {src})."
        ovp = f"research/overlays/desk_tempho_1995_seed{s}_{kind}.json"
        json.dump(ov, open(os.path.join(R, ovp), "w"), indent=1)
        name, cache = f"desk-tempho1995-seed{s}-{kind}", f"desk_tempho_1995_s{s}_{kind}"
        json.dump({"exp": "E086", "name": name, "group": "gpu", "overlay": ovp, "after_jobs": ["E086/e086v-data"],
                   "cmd": ["bash", "-c", "set -e; export RESEARCH_OUT={out} TMPDIR={out}/tmp MPLCONFIGDIR={out}/tmp/mpl "
                           "WANDB_MODE=offline WANDB_DIR={out}; mkdir -p {out}/desk {out}/tmp; " + PY + " scripts/run_encoder.py desk; "
                           + PY + f" research/exp/cache.py --out {{cache}}/{cache}; " + PY + " research/exp/epoch_variant.py --run {out} "
                           f"--overlay {ovp} --cache-out {{cache}}/{cache}_late --work {{out}}/late"],
                   "expected_minutes": 90,
                   "summarize": ["python", "-c", "import json; s=json.load(open('{out}/desk/run_summary.json')); print({k: s.get(k) "
                                 "for k in ('best_epoch','best_val_kernel','best_val_zmse','epochs_run')})"],
                   "note": f"E086v: {kind} T0 1995 seed {s}"}, open(os.path.join(SP, f"E086v_desk_t1995_s{s}_{kind}.json"), "w"), indent=1)
        jobs.append(f"E086/{name}"); models.append((f"{kind}_s{s}", f"{cache}_late"))
cmd = ["set -e", ENV] + [f"bash research/bin/fill_esk.sh {{cache}}/desk_tempho_1995_s1 {{cache}}/{c} || true" for _, c in models]
for reg, extra in REG:
    for n, c in models:
        cmd.append(f"[ -f {{out}}/{reg}/{n}/species_components_p96dev.npz ] || bash research/bin/score_both.sh {PY} {{out}}/{reg}/{n} "
                   f"--cache {{cache}}/{c} --rank 24{extra}")
json.dump({"exp": "E086", "name": "e086v-eval", "group": "gpu", "overlay": "research/overlays/desk_tempho_1995_seed1.json",
           "after_jobs": jobs, "cmd": ["bash", "-c", "; ".join(cmd)], "expected_minutes": 320,
           "note": "E086v eval: late caches, fair_rivals dev r24 full + row 3%; ruz / ruzx / control from E086n / E086s / E086u"},
          open(os.path.join(SP, "E086v_eval.json"), "w"), indent=1)
print("ok", len(jobs))
