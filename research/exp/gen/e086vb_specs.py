"""Generate E086v-b: the presence arms of E086v with a WINTER observation operator (build_cbc_presence --readout-target
cbc, fitted on 1996-2024 lists) instead of the breeding readout. T0 1996 s1-4: presdw (degraded), prescw (clean)."""
import json
import os

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OV, SP = os.path.join(R, "research", "overlays"), os.path.join(R, "research", "specs")
PY = "{python}"
DD = "/mnt/e/Datasets/cbc_raw/pseudo_early3_w6695/prepared"
ENV = "export BLR_DEVICE=cuda BLR_MAX_ITER=100 BLR_STARTS=0.5 SCORE_TARGETS=dev"
REG = (("full", ""), ("row03t1", " --thin 0.03 --thin-unit row --thin-seed 1"), ("row03t2", " --thin 0.03 --thin-unit row --thin-seed 2"))
jobs, models = [], []
for s in (1, 2, 3, 4):
    for kind, f in (("presdw", "deg_wop"), ("prescw", "clean_wop")):
        src = f"desk_tempho_1995_seed{s}_ruzx.json"
        ov = json.loads(json.dumps(json.load(open(os.path.join(OV, src)))))
        ov["desk"]["aux_presence"] = {"path": f"{DD}/cbc_presence_res_comp_w6695_{f}.npz", "weight": 0.01,
                                      "label": f"cbc_presence_{kind}", "include_withheld": True, "completeness_slopes": True}
        ov["_comment"] = f"E086v-b: {kind} T0 1996 seed {s} (from {src}; winter observation operator)."
        ovp = f"research/overlays/desk_tempho_1995_seed{s}_{kind}.json"
        json.dump(ov, open(os.path.join(R, ovp), "w"), indent=1)
        name, cache = f"desk-tempho1995-seed{s}-{kind}", f"desk_tempho_1995_s{s}_{kind}"
        json.dump({"exp": "E086", "name": name, "group": "gpu", "overlay": ovp, "after_jobs": ["E086/e086vb-data"],
                   "cmd": ["bash", "-c", "set -e; export RESEARCH_OUT={out} TMPDIR={out}/tmp MPLCONFIGDIR={out}/tmp/mpl "
                           "WANDB_MODE=offline WANDB_DIR={out}; mkdir -p {out}/desk {out}/tmp; " + PY + " scripts/run_encoder.py desk; "
                           + PY + f" research/exp/cache.py --out {{cache}}/{cache}; " + PY + " research/exp/epoch_variant.py --run {out} "
                           f"--overlay {ovp} --cache-out {{cache}}/{cache}_late --work {{out}}/late"],
                   "expected_minutes": 90,
                   "summarize": ["python", "-c", "import json; s=json.load(open('{out}/desk/run_summary.json')); print({k: s.get(k) "
                                 "for k in ('best_epoch','best_val_kernel','best_val_zmse','epochs_run')})"],
                   "note": f"E086v-b: {kind} T0 1995 seed {s}"}, open(os.path.join(SP, f"E086vb_desk_t1995_s{s}_{kind}.json"), "w"), indent=1)
        jobs.append(f"E086/{name}"); models.append((f"{kind}_s{s}", f"{cache}_late"))
cmd = ["set -e", ENV] + [f"bash research/bin/fill_esk.sh {{cache}}/desk_tempho_1995_s1 {{cache}}/{c} || true" for _, c in models]
for reg, extra in REG:
    for n, c in models:
        cmd.append(f"[ -f {{out}}/{reg}/{n}/species_components_p96dev.npz ] || bash research/bin/score_both.sh {PY} {{out}}/{reg}/{n} "
                   f"--cache {{cache}}/{c} --rank 24{extra}")
json.dump({"exp": "E086", "name": "e086vb-eval", "group": "gpu", "overlay": "research/overlays/desk_tempho_1995_seed1.json",
           "after_jobs": jobs, "cmd": ["bash", "-c", "; ".join(cmd)], "expected_minutes": 220,
           "note": "E086v-b eval: winter-operator presence arms, late caches, fair_rivals dev r24 full + row 3%"},
          open(os.path.join(SP, "E086vb_eval.json"), "w"), indent=1)
print("ok", len(jobs))
