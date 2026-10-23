"""Generate E086s overlays + specs: CBC Ruzicka term with same-cell cross-year pairs (ruzt) at T0 1996 seeds 1-4, plain
Ruzicka (ruz, E081 recipe) at seeds 3-4, then one GPU eval (fair_rivals, late, as E086n) and one CPU early-list job."""
import json
import os

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OV, SP = os.path.join(R, "research", "overlays"), os.path.join(R, "research", "specs")
PY = "{python}"
SIM = "/home/breallis/houfin/work/houfin/research/E059/species-strata-066f943afb0e/species_similarity.csv"
CBC = "/mnt/e/Datasets/cbc_raw/prepared/cbc_circles_comm.npz"
E086N = "/home/breallis/houfin/work/houfin/research/E086/e086n-cbc-sparse-e178b69df515"
E086P = "/home/breallis/houfin/work/houfin/research/E086/e086p-early-lists-83b6c448bccf/ext"
RUZ = json.load(open(os.path.join(OV, "desk_tempho_1995_seed1_ruz.json")))["desk"]["aux_ruzicka"]
arms = [("ruzt", s) for s in (1, 2, 3, 4)] + [("ruz", s) for s in (3, 4)]
train_jobs = []
for kind, s in arms:
    base = json.load(open(os.path.join(OV, f"desk_tempho_1995_seed{s}.json")))
    ov = json.loads(json.dumps(base))
    aux = dict(RUZ)
    if kind == "ruzt":
        aux.update({"label": "cbc_ruzicka_t", "same_cell_frac": 0.5, "same_cell_min_gap": 10})
    ov["desk"]["aux_ruzicka"] = aux
    ov["_comment"] = (f"E086s: desk_tempho_1995_seed{s} + aux_ruzicka " +
                      ("with half the pairs same-cell, >= 10 y apart (ruzt)" if kind == "ruzt" else "(E081 recipe, ruz)") +
                      f". Controls: seed {s} plain; ruz seed {s}.")
    ovp = f"research/overlays/desk_tempho_1995_seed{s}_{kind}.json"
    if kind == "ruz" and s in (1, 2):
        raise SystemExit("ruz s1/s2 exist (E081)")
    json.dump(ov, open(os.path.join(R, ovp), "w"), indent=1)
    cache = f"desk_tempho_1995_s{s}_{kind}"
    name = f"desk-tempho1995-seed{s}-{kind}"
    spec = {"exp": "E086", "name": name, "group": "gpu", "overlay": ovp,
            "cmd": ["bash", "-c", "set -e; export RESEARCH_OUT={out} TMPDIR={out}/tmp MPLCONFIGDIR={out}/tmp/mpl "
                    "WANDB_MODE=offline WANDB_DIR={out}; mkdir -p {out}/desk {out}/tmp; " + PY + " scripts/run_encoder.py desk; "
                    + PY + f" research/exp/cache.py --out {{cache}}/{cache}; " + PY + " research/exp/epoch_variant.py --run {out} "
                    f"--overlay {ovp} --cache-out {{cache}}/{cache}_late --work {{out}}/late"],
            "expected_minutes": 90,
            "summarize": ["python", "-c", "import json; s=json.load(open('{out}/desk/run_summary.json')); print({k: s.get(k) "
                          "for k in ('best_epoch','best_val_kernel','best_val_zmse','epochs_run')})"],
            "note": f"E086s: {kind} T0 1995 seed {s}"}
    json.dump(spec, open(os.path.join(SP, f"E086s_desk_t1995_s{s}_{kind}.json"), "w"), indent=1)
    train_jobs.append(f"E086/{name}")
# ---- GPU eval: fair_rivals dev r24, late caches, full + row 3% (seeds 1, 2) -- same commands as E086n
cmd = ["set -e", "export BLR_DEVICE=cuda BLR_MAX_ITER=100 BLR_STARTS=0.5 SCORE_TARGETS=dev"]
for kind, s in arms:
    cmd.append(f"bash research/bin/fill_esk.sh {{cache}}/desk_tempho_1995_s1 {{cache}}/desk_tempho_1995_s{s}_{kind}_late || true")
for reg, extra in (("full", ""), ("row03t1", " --thin 0.03 --thin-unit row --thin-seed 1"),
                   ("row03t2", " --thin 0.03 --thin-unit row --thin-seed 2")):
    for kind, s in arms:
        cmd.append(f"[ -f {{out}}/{reg}/{kind}_s{s}/species_components_p96dev.npz ] || bash research/bin/score_both.sh {PY} "
                   f"{{out}}/{reg}/{kind}_s{s} --cache {{cache}}/desk_tempho_1995_s{s}_{kind}_late --rank 24{extra}")
ev = {"exp": "E086", "name": "e086s-eval", "group": "gpu", "overlay": "research/overlays/desk_tempho_1995_seed1.json",
      "after_jobs": train_jobs, "cmd": ["bash", "-c", "; ".join(cmd)], "expected_minutes": 240,
      "note": f"E086s eval: fair_rivals dev r24 late, full + row 3%; controls and ruz s1/s2 reused from {E086N}"}
json.dump(ev, open(os.path.join(SP, "E086s_eval.json"), "w"), indent=1)
# ---- CPU: early-list encodes (selected state) + scores + E086m change fidelity
cmd = ["set -e", "export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="]
enc = {f"{k}_s{s}": f"desk_tempho_1995_s{s}_{k}" for k, s in arms}
enc.update({f"ctl_s{s}": f"desk_tempho_1995_s{s}" for s in (3, 4)})
ext = {k: f"{{out}}/ext/{k}_sel" for k in enc}
ext.update({f"ctl_s{s}": f"{E086P}/ctl_s{s}_sel" for s in (1, 2)}); ext.update({f"ruz_s{s}": f"{E086P}/ruz_s{s}_sel" for s in (1, 2)})
for k, c in enc.items():
    cmd.append(f"[ -f {ext[k]}/z_ema_ext.npy ] || {PY} research/exp/e086p_early_lists.py encode --cache {{cache}}/{c} --cbc {CBC} --out {ext[k]}")
cache_of = dict(enc); cache_of.update({f"ctl_s{s}": f"desk_tempho_1995_s{s}" for s in (1, 2, 3, 4)})
cache_of.update({f"ruz_s{s}": f"desk_tempho_1995_s{s}_ruz" for s in (1, 2)})
pairs = [(f"ruzt_s{s}", f"ctl_s{s}") for s in (1, 2, 3, 4)] + [(f"ruzt_s{s}", f"ruz_s{s}") for s in (1, 2, 3, 4)] \
    + [(f"ruz_s{s}", f"ctl_s{s}") for s in (3, 4)]
for arm, ctl in pairs:
    cmd.append(f"{PY} research/exp/e086p_early_lists.py score --arm {{cache}}/{cache_of[arm]} --arm-ext {ext[arm]} --control "
               f"{{cache}}/{cache_of[ctl]} --control-ext {ext[ctl]} --cbc {CBC} --similarity {SIM} --out {{out}}/scores "
               f"--label {arm}_vs_{ctl}")
for k in [f"ruzt_s{s}" for s in (1, 2, 3, 4)] + [f"ruz_s{s}" for s in (1, 2, 3, 4)] + [f"ctl_s{s}" for s in (1, 2, 3, 4)]:
    cmd.append(f"bash research/bin/fill_esk.sh {{cache}}/desk_tempho_1995_s1 {{cache}}/{cache_of[k]} || true")
    cmd.append(f"{PY} research/exp/e086m_space_compare.py --cache {{cache}}/{cache_of[k]} --out {{out}}/space/{k} --ks 24 64")
cpu = {"exp": "E086", "name": "e086s-early-space", "group": "cpu", "overlay": "research/overlays/desk_tempho_1995_seed1.json",
       "after_jobs": train_jobs, "cmd": ["bash", "-c", "; ".join(cmd)], "expected_minutes": 240,
       "note": "E086s: early-list scores (selected state) and E086m change fidelity"}
json.dump(cpu, open(os.path.join(SP, "E086s_early_space.json"), "w"), indent=1)
print("wrote", len(arms), "training specs + eval + early/space")
