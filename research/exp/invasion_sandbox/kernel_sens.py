"""KERNEL-SENS (2026-11-05, user-approved): dispersal-kernel sensitivity inputs for the leading candidate.

The kernels are NOT learnable here (every opened dispersal knob was compensated; BBS at 54 km sees density, not movement),
so the candidate is refitted on inputs rebuilt with fixed alternative kernels and the compensation pattern is read.
Inputs are rebuilt with the exact FIELD54 Stage F recipe (code_bc/research/field54/f_inputs.slurm): Z_disp from the
field54prod_ema s64 cube with cell-integrated kernels and the land-share floor 1e-4 (code_bc), model_inputs from the
patched snap_d0e59ff snapshot; ONLY dispersal.juvenile_mdd_km / adult_mdd_km differ (shapes, band splits, steps fixed),
then the established-1951 derivation (inv_timestep 38 -> 49, exactly the INPUT_CONTRACT of grid54/inputs/established1951).
  overlay --arm NAME --juv KM --adult KM --root DIR     -> DIR/NAME/overlay.json (production overlay + kernel + paths)
  derive  --src INPUT --dst DIR                         -> est-1951 input (metadata copy, .dat symlinks, INPUT_CONTRACT)
"""
import argparse, hashlib, json, os, pickle
from pathlib import Path

PROD = os.environ.get('KSENS_PROD_OVERLAY', '/work/07980/bwa386/ls6/houfin/processed/field54/stageF/overlays/grid54_age_field54prod_ema_obsroute.json')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def overlay(a):
    d = json.load(open(PROD))
    base = Path(a.root) / a.arm
    base.mkdir(parents=True, exist_ok=False)
    d['dispersal']['juvenile_mdd_km'] = float(a.juv)
    d['dispersal']['adult_mdd_km'] = float(a.adult)
    d['path_features']['output_dir'] = str(base / 'latent_avian_paths')
    d['raw_z_dir'] = str(base / 'latent_avian_paths')
    d['input_dir'] = str(base / 'numpyro_input')
    d['path_diagnostics_dir'] = str(base / 'path_diagnostics')
    d['_kernel_sens'] = dict(arm=a.arm, juvenile_mdd_km=float(a.juv), adult_mdd_km=float(a.adult), parent_overlay=PROD,
                             note='KERNEL-SENS sensitivity input (Claude 2026-11-05); only the kernel MDDs and paths differ')
    (base / 'overlay.json').write_text(json.dumps(d, indent=2) + '\n')
    print(base / 'overlay.json')


def derive(a):
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=False)
    m = pickle.load(open(src / 'metadata.pkl', 'rb'))
    years = list(map(int, m['years']))
    old = int(m['inv_timestep'])
    assert years[old] == 1940, (old, years[old])
    m['inv_timestep'] = years.index(int(a.year))
    for f in src.iterdir():
        if f.name != 'metadata.pkl':
            os.symlink(f.resolve(), dst / f.name)
    with open(dst / 'metadata.pkl', 'wb') as fh:
        pickle.dump(m, fh)
    contract = dict(parent=str(src), derived=str(dst), changed_metadata_keys=['inv_timestep'], old_year=1940, new_year=int(a.year),
                    old_index=old, new_index=m['inv_timestep'], old_metadata_sha256=sha(src / 'metadata.pkl'),
                    new_metadata_sha256=sha(dst / 'metadata.pkl'), observations_features_grid_years_unchanged=True,
                    files='symlinks to the parent input', recipe='as research/grid54/inputs/established1951/INPUT_CONTRACT.json')
    (dst / 'INPUT_CONTRACT.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(json.dumps(contract))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='cmd', required=True)
    o = s.add_parser('overlay'); o.add_argument('--arm', required=True); o.add_argument('--juv', type=float, required=True)
    o.add_argument('--adult', type=float, required=True); o.add_argument('--root', required=True)
    d = s.add_parser('derive'); d.add_argument('--src', required=True); d.add_argument('--dst', required=True); d.add_argument('--year', type=int, default=1951)
    a = p.parse_args()
    overlay(a) if a.cmd == 'overlay' else derive(a)
