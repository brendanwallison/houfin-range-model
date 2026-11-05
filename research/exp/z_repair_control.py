"""Prepare/run a frozen, mask-valid covariate-only seasonal DESK control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',choices=['backcast40','backcast20','production'],default='backcast40')
    p.add_argument('--seed',type=int,default=1)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='prepare':
        from src.config_utils import load_config
        repo=Path(__file__).resolve().parents[2]
        code=root/'control_code'
        if code.exists():raise RuntimeError('Refuse to overwrite an existing source snapshot')
        for dirname in ['src','research/exp','research/lib']:
            for source in (repo/dirname).rglob('*.py'):
                if '__pycache__' in source.parts:continue
                target=code/source.relative_to(repo)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        for name in ['esk_desk_config.json','data_config.json']:
            target=code/'config'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(repo/'config'/name,target)
        for name in ['research/__init__.py']:
            if (repo/name).exists():shutil.copyfile(repo/name,code/name)
        data=root/'data'/a.task
        summer=root/'teachers'/a.task/'summer/spacetime'
        winter=root/'teachers'/a.task/'winter_full/spacetime'
        aux=root/'aux'/a.task/'full'
        cfg=load_config(repo/'config/overlays/desk_final_clean_v2_s1.json')
        cfg['paths']['desk_output_dir']=str(root/'models'/a.task/f'control_s{a.seed}')
        cfg['target']['points_dir']=str(data/'summer_train_points')
        cfg['trend']['community_trend_list']=str(data/'summer_species.csv')
        cfg['desk']['z_dir']=str(summer)
        cfg['desk']['seed']=a.seed
        cfg['desk']['epochs']=500;cfg['desk']['stop_at_epoch']=500
        cfg['desk']['checkpoint_every']=10;cfg['desk']['snapshot_every']=50
        cfg['desk']['trend'].update(holdout_frac=0.,min_val_cells=0,holdout_years=[])
        cfg['desk']['select_on_withheld']=False
        cfg['desk']['aux_ruzicka']['path']=str(aux/'cbc_ruzicka.npz')
        cfg['desk']['aux_presence']['path']=str(aux/'cbc_presence_early_w.npz')
        head=cfg['desk']['aux_metric_heads'][0]
        head['path']=str(aux/'cbc_ruzicka.npz')
        head['coord_target']['basis_dir']=str(winter)
        head['coord_target']['path']=str(aux/'winter_coords.npz')
        cfg['latent_cube'].update(z_dir=str(summer),z_ref_path=str(summer/'Z.npy'),
            mask_ref_path=str(summer/'valid_mask.npy'),
            model_path=str(Path(cfg['paths']['desk_output_dir'])/'env_model_semisup.pth'),
            output_dir=str(root/'cubes'/a.task/f'control_s{a.seed}'),include_private=True)
        cfg['z_repair']={'data_package':str(root/'data/DONE.json'),
            'task':a.task,'seed':a.seed,'arm':'covariate_only_control',
            'validation_policy':'No evaluation observations in training sources; fixed500 budget, no epoch selection on historical outcomes.',
            'initial_checkpoint':None,'presence_weight_policy':'Inherited0.01 for matched control, not claimed calibrated.'}
        config_path=code/'config/z_repair_control.json'
        config_path.write_text(json.dumps(cfg,indent=2))
        manifest={str(q.relative_to(code)):hashlib.sha256(q.read_bytes()).hexdigest()
                  for q in sorted(code.rglob('*')) if q.is_file()}
        (code/'FROZEN_SOURCE.json').write_text(json.dumps(manifest,indent=2))
        print(code,flush=True)
        return
    # Run only from the snapshot, so later workspace edits cannot change this run.
    code=Path(__file__).resolve().parents[2]
    if code!=root/'control_code':raise ValueError('Run the frozen snapshot entry point')
    for name,sha in json.loads((code/'FROZEN_SOURCE.json').read_text()).items():
        if hashlib.sha256((code/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Source changed: '+name)
    from src.config_utils import load_config
    cfg=load_config(code/'config/z_repair_control.json')
    if not (root/'aux'/a.task/'full/DONE.json').exists():
        raise ValueError('Auxiliary preparation did not complete')
    import numpy as np
    keys=np.load(Path(cfg['target']['points_dir'])/'point_index.npy')
    start=json.loads((root/'data'/a.task/'manifest.json').read_text())['specification']['summer_start']
    assert (keys[:,2]>=start).all()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Control training requires the queued local GPU')
    print('Frozen masked control; fixed500 epochs; historical winter supervision retained',flush=True)
    from src.community_encoder.train_DESK.desk_training import run_desk_experiment
    output=Path(cfg['paths']['desk_output_dir'])
    if not (output/'CONTROL_TRAINING_DONE').exists():
        run_desk_experiment(cfg)
        (output/'CONTROL_TRAINING_DONE').write_text('Training completed; no test outcome used for epoch selection.\n')
    from src.community_encoder.build_final_z_cube import build_spacetime_cube
    build_spacetime_cube(cfg)
    (output/'CONTROL_CUBE_DONE').write_text('Frozen control cube exported; not a promoted Z.\n')


if __name__=='__main__':main()
