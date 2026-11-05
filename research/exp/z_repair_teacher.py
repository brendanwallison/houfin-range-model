"""Fit a fresh ESK teacher from an already filtered reconstruction point set."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--landmarks', type=int, default=16000)
    args = ap.parse_args()
    import torch
    import numpy as np
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.esk_kernel import run_spacetime_esk
    if torch.cuda.is_available():
        raise ValueError('This preparation is CPU-only; set CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(args.threads)
    if (args.out/'spacetime/meta.json').exists():
        raise RuntimeError('Refuse to overwrite an existing teacher')
    pm = json.loads((args.points/'points_meta.json').read_text())
    if 'z_repair_task' not in pm:
        raise ValueError('Require a training-only z_repair_data point set')
    if not (args.package/'DONE.json').exists():
        raise ValueError('Require a completed reconstruction data package')
    keys = np.load(args.points/'point_index.npy')
    task = pm['z_repair_task']
    if pm.get('target_source') == 'bbs_raw' and np.any(keys[:,2] < task['summer_start']):
        raise ValueError('Forbidden summer year in teacher sources')
    cfg = load_config()
    cfg['target'] = {'points_dir': str(args.points.resolve())}
    cfg['paths']['esk_output_dir'] = str(args.out.resolve())
    cfg['esk']['seed'] = 20261026
    cfg['esk']['spacetime'].update(latent_dim=64, n_landmarks=args.landmarks,
                                  landmark_mode='diverse', augment_windowed=True,
                                  augment_half_width=2, augment_min_years=2)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'resolved_config.json').write_text(json.dumps(cfg, indent=2))
    run_spacetime_esk(cfg)
    identity = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), args.package/'DONE.json',
                          args.points/'points_meta.json', args.points/'point_index.npy',
                          args.points/'X_points.npy']}
    (args.out/'TRAINING_ONLY_TEACHER.json').write_text(json.dumps({
        'source_sha256': identity, 'task': task,
        'existing_encoder_checkpoint_used': False,
        'landmark_budget': args.landmarks, 'rank': 64,
        'source_rows': len(keys)}, indent=2))


if __name__ == '__main__':
    main()
