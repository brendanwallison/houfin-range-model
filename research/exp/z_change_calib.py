"""Two-pass replacement for change100's one-time gradient calibration. The original takes torch.autograd.grad through
the per-year checkpointed forwards and OOMs (~172 GB allocated); a single autograd.grad without retain_graph OOMed the
same way, while loss.backward() fits as in normal training (mechanism unconfirmed).

Each pass runs change100's frozen trainer to its first optimizer step, calls .backward() on ONE loss (reference, then
change), reads the gradient norm at mixer[-1].weight, and exits. Seeds are identical, so both passes see the
same warm start, dropout and augmentation draws. The result is written to change100's calibration_path in the same
format; change100 then runs from its unchanged frozen snapshot and skips calibration.

    python research/exp/z_change_calib.py prepare --root R
    python research/exp/z_change_calib.py pass --root R --which g0|g1     (from change100_calib_code)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

OLD = '''        if self.balance is None:
            g0=torch.autograd.grad(reference_loss,parameter,retain_graph=True)[0].norm()'''
NEW = '''        if self.balance is None:
            from research.exp.z_change_calib import single_pass
            single_pass(self, reference_loss, loss, parameter)
            g0=torch.autograd.grad(reference_loss,parameter,retain_graph=True)[0].norm()'''


def single_pass(change_loss, reference_loss, loss, parameter):
    import torch
    which = os.environ['CHANGE_CALIB_PASS']
    # .backward() as in a normal training step: torch.autograd.grad through the per-year checkpoints held every
    # recomputed year at once (172 GB) on this model.
    parameter.grad = None
    (reference_loss if which == "g0" else loss).backward()
    g = parameter.grad.norm()
    out = change_loss.balance_path.parent
    out.mkdir(parents=True, exist_ok=True)
    (out/f'calibration_{which}.json').write_text(json.dumps({'norm': float(g), 'target_sha256': change_loss.spec['sha256']}))
    print(f'[calib] {which} gradient norm {float(g):.6g}', flush=True)
    parts = {k: json.loads((out/f'calibration_{k}.json').read_text()) for k in ('g0', 'g1')
             if (out/f'calibration_{k}.json').exists()}
    if len(parts) == 2:
        g0, g1 = parts['g0']['norm'], parts['g1']['norm']
        if not (g1 > 0 and all(map(lambda v: v == v and abs(v) != float('inf'), (g0, g1)))):
            raise ValueError('Invalid change-gradient calibration')
        balance = min(max(g0 / g1, .01), 1000.)
        change_loss.balance_path.write_text(json.dumps({'weight': balance, 'reference_gradient': g0, 'change_gradient': g1,
            'target_sha256': change_loss.spec['sha256'],
            'status': 'Training-only initial-gradient pilot heuristic; not validated temporal calibration. '
                      'Computed in two passes (z_change_calib.py) because the retained-graph version exceeds GPU memory.'},
            indent=2))
        print('[calib] wrote', change_loss.balance_path, flush=True)
    raise SystemExit(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'pass'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--which', choices=['g0', 'g1'])
    a = p.parse_args()
    root = a.root.resolve()
    code = root/'change100_calib_code'
    if a.stage == 'prepare':
        if code.exists():
            raise RuntimeError('Refuse to overwrite a source snapshot')
        shutil.copytree(root/'change100_code', code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copyfile(Path(__file__), code/'research/exp/z_change_calib.py')
        path = code/'research/exp/z_change_neural.py'
        text = path.read_text()
        if text.count(OLD) != 1:
            raise ValueError('Calibration context changed')
        path.write_text(text.replace(OLD, NEW))
        cfg = json.loads((code/'config/z_change_neural.json').read_text())
        cfg['paths']['desk_output_dir'] = str(root/'models/backcast40/change100_calib')
        (code/'config/z_change_neural.json').write_text(json.dumps(cfg, indent=2))
        print(code, cfg['desk']['change_supervision']['calibration_path'])
        return
    if Path(__file__).resolve().parents[2] != code:
        raise ValueError('Run the calibration snapshot entry point')
    sys.path[:0] = [str(code), str(code/'src')]
    os.environ['CHANGE_CALIB_PASS'] = a.which
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.desk_training import run_desk_experiment
    run_desk_experiment(load_config(code/'config/z_change_neural.json'))
    raise RuntimeError('Trainer returned without reaching the calibration step')


if __name__ == '__main__':
    main()
