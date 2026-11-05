"""Validate numerical reconstruction, preserve exact original pair supervision, publish inputs."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

root = Path('/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025')
encoder = Path('/home/breallis/houfin/work/houfin/processed/encoder')
old_path = encoder / 'cbc_aux_final/cbc_ruzicka.npz'
new_path = encoder / 'cbc_aux_final_clean/cbc_ruzicka.npz'
old, new = np.load(old_path), np.load(new_path)
assert set(old.files) == set(new.files)
for key in old.files:
    if old[key].dtype.kind in 'SU':
        assert np.array_equal(old[key], new[key]), key
    else:
        # CPU/GPU projection/OLS rounding is ~1e-7; catches wrong masks/calibration.
        np.testing.assert_allclose(old[key], new[key], rtol=1e-5, atol=1e-6, err_msg=key)
old.close(); new.close()
# Completeness never enters this term. Keep the exact artifact, not a rounded reconstruction.
shutil.copyfile(old_path, new_path)
old_coords = encoder / 'cbc_aux_final_v2/winter_coords.npz'
new_coords = encoder / 'cbc_aux_final_v2_clean/winter_coords.npz'
with np.load(old_coords) as original, np.load(new_coords) as rebuilt:
    assert set(original.files) == set(rebuilt.files)
    for key in original.files:
        np.testing.assert_allclose(original[key], rebuilt[key], rtol=1e-5, atol=1e-6, err_msg=key)
shutil.copyfile(old_coords, new_coords)
for folder, name in [('cbc_aux_final_clean', 'cbc_presence_early.npz'),
                     ('cbc_aux_final_v2_clean', 'cbc_presence_early_w.npz')]:
    with np.load(encoder / folder / name) as data, np.load(encoder / folder.replace('_clean', '') / name) as original:
        assert 'houfin' not in data['species']
        for key in ('year', 'row', 'col', 'circle', 'P', 'species'):
            assert np.array_equal(data[key], original[key]), (folder, key)
        for key in ('W', 'b', 'comp'):
            assert np.isfinite(data[key]).all(), (folder, key)
paths = [p for d in ('cbc_aux_final_clean', 'cbc_aux_final_v2_clean')
         for p in (encoder / d).iterdir() if p.is_file()]
paths += list((root / 'original_masks').glob('*.npy'))
paths.append(root.parent / 'FINAL_HOLDOUT_20261025/audit/completeness_training_only.npz')
manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
assert manifest[str(new_path)] == hashlib.sha256(old_path.read_bytes()).hexdigest()
(root / 'aux_manifest.json').write_text(json.dumps(manifest, indent=2))
(root / 'PREPARED').write_text(datetime.now().astimezone().isoformat() + '\n')
print('Validated clean operators; exact original Ruzicka artifact preserved; PREPARED')
