import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from community_encoder.build_final_z_cube import build_spacetime_cube


if __name__ == "__main__":
    config_path = Path(os.environ.get("ESK_DESK_CONFIG", REPO_ROOT / "config" / "esk_desk_config.json")).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    build_spacetime_cube(config)
