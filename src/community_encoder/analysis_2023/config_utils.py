import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[3] / "config" / "esk_desk_config.json"
    else:
        config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
