"""Shared configuration loader for both the community-encoder (ESK/DESK) and
age-structured model pipelines.

A single JSON file per pipeline lives under ``config/``. Callers either pass an
explicit path, set the matching environment variable, or fall back to the
default file shipped in the repo.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Default config filenames, keyed by a short pipeline name.
ESK_DESK_CONFIG = "esk_desk_config.json"
AGE_MODEL_CONFIG = "age_model_config.json"
DATA_CONFIG = "data_config.json"

# Environment variable that overrides the default path for each pipeline.
ESK_DESK_ENV = "ESK_DESK_CONFIG"
AGE_MODEL_ENV = "AGE_MODEL_CONFIG"
DATA_ENV = "DATA_CONFIG"


def resolve_repo_root() -> Path:
    """Repository root, i.e. the directory that contains ``config/`` and ``src/``."""
    return Path(__file__).resolve().parents[1]


def config_dir() -> Path:
    return resolve_repo_root() / "config"


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    *,
    default_name: str = ESK_DESK_CONFIG,
    env_var: Optional[str] = ESK_DESK_ENV,
) -> Dict[str, Any]:
    """Load a pipeline config as a dict.

    Resolution order: explicit ``config_path`` > ``$env_var`` > repo default.
    """
    if config_path is None and env_var:
        config_path = os.environ.get(env_var)
    if config_path is None:
        config_path = config_dir() / default_name
    config_path = Path(config_path).expanduser()

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_age_model_config(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for the age-structured model config."""
    return load_config(config_path, default_name=AGE_MODEL_CONFIG, env_var=AGE_MODEL_ENV)


def load_data_config(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for the raw/processed dataset roots (one-off ETL)."""
    return load_config(config_path, default_name=DATA_CONFIG, env_var=DATA_ENV)
