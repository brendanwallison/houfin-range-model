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
SECRETS_CONFIG = "secrets.json"

# Environment variable that overrides the default path for each pipeline.
ESK_DESK_ENV = "ESK_DESK_CONFIG"
AGE_MODEL_ENV = "AGE_MODEL_CONFIG"
DATA_ENV = "DATA_CONFIG"
SECRETS_ENV = "HOUFIN_SECRETS"


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


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively overlay ``overlay`` onto a copy of ``base`` (dicts merge, leaves replace)."""
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_data_config(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load the data-pipeline config, with an optional overlay deep-merged on top.

    The repo default ``config/data_config.json`` is always the base (so every
    product block is present). If ``config_path`` or ``$DATA_CONFIG`` points at a
    *different* file, it is treated as a small **override layer** (e.g. a local
    run's ``datasets_root`` / ``grid`` / ``timeline``) and deep-merged onto the
    base, rather than replacing it wholesale.
    """
    base_path = config_dir() / DATA_CONFIG
    with base_path.open("r", encoding="utf-8") as handle:
        base = json.load(handle)

    overlay_path = config_path or os.environ.get(DATA_ENV)
    if overlay_path:
        overlay_path = Path(overlay_path).expanduser()
        if overlay_path.resolve() != base_path.resolve():
            with overlay_path.open("r", encoding="utf-8") as handle:
                base = _deep_merge(base, json.load(handle))
    return base


def load_secrets(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load API tokens/keys from the gitignored ``config/secrets.json``.

    Returns an empty dict if no secrets file is present, so callers can fall
    back to environment variables (see :func:`get_secret`). Never commit the
    resolved file: ``config/secrets.json`` is gitignored; use
    ``config/secrets.example.json`` as a template.
    """
    if config_path is None:
        config_path = os.environ.get(SECRETS_ENV)
    if config_path is None:
        config_path = config_dir() / SECRETS_CONFIG
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_secret(
    key: str,
    *,
    env_var: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Resolve one secret: ``$env_var`` first, then ``config/secrets.json[key]``.

    The env-var path lets headless/CI runs supply the token without a file.
    Returns ``None`` if neither source provides it.
    """
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val:
            return env_val
    return load_secrets(config_path).get(key)
