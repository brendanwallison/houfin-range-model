"""Backward-compatible shim. The canonical loader now lives in ``src/config_utils.py``.

Community-encoder entrypoints put ``src/`` on ``sys.path`` (so the module is
importable as ``config_utils``); the age-model pipeline puts the repo root on
``sys.path`` (so it is ``src.config_utils``). Support both.
"""
try:
    from src.config_utils import load_config, resolve_repo_root
except ModuleNotFoundError:
    from config_utils import load_config, resolve_repo_root

__all__ = ["load_config", "resolve_repo_root"]
