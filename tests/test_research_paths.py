"""research/lib/paths.job_env: a GPU job must see the GPU even when enqueued from a CPU-only shell."""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "lib"))


def test_gpu_job_drops_an_inherited_empty_cuda_visible_devices(monkeypatch):
    paths = importlib.import_module("paths")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    env = paths.job_env("gpu")
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert paths.job_env("cpu")["CUDA_VISIBLE_DEVICES"] == ""


def test_gpu_job_keeps_an_explicit_device_choice(monkeypatch):
    paths = importlib.import_module("paths")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert paths.job_env("gpu")["CUDA_VISIBLE_DEVICES"] == "0"
