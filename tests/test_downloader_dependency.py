import importlib
import os
import sys

import pytest


def reload_mflux_air():
    module_name = "Mflux_Comfy.Mflux_Air"
    module = sys.modules.get(module_name)
    if module is None:
        return importlib.import_module(module_name)
    return importlib.reload(module)


def test_download_requires_huggingface(monkeypatch):
    module = importlib.import_module("Mflux_Comfy.Mflux_Air")
    monkeypatch.setattr(module, "snapshot_download", None, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        module.download_hg_model("flux.1-dev-mflux-4bit")

    assert "huggingface_hub is required" in str(excinfo.value)


def test_hf_cache_redirects(monkeypatch, tmp_path):
    module = reload_mflux_air()
    expected_home = os.path.abspath(module.mflux_dir)
    expected_hub = os.path.join(expected_home, "hub")

    with monkeypatch.context() as patch:
        patch.setenv("HF_HOME", str(tmp_path / "hf-home"))
        patch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache"))
        patch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf-cache-legacy"))
        module = reload_mflux_air()
        assert os.environ["HF_HOME"] == expected_home
        assert os.environ["HF_HUB_CACHE"] == expected_hub
        assert os.environ["HUGGINGFACE_HUB_CACHE"] == expected_hub


def test_is_model_downloaded_helper(tmp_path):
    module = reload_mflux_air()

    local_root = tmp_path / "Mflux"
    model_name = "flux.test-model"
    model_dir = local_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "weights.bin").write_bytes(b"0")

    module._write_marker(str(model_dir), "example/repo")

    info = module.get_model_download_info(model_name, root_dir=str(local_root))
    assert info["downloaded"] is True
    assert info["path"] == str(model_dir)
    assert "metadata" in info
    assert module.is_model_downloaded(model_name, root_dir=str(local_root)) is True

    marker_file = module._marker_path(str(model_dir))
    os.remove(marker_file)

    info_after = module.get_model_download_info(model_name, root_dir=str(local_root))
    assert info_after["downloaded"] is False
    assert info_after["path"] == str(model_dir)
    assert module.is_model_downloaded(model_name, root_dir=str(local_root)) is False
