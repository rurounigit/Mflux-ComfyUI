import os
import sys
import shutil
from unittest.mock import MagicMock
import pytest

@pytest.fixture(scope="session", autouse=True)
def setup_dummy_directories():
    """Create dummy directories expected by folder_paths mocks."""
    dirs = ["input", "output", "models", "models/Mflux", "models/loras"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    yield
    # Cleanup (optional, GH Actions cleans up anyway)
    # for d in dirs:
    #     if os.path.exists(d): shutil.rmtree(d)

# 1. Mock ComfyUI 'folder_paths' module
if "folder_paths" not in sys.modules:
    folder_paths_mock = MagicMock()
    # Define paths as strings (relative to test root)
    folder_paths_mock.models_dir = "models"
    folder_paths_mock.get_input_directory.return_value = "input"
    folder_paths_mock.get_output_directory.return_value = "output"
    folder_paths_mock.get_save_image_path.return_value = ("output", "Mflux_0001", 0, "Mflux", "Mflux")
    folder_paths_mock.get_filename_list.return_value = ["dummy_lora.safetensors"]
    folder_paths_mock.get_annotated_filepath.side_effect = lambda x: os.path.abspath(f"input/{x}")
    folder_paths_mock.exists_annotated_filepath.return_value = True

    sys.modules["folder_paths"] = folder_paths_mock

# 2. Mock 'comfy' package
if "comfy" not in sys.modules:
    comfy_mock = MagicMock()
    sys.modules["comfy"] = comfy_mock
    sys.modules["comfy.utils"] = MagicMock()

# 3. Mock mflux and mlx for non-Apple Silicon environments
try:
    import mflux
    import mlx
except ImportError:
    mlx_mock = MagicMock()
    sys.modules["mlx"] = mlx_mock
    sys.modules["mlx.core"] = mlx_mock

    mflux_mock = MagicMock()
    sys.modules["mflux"] = mflux_mock

    modules_to_mock = [
        "mflux.models",
        "mflux.models.common",
        "mflux.models.common.config",
        "mflux.callbacks",
        "mflux.callbacks.callback_registry",
        "mflux.models.flux",
        "mflux.models.flux.variants",
        "mflux.models.flux.variants.txt2img",
        "mflux.models.flux.variants.txt2img.flux",
        "mflux.models.flux.variants.controlnet",
        "mflux.models.flux.variants.controlnet.flux_controlnet",
        "mflux.models.flux.variants.fill",
        "mflux.models.flux.variants.fill.flux_fill",
        "mflux.models.flux.variants.depth",
        "mflux.models.flux.variants.depth.flux_depth",
        "mflux.models.flux.variants.redux",
        "mflux.models.flux.variants.redux.flux_redux",
        "mflux.models.z_image",
        "mflux.models.z_image.variants",
        "mflux.models.z_image.variants.turbo",
        "mflux.models.z_image.variants.turbo.z_image_turbo",
        "mflux.controlnet",
        "mflux.controlnet.controlnet_util",
        "mflux.utils",
        "mflux.utils.version_util",
    ]

    for mod_name in modules_to_mock:
        sys.modules[mod_name] = MagicMock()

# Disable native imports during tests
os.environ.setdefault("MFLUX_COMFY_DISABLE_CONTROLNET_IMPORT", "1")
os.environ.setdefault("MFLUX_COMFY_DISABLE_MLX_IMPORT", "1")
os.environ.setdefault("MFLUX_COMFY_DISABLE_MFLUX_IMPORT", "1")