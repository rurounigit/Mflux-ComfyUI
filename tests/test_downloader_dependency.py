import sys
import os
import pytest
from unittest.mock import MagicMock

# Import the module under test
import Mflux_Comfy.Mflux_Air as mflux_air

def test_download_requires_huggingface(monkeypatch):
    """
    Verify that download_hg_model raises RuntimeError if huggingface_hub
    (snapshot_download) is not available.
    """
    # Simulate snapshot_download being None (as set in the try/except block of Mflux_Air)
    monkeypatch.setattr(mflux_air, "snapshot_download", None)

    with pytest.raises(RuntimeError) as excinfo:
        mflux_air.download_hg_model("flux.1-dev-mflux-4bit")

    assert "huggingface_hub is required" in str(excinfo.value)

def test_download_logic(monkeypatch, tmp_path):
    """
    Verify the download logic:
    1. Calls snapshot_download with correct args.
    2. Skips download if directory exists.
    3. Forces download if force_redownload is True.
    """
    # Mock the snapshot_download function
    mock_download = MagicMock()
    monkeypatch.setattr(mflux_air, "snapshot_download", mock_download)

    # Redirect mflux_dir to a temporary directory for this test
    monkeypatch.setattr(mflux_air, "mflux_dir", str(tmp_path))

    model_name = "flux.1-schnell-mflux-4bit"
    expected_repo_id = "madroid/flux.1-schnell-mflux-4bit"

    # --- Case 1: Directory does not exist (Should download) ---
    mflux_air.download_hg_model(model_name)

    mock_download.assert_called_once()
    call_kwargs = mock_download.call_args[1]
    assert call_kwargs["repo_id"] == expected_repo_id
    assert call_kwargs["local_dir_use_symlinks"] is False

    # --- Case 2: Directory exists (Should skip) ---
    mock_download.reset_mock()

    # Create the model directory to simulate it already being downloaded
    model_dir = tmp_path / model_name
    model_dir.mkdir()

    mflux_air.download_hg_model(model_name, force_redownload=False)

    mock_download.assert_not_called()

    # --- Case 3: Directory exists + Force Redownload (Should download) ---
    mflux_air.download_hg_model(model_name, force_redownload=True)

    mock_download.assert_called_once()