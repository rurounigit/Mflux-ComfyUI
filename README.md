<h1 align="center">Mflux-ComfyUI (v2)</h1>

<p align="center">
    <strong>ComfyUI nodes for mflux 0.10.x (Apple Silicon/MLX)</strong><br/>
    <a href="README.md.kr">한국어</a> | <a href="README_zh.md">中文</a>
</p>

## Overview

This fork upgrades the original nodes to use mflux >= 0.10.0 while keeping ComfyUI workflow compatibility (no graph rewiring required). It adds third‑party model support, richer quantization options, better LoRA handling, and a small MLX version hint in the UI.

- Backend: mflux 0.10.x only (legacy 0.4.1 runtime not supported)
- Graph compatibility: legacy inputs are migrated internally so your old graphs still work
- OS/Accel: macOS + MLX (Apple Silicon). MLX >= 0.27.0 recommended

## Key features

- Quick text2img and img2img in one node (MFlux/Air → QuickMfluxNode)
- LoRA pipeline with validation (quantize must be 8 when applying LoRAs)
- ControlNet Canny preview and best‑effort conditioning
- Third‑party Hugging Face repo IDs (e.g., filipstrand/..., akx/...) with base_model selection
- Quantization choices: None, 3, 4, 5, 6, 8 (default 8)
- Metadata saving (PNG + JSON with both legacy and new fields)

## Installation

Using ComfyUI-Manager (recommended):
- Search for “Mflux-ComfyUI” and install.

Manual:
1) cd /path/to/ComfyUI/custom_nodes
2) git clone https://github.com/joonsoome/Mflux-ComfyUI.git
3) Activate your ComfyUI venv and install mflux 0.10.x:
     - pip install --upgrade pip wheel setuptools
     - pip install 'mlx>=0.27.0' 'huggingface_hub>=0.24'
     - pip install 'mflux==0.10.0'
4) Restart ComfyUI

Notes:
- requirements.txt pins mflux==0.10.0; pyproject uses mflux>=0.10.0
- MLX < 0.27.0 will show a UI hint; upgrade is recommended for stability/perf

## Environment (macOS / zsh)

If you use the repository's virtual environment, this project expects the venv at:

    /Volumes/Macintosh\ HD2/ComfyUI/.venv

Activate it in zsh with:

```zsh
# Use the exact venv path above
source /Volumes/Macintosh\ HD2/ComfyUI/.venv/lib/activate

# Or, when your environment has the typical layout:
# source /Volumes/Macintosh\ HD2/ComfyUI/.venv/bin/activate
```

## Nodes

Under MFlux/Air:
- QuickMfluxNode (txt2img/img2img/LoRA/ControlNet in one)
- Mflux Models Loader (pick saved models under models/Mflux)
- Mflux Models Downloader (fetch curated or third‑party models)
- Mflux Custom Models (compose and save custom quantized variants)

Under MFlux/Pro:
- Mflux Img2Img
- Mflux Loras Loader
- Mflux ControlNet Loader

## Developer notes — additional nodes & tests

This fork also includes several Pro-phase nodes and unit tests that are not part of the upstream quick-start. They are intended for contributors and advanced users:

- MfluxUpscale (MFlux/Pro): an upscaler node that accepts a ComfyUI `IMAGE` tensor as the preferred input and has backwards-compatible support for legacy file-selection widgets. The node writes a temporary PNG when an IMAGE tensor is provided and will save PNG + JSON metadata by default.
- MfluxFill, MfluxDepth, MfluxRedux (MFlux/Pro): additional Flux tools (inpainting/fill, depth-conditioned generation, and redux/variations). These nodes forward phase-2 style config keys such as `masked_image_path`, `depth_image_path`, `redux_image_paths`, and `redux_image_strengths` to the core `generate_image` flow.

Tests added in the `tests/` folder include unit-level tests that:

- Verify parameter migration and metadata saving (`tests/test_metadata_contents.py`).
- Exercise node-level behavior for Fill/Depth/Redux by spying on the `save_images_with_metadata` call (`tests/test_fill_node.py`, `tests/test_depth_node.py`, `tests/test_redux_node.py`).
- Validate the `generate_image` TypeError fallback path used to remove unknown config keys (`tests/test_config_fallback.py`).
- Contain upscale-specific internals tests for converting ComfyUI IMAGE tensors and preserving backwards-compatibility (`tests/test_upscale_internals.py`).

If you plan to extend the pack with more Flux tools (Kontext, In-Context LoRA, CatVTON, Concept-Attention), the existing tests and `Mflux_Comfy/Mflux_Core.py` retry-on-TypeError pattern are useful blueprints for handling API drift across mflux versions.

## Usage tips

- LoRA + quantize < 8 is not supported → set quantize to 8 when using LoRAs
- Width/Height should be multiples of 8
- dev model respects guidance; schnell ignores guidance (safe to set)
- Seed −1 randomizes each run. Presets are provided for size/quality.

### Paths
- Quantized models: ComfyUI/models/Mflux
- LoRAs: ComfyUI/models/loras (tip: create models/loras/Mflux to keep them tidy)
- Full models cache (HF): ~/Library/Caches/mflux (or system cache)

## Workflows

See the workflows folder for examples (txt2img, img2img, LoRA stack, ControlNet canny, and integrations).
If nodes are red, use ComfyUI-Manager “One‑click Install Missing Nodes.”

## Acknowledgements

- mflux by @filipstrand and contributors: https://github.com/filipstrand/mflux
- Some code structure inspired by @CharafChnioune’s MFLUX-WEBUI (Apache‑2.0) with license notes in the referenced areas

## License

MIT
