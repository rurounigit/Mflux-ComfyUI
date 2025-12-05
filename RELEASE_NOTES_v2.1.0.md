# Release Notes — v2.1.0

This release upgrades the backend to **mflux 0.13.1** and introduces support for the high-speed **Z-Image Turbo** model. It features a unified model loading architecture, a dedicated Z-Image node, and support for the Flux ControlNet Upscaler.

## Highlights
- **Backend**: Upgraded to mflux 0.13.1 (requires macOS + Apple Silicon).
- **New Model**: Added support for **Z-Image Turbo** (6B parameters), a distilled model optimized for speed.
- **New Node**: `MFlux Z-Image Turbo` dedicated node with optimized defaults (9 steps, 0 guidance, 4-bit quantization).
- **Unified Loading**: Seamlessly handles local paths, HuggingFace repo IDs, and aliases (dev/schnell) without complex configuration.
- **ControlNet**: Added support for the **Flux ControlNet Upscaler** alongside Canny.
- **Quantization**: Full support for 4-bit quantized models (essential for running Z-Image Turbo on consumer hardware).
- **FIBO VLM**: Backend support for quantized FIBO VLM commands.

## Breaking changes
- **Requirement**: Now requires `mflux==0.13.1` and `huggingface_hub>=0.26.0`.
- **Z-Image**: Does not support Classifier-Free Guidance (guidance must be 0). Use the dedicated node to handle this automatically.

## Installation
- **ComfyUI-Manager**: Search “Mflux-ComfyUI” and install/update.
- **Manual**:
  1. cd /path/to/ComfyUI/custom_nodes
  2. git clone https://github.com/rurounigit/Mflux-ComfyUI.git
  3. Activate venv and install deps:
     - pip install --upgrade pip wheel setuptools
     - pip install 'mlx>=0.27.0' 'huggingface_hub>=0.26.0'
     - pip install 'mflux==0.13.1'
  4. Restart ComfyUI

## Usage notes
- **Z-Image Turbo**: Use the dedicated node. It defaults to the 4-bit quantized model (`filipstrand/Z-Image-Turbo-mflux-4bit`) and 9 steps. This will be downloaded the first time you use it and will be saved in: `User/.cache/huggingface/hub`.Press `Cmd + Shift + .` to unhide the .cache folder.
- **Standard Flux**: Continue using `QuickMfluxNode` for Dev/Schnell models.
- **LoRA**: Z-Image Turbo supports LoRAs (e.g., "Technically Color").
- **Width/Height**: Should be multiples of 16.

## Known limitations
- ControlNet support is currently best‑effort (depends on backend build).
- Z-Image Turbo requires downloading ~12GB of weights (for 4-bit) on the first run.

## Thanks
- mflux by @filipstrand and contributors
- MFlux-ComfyUI 2.0.0 by @joonsoome.
- MFLUX-WEBUI by @CharafChnioune (Apache‑2.0 inspiration)