<h1 align="center">Mflux-ComfyUI 2.1.0</h1>

<p align="center">
    <strong>ComfyUI nodes for mflux 0.13.1 (Apple Silicon/MLX)</strong><br/>
    <a href="README.md.kr">한국어</a> | <a href="README_zh.md">中文</a>
</p>

## Overview

This fork upgrades the original nodes to use **mflux 0.13.1** while keeping ComfyUI workflow compatibility. It leverages the new unified architecture of mflux 0.13.x to support standard FLUX generation as well as specialized variants like Fill, Depth, Redux, and Z-Image Turbo.

- **Backend**: mflux 0.13.1 (requires macOS + Apple Silicon).
- **Graph compatibility**: Legacy inputs are migrated internally so your old graphs still work.
- **Unified Loading**: Seamlessly handles local paths, HuggingFace repo IDs, and predefined aliases (e.g., `dev`, `schnell`).

## What's New in mflux 0.13.1
This version brings significant backend enhancements:
- **Z-Image Turbo Support**: Support for the fast, distilled Z-Image variant optimized for speed (6B parameters).
- **FIBO VLM Quantization**: Support for quantized (3/4/5/6/8-bit) FIBO VLM commands (`inspire`/`refine`).
- **Unified Architecture**: Improved resolution for models, LoRAs, and tokenizers.

## Key features

- **Core Generation**: Quick text2img and img2img in one node (`QuickMfluxNode`).
- **Z-Image Turbo**: Dedicated node for the new high-speed model (`MFlux Z-Image Turbo`).
- **FLUX Tools Support**: Dedicated nodes for **Fill** (Inpainting), **Depth** (Structure guidance), and **Redux** (Image variation).
- **ControlNet**: Canny preview and best‑effort conditioning; includes support for the **Upscaler** ControlNet.
- **LoRA Support**: Unified LoRA pipeline (quantize must be 8 when applying LoRAs).
- **Quantization**: Rich options (None, 3, 4, 5, 6, 8-bit) for memory efficiency.
- **Metadata**: Saves full generation metadata (PNG + JSON) compatible with mflux CLI tools.

## Installation

### Using ComfyUI-Manager (Recommended)
- Search for “Mflux-ComfyUI” and install.

### Manual Installation
1. Navigate to your custom nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/rurounigit/Mflux-ComfyUI.git
   ```
3. Activate your ComfyUI virtual environment and install dependencies:
   ```bash
   # Example for standard venv
   source /path/to/ComfyUI/venv/bin/activate

   pip install --upgrade pip wheel setuptools
   pip install 'mlx>=0.27.0' 'huggingface_hub>=0.26.0'
   pip install 'mflux==0.13.1'
   ```
4. Restart ComfyUI.

**Note**: `mflux 0.13.1` requires `mlx >= 0.27.0`. If you are on an older version, please upgrade.

## Nodes

### MFlux/Air (Standard)
- **QuickMfluxNode**: The all-in-one node for standard FLUX txt2img, img2img, LoRA, and ControlNet.
- **MFlux Z-Image Turbo**: Dedicated node for Z-Image generation (optimized defaults: 9 steps, no guidance).
- **Mflux Models Loader**: Select local models from `models/Mflux`.
- **Mflux Models Downloader**: Download quantized or full models directly from HuggingFace.
- **Mflux Custom Models**: Compose and save custom quantized variants.

### MFlux/Pro (Advanced)
- **Mflux Fill**: FLUX.1-Fill support for inpainting and outpainting (requires mask).
- **Mflux Depth**: FLUX.1-Depth support for structure-guided generation.
- **Mflux Redux**: FLUX.1-Redux support for mixing image styles/structures.
- **Mflux Upscale**: Image upscaling using the Flux ControlNet Upscaler.
- **Mflux Img2Img / Loras / ControlNet**: Modular loaders for building custom pipelines.

## Usage Tips

- **Z-Image Turbo**: Use the dedicated node. It defaults to **9 steps** and **0 guidance** (required for this model).
- **LoRA Compatibility**: LoRAs currently require the base model to be loaded with `quantize=8` (or None).
- **Dimensions**: Width and Height should be multiples of 16 (automatically adjusted if needed).
- **Guidance**:
  - `dev` models respect guidance (default ~3.5).
  - `schnell` models ignore guidance (safe to leave as is).
- **Paths**:
  - Quantized models: `ComfyUI/models/Mflux`
  - LoRAs: `ComfyUI/models/loras` (create a `Mflux` subdirectory to keep them organized).
  - Automatically downloaded models from HuggingFace (like filipstrand/Z-Image-Turbo-mflux-4bit when using the Z-Image Turbo node for the first time): `User/.cache/huggingface/hub`, press `Cmd + Shift + .` to unhide the .cache folder.

## Workflows

Check the `workflows` folder for JSON examples:
- `Mflux text2img.json`
- `Mflux img2img.json`
- `Mflux ControlNet.json`
- `Mflux Fill/Redux/Depth` examples (if available)
- `Mflux Z-Image Turbo.json`
- `Mflux Z-Image Turbo img2img lora.json`

If nodes appear red in ComfyUI, use the Manager's "Install Missing Custom Nodes" feature.

## Acknowledgements

- **mflux** by [@filipstrand](https://github.com/filipstrand) and contributors.
- Original ComfyUI integration concepts by **raysers**.
- MFlux-ComfyUI 2.0.0 by **joonsoome**.
- Some code structure inspired by **MFLUX-WEBUI**.

## License

MIT