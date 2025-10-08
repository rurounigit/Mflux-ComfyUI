# Release Notes — v2.0.0

This release upgrades the backend to mflux 0.10.x and keeps ComfyUI workflow compatibility. It focuses on simpler UI, better LoRA handling, ControlNet Canny support, and a small MLX version hint.

## Highlights
- Backend: mflux >= 0.10.0 (legacy 0.4.1 runtime removed)
- Graph compatibility: legacy `init_image_*` inputs are migrated internally
- Third‑party HF repo support (e.g., filipstrand/..., akx/...) with `base_model` selection
- LoRA flow fixed: construct models via `ModelConfig + Flux1(...)` (supports `lora_paths/scales`)
- ControlNet Canny: preview image and best‑effort conditioning
- Quantize options expanded: None, 3, 4, 5, 6, 8 (default 8)
- MLX version hint in UI tooltips; recommend MLX >= 0.27.0
- Metadata JSON includes both legacy and new fields; adds base_model, low_ram, mflux_version
- HuggingFace downloads relocate to `models/Mflux` and expose helpers to check completion

## Breaking changes
- Requires mflux 0.10.x (remove any 0.4.1 runtime)
- LoRAs with quantize < 8 are not supported

## Installation
- ComfyUI-Manager: search “Mflux-ComfyUI” and install
- Manual:
  1. cd /path/to/ComfyUI/custom_nodes
  2. git clone https://github.com/joonsoome/Mflux-ComfyUI.git
  3. Activate venv and install deps:
     - pip install --upgrade pip wheel setuptools
     - pip install 'mlx>=0.27.0' 'huggingface_hub>=0.24'
     - pip install 'mflux==0.10.0'
  4. Restart ComfyUI

## Usage notes
- dev respects guidance; schnell ignores it
- Width/Height should be multiples of 8
- Seed -1 for randomization; presets for size and quality are available

## Known limitations
- ControlNet support is canny-only and best‑effort (depends on backend build)
- Some LoRAs may log “Unsupported keys for diffusers” (benign)

## Thanks
- mflux by @filipstrand and contributors
- MFLUX-WEBUI by @CharafChnioune (Apache‑2.0 inspiration)
