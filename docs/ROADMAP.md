# Mflux-ComfyUI Roadmap (legacy from 2.0.0 by @joonsoome)

This document tracks the outstanding work required to reach full feature parity with the upstream [mflux](https://github.com/filipstrand/mflux) toolchain while keeping the ComfyUI integration stable.

## Phase 0 — Hardening (current)

- ✅ Make startup resilient to missing optional dependencies (`huggingface_hub`, `tqdm`).
- Align packaging metadata so `huggingface_hub>=0.24` is installed by default.
- Add unit coverage for degraded paths (e.g., downloader without HF Hub, metadata writers without ControlNet).
- Finish converting legacy parameters (`init_*`) to the stable `image_*` naming everywhere.
- Stabilise automated tests by providing fixtures that stub `mflux` and MLX heavy objects.

## Phase 1 — Core feature parity

- **Kontext / In-Context LoRA**: expose nodes mirroring `mflux`'s context-aware pipelines, including UI wiring for reference image / prompt pairs.
- **CatVTON / Apparel transfer**: wrap the virtual try-on flow with clear validation and preview outputs.
- **Concept Attention & Redux**: extend existing Redux node to support batch prompts, strength scheduling, and variation seeds.
- **ControlNet enhancements**: generalise loader node to enumerate all ControlNet checkpoints available in the runtime and support multiple conditions per generation.

## Phase 2 — UX polish & integrations

- Inline dependency health in node tooltips (detect `mflux`, `mlx`, `huggingface_hub`, `torch`).
- Add preset bundles (size, quality, conditioning) that mirror the upstream CLI presets.
- Provide ready-to-run ComfyUI workflow examples for every node category (Air/Pro) with README links.
- Introduce a configuration node for setting global defaults (output directories, metadata toggles, LoRA search paths).

## Phase 3 — Performance & distribution

- Implement in-process caching for ControlNet preprocessors to avoid repeated Canny computation.
- Investigate streaming generation callbacks to surface intermediate previews in ComfyUI.
- Package nightly builds with regression tests on Apple Silicon runners (CI).
- Prepare an official ComfyUI Registry submission once Phase 1 parity is achieved.

## Contribution guidelines

- Use `tests/` to add regression coverage for every new node or parameter migration.
- Keep optional dependencies guarded at import-time; surface runtime guidance instead of blocking ComfyUI boot.
- Document new features in `README.md` and add workflow screenshots under `examples/`.
