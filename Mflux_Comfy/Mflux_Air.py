import os
import json
from typing import Any, Dict, Optional

from folder_paths import get_filename_list, get_output_directory, models_dir

_skip_mflux_import = os.environ.get("MFLUX_COMFY_DISABLE_MFLUX_IMPORT") == "1"

if not _skip_mflux_import:
    try:
        from mflux.flux.flux import Flux1  # type: ignore
    except Exception as e:
        raise ImportError("[MFlux-ComfyUI] mflux>=0.10.0 is required. Activate your ComfyUI venv and install with: pip install 'mflux==0.10.0'") from e
else:
    class Flux1:  # type: ignore
        def __init__(self, *_, **__):
            raise RuntimeError("mflux runtime disabled (MFLUX_COMFY_DISABLE_MFLUX_IMPORT=1).")

        @classmethod
        def from_name(cls, *_, **__):
            raise RuntimeError("mflux runtime disabled (MFLUX_COMFY_DISABLE_MFLUX_IMPORT=1).")

        def save_model(self, *_, **__):
            raise RuntimeError("mflux runtime disabled (MFLUX_COMFY_DISABLE_MFLUX_IMPORT=1).")

if not _skip_mflux_import:
    try:
        from mflux.config.model_config import ModelConfig  # type: ignore
    except Exception:
        ModelConfig = None  # type: ignore
else:
    ModelConfig = None  # type: ignore

from .Mflux_Core import get_lora_info, generate_image, save_images_with_metadata, is_third_party_model, infer_quant_bits

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def _ensure_hf_cache_root(root: str) -> tuple[str, str]:
    """Force huggingface_hub to use our managed models directory."""
    resolved_root = os.path.abspath(root)
    hub_cache = os.path.join(resolved_root, "hub")
    os.makedirs(resolved_root, exist_ok=True)
    os.makedirs(hub_cache, exist_ok=True)
    targets = {
        "HF_HOME": resolved_root,
        "HF_HUB_CACHE": hub_cache,
        "HUGGINGFACE_HUB_CACHE": hub_cache,
    }
    for env_key, target in targets.items():
        current = os.environ.get(env_key)
        if current and os.path.abspath(current) != os.path.abspath(target):
            print(f"[MFlux-ComfyUI] Redirecting {env_key} from {current} to {target}")
        os.environ[env_key] = target
    return resolved_root, hub_cache

mflux_dir = os.path.join(models_dir, "Mflux")
_HF_HOME_DIR, _HF_HUB_CACHE_DIR = _ensure_hf_cache_root(mflux_dir)
create_directory(mflux_dir)

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None  # type: ignore[assignment]

def get_full_model_path(model_dir, model_name):
    return os.path.join(model_dir, model_name)

def _marker_path(dir_path: str) -> str:
    return os.path.join(dir_path, ".mflux_download.json")

def _write_marker(dir_path: str, repo_id: str):
    try:
        files_count = 0
        for root, _, files in os.walk(dir_path):
            files_count += len([f for f in files if not f.startswith(".")])
        data = {"repo_id": repo_id, "files_count": files_count}
        with open(_marker_path(dir_path), "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[MFlux-ComfyUI] Warning: Failed to write marker for {dir_path}: {e}")

def _has_marker(dir_path: str) -> bool:
    return os.path.exists(_marker_path(dir_path))


def _read_marker(dir_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(_marker_path(dir_path), "r") as f:
            return json.load(f)
    except Exception:
        return None


def get_model_download_info(model_version: str, root_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return download status and metadata for a given model version."""
    base_dir = os.path.abspath(root_dir) if root_dir is not None else mflux_dir
    model_dir = get_full_model_path(base_dir, model_version)
    info: Dict[str, Any] = {"path": model_dir, "downloaded": False}
    marker_data = _read_marker(model_dir)
    if marker_data is not None:
        info["downloaded"] = True
        info["metadata"] = marker_data
    return info


def is_model_downloaded(model_version: str, root_dir: Optional[str] = None) -> bool:
    """Convenience helper for status checks without inspecting the marker payload."""
    return get_model_download_info(model_version, root_dir=root_dir)["downloaded"]

def _looks_like_model_root(dir_path: str) -> bool:
    """Heuristic to decide if a directory is a model root.

    Prefer explicit marker; otherwise check for common subdirs created by mflux saves/downloads.
    Avoid cache/huggingface/download leaf paths.
    """
    if _has_marker(dir_path):
        return True
    try:
        # Exclude any path that clearly lives under caches
        lowered = dir_path.lower()
        if any(seg in lowered for seg in ("/cache/", "/.cache/", "/huggingface/", "/download/")):
            return False
        entries = [e.name for e in os.scandir(dir_path) if e.is_dir()]
        # Typical mflux model roots have some of these components as immediate subfolders
        typical = {"vae", "tokenizer", "text_encoder", "text_encoder_2", "transformer"}
        if len(typical.intersection(set(entries))) >= 2:
            return True
        # Or the directory contains a non-trivial number of files at its root
        file_count = sum(1 for e in os.scandir(dir_path) if e.is_file())
        if file_count >= 5:
            return True
    except Exception:
        pass
    return False

def _materialize_builtin_model(model_version: str, target_dir: str):
    # Map friendly names to alias + quantization
    mapping = {
        "MFLUX.1-dev-8-bit": ("dev", 8),
        "MFLUX.1-schnell-8-bit": ("schnell", 8),
        "flux.1-dev-mflux-4bit": ("dev", 4),
        "flux.1-schnell-mflux-4bit": ("schnell", 4),
    }
    if model_version not in mapping:
        return False
    alias, qbits = mapping[model_version]
    print(f"[MFlux-ComfyUI] Preparing local built-in model: alias={alias}, quantize={qbits} at {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    try:
        if ModelConfig is not None:
            mc = ModelConfig.from_name(model_name=alias, base_model=None)  # type: ignore[attr-defined]
            flux = Flux1(model_config=mc, quantize=int(qbits))
        else:
            flux = Flux1.from_name(model_name=alias, quantize=int(qbits))
        flux.save_model(target_dir)
        _write_marker(target_dir, f"builtin:{alias}:{qbits}")
        print(f"[MFlux-ComfyUI] Built-in model saved: {target_dir}")
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to save built-in model {model_version}: {e}") from e

def download_hg_model(model_version, force_redownload=False):
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required for model downloads. "
            "Activate your ComfyUI virtual environment and install it with: "
            "pip install 'huggingface_hub>=0.24'"
        )
    # If a slash is present, treat it as a full HF repo id; otherwise keep legacy mapping
    repo_id = model_version if "/" in model_version else (f"madroid/{model_version}" if "4bit" in model_version else f"AITRADER/{model_version}")
    model_checkpoint = get_full_model_path(mflux_dir, model_version)  
    must_download = True
    if os.path.exists(model_checkpoint):
        if _has_marker(model_checkpoint):
            if force_redownload:
                print(f"Model {model_version} exists but force_redownload=True. Re-downloading...")
            else:
                print(f"Model {model_version} already exists at {model_checkpoint}. Skipping download.")
                must_download = False
        else:
            print(f"Model folder exists without completion marker: {model_checkpoint}. Resuming download...")

    if must_download:
        # First, see if this is one of our built-in aliases; if so, save from runtime
        if _materialize_builtin_model(model_version, model_checkpoint):
            return model_checkpoint
        print(f"Downloading model {model_version} to {model_checkpoint}...")
        try:
            # Disable hf_transfer acceleration via env for broader compatibility
            os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")
            # Prefer resume + no symlinks; fallback if hub version doesn't support args
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=model_checkpoint,
                    local_dir_use_symlinks=False,
                    resume_download=not force_redownload,
                    force_download=force_redownload,
                )
            except TypeError:
                # Older huggingface_hub versions: retry with minimal args
                if force_redownload and os.path.isdir(model_checkpoint):
                    # Best-effort emulate force by wiping target and re-downloading
                    import shutil
                    try:
                        shutil.rmtree(model_checkpoint)
                    except Exception:
                        pass
                snapshot_download(repo_id=repo_id, local_dir=model_checkpoint)
            _write_marker(model_checkpoint, repo_id)
        except Exception as e:
            # Surface a clear error so the graph stops here rather than failing downstream
            raise RuntimeError(f"Failed to download '{repo_id}' → {model_checkpoint}: {e}") from e
    return model_checkpoint

class MfluxModelsDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": ([
                    "flux.1-schnell-mflux-4bit",
                    "flux.1-dev-mflux-4bit",
                    "MFLUX.1-schnell-8-bit",
                    "MFLUX.1-dev-8-bit",
                    "filipstrand/FLUX.1-Krea-dev-mflux-4bit",
                    "akx/FLUX.1-Kontext-dev-mflux-4bit",
                ], {"default": "flux.1-schnell-mflux-4bit", "tooltip": "Choose a model to download. Items with '/' are HuggingFace repo IDs and will download from the Hub."}),
            }
            ,
            "optional": {
                "force_redownload": ("BOOLEAN", {"default": False, "label_on": "Force", "label_off": "No", "tooltip": "Re-download even if present (use if a local model seems corrupted)."}),
            }
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Downloaded_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "download_model"

    def download_model(self, model_version, force_redownload=False):
        model_path = download_hg_model(model_version, force_redownload=force_redownload)
        return (model_path,)
    
class MfluxCustomModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["dev", "schnell"], {"default": "schnell", "tooltip": "Base model alias to start from (dev or schnell)."}),
                "quantize": (["3", "4", "5", "6", "8"], {"default": "8", "tooltip": "Quantization bits used when saving a custom model."}),
            },
            "optional": {
                "Loras": ("MfluxLorasPipeline", {"tooltip": "Optional: Apply LoRAs to the saved model."}),
                "custom_identifier": ("STRING", {"default": "", "tooltip": "Name suffix for the saved model folder (e.g., 'myrun')."}),
                "base_model": (["dev", "schnell"], {"default": "dev", "tooltip": "If 'model' is a HuggingFace repo id, choose its base (dev/schnell)."}),
            }
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Custom_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "save_model"

    def save_model(self, model, quantize, Loras=None, custom_identifier="", base_model="dev"):
        identifier = custom_identifier if custom_identifier else "default"
        save_dir = get_full_model_path(mflux_dir, f"Mflux-{model}-{quantize}bit-{identifier}")
        create_directory(save_dir)
        print(f"Saving model: {model}, quantize: {quantize}, save_dir: {save_dir}")
        lora_paths, lora_scales = get_lora_info(Loras)
        if lora_paths:
            print(f"LoRA paths: {lora_paths}")
            print(f"LoRA scales: {lora_scales}")
        # Support HF repo ids with base_model; else use alias via from_name
        if is_third_party_model(model) or "/" in str(model):
            if ModelConfig is None:
                # Third-party repo ids require ModelConfig to resolve base model and weights
                raise RuntimeError("Third-party HuggingFace models require mflux>=0.10.0 with ModelConfig support. Please upgrade and try again.")
            else:
                model_config = ModelConfig.from_name(model_name=model, base_model=base_model)  # type: ignore[attr-defined]
                flux = Flux1(model_config=model_config, quantize=int(quantize), lora_paths=lora_paths, lora_scales=lora_scales)
        else:
            if ModelConfig is None:
                if lora_paths:
                    print("[MFlux-ComfyUI] Warning: ModelConfig not available; LoRAs will be ignored when saving.")
                flux = Flux1.from_name(model_name=model, quantize=int(quantize))
            else:
                model_config = ModelConfig.from_name(model_name=model, base_model=None)  # type: ignore[attr-defined]
                flux = Flux1(model_config=model_config, quantize=int(quantize), lora_paths=lora_paths, lora_scales=lora_scales)
        flux.save_model(save_dir)
        print(f"Model saved to {save_dir}")
        return (save_dir,)

class MfluxModelsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls.get_sorted_model_paths() or ["None"], {"default": cls.get_sorted_model_paths()[0] if cls.get_sorted_model_paths() else "None", "tooltip": "Pick a local model folder from Mflux models directory."}),  
            },
            "optional": {
                "free_path": ("STRING", {"default": "", "tooltip": "Manually input an absolute model path to override the selection above."}),
            }
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Local_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "load"

    @classmethod
    def get_sorted_model_paths(cls):
        # Only show root-level model folders:
        # - depth 1: e.g., 'flux.1-dev-mflux-4bit' or 'Mflux-dev-8bit-default'
        # - depth 2: e.g., 'filipstrand/FLUX.1-Krea-dev-mflux-4bit'
        candidates = set()
        try:
            for entry in os.scandir(mflux_dir):
                if not entry.is_dir():
                    continue
                level1_path = entry.path
                # If this is a root itself, include
                if _looks_like_model_root(level1_path):
                    candidates.add(os.path.relpath(level1_path, mflux_dir))
                    continue
                # Otherwise, look one level deeper (owner/repo layout)
                try:
                    for sub in os.scandir(level1_path):
                        if not sub.is_dir():
                            continue
                        level2_path = sub.path
                        if _looks_like_model_root(level2_path):
                            rel = os.path.relpath(level2_path, mflux_dir)
                            candidates.add(rel)
                except Exception:
                    pass
        except FileNotFoundError:
            return []
        return sorted(candidates, key=lambda x: x.lower())

    def load(self, model_name="", free_path=""):
        if free_path:
            full_model_path = free_path
            if not os.path.exists(full_model_path):
                raise ValueError(f"Provided custom path does not exist: {full_model_path}")
        elif model_name and model_name != "None":  
            full_model_path = get_full_model_path(mflux_dir, model_name)
        else:
            raise ValueError("Either 'model_name' must be provided or 'free_path' must be used.")

        return (full_model_path,)

class QuickMfluxNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Detect MLX version once for UI hints; non-fatal and no hard dependency here
        def _mlx_notice():
            try:
                import mlx  # type: ignore
                ver = getattr(mlx, "__version__", "unknown")
                return f"MLX: {ver} (recommend >= 0.27.0)"
            except Exception:
                return "MLX not found (recommend MLX >= 0.27.0)"
        MLX_NOTICE = _mlx_notice()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "What to generate. Keep it short and clear for best results.", "default": "Luxury food photograph"}),
                "model": (["dev", "schnell"], {"default": "schnell", "tooltip": f"dev = higher quality, slower; schnell = very fast.  {MLX_NOTICE}"}),
                "quantize": (["None", "3", "4", "5", "6", "8"], {"default": "8", "tooltip": f"Ignored when Local_model is connected (uses saved precision). Otherwise: lower bits = smaller/faster, 8 = best compatibility. 'None' uses full precision.  {MLX_NOTICE}"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "-1 = random each run. Use a fixed number to reproduce results."}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1536, "step": 8, "tooltip": "Image width. Use steps of 8 for best performance."}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1536, "step": 8, "tooltip": "Image height. Use steps of 8 for best performance."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100, "tooltip": "More steps = more detail but slower. 25 is a good default for dev."}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 15.0, "step": 0.1, "tooltip": "Prompt adherence (dev only)."}),
                "metadata": ("BOOLEAN", {"default": True, "label_on": "Save", "label_off": "Skip", "tooltip": "Save PNGs and a JSON with all settings."}),
            },
            "optional": {
                "Local_model": ("PATH", {"tooltip": "Optional: Path to a local Mflux model folder saved earlier."}),
                "Loras": ("MfluxLorasPipeline", {"tooltip": "Optional: Apply LoRAs during generation (quantize must be 8)."}),
                "img2img": ("MfluxImg2ImgPipeline", {"tooltip": "Optional: Use an input image as a starting point (img2img)."}),
                "ControlNet": ("MfluxControlNetPipeline", {"tooltip": "Optional: Add edges (Canny) as guidance. Preview is shown; generation may ignore if backend not supported."}),
                "base_model": (["dev", "schnell"], {"default": "dev", "tooltip": "Required if 'model' is a HuggingFace repo id."}),
                "low_ram": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off", "tooltip": f"Reduce memory usage on smaller GPUs/VRAM. May be slower.  {MLX_NOTICE}"}),
                "size_preset": (["Custom", "512x512", "768x1024", "1024x1024", "1024x768"], {"default": "Custom", "tooltip": "Quickly set common sizes."}),
                "apply_size_preset": ("BOOLEAN", {"default": True, "label_on": "Use preset", "label_off": "Ignore", "tooltip": "Apply the chosen size preset to width/height."}),
                "quality_preset": (["Balanced (25 steps)", "Fast (12 steps)", "High Quality (35 steps)", "Custom"], {"default": "Balanced (25 steps)", "tooltip": "Quickly set a quality/step preset."}),
                "apply_quality_preset": ("BOOLEAN", {"default": True, "label_on": "Use preset", "label_off": "Ignore", "tooltip": "Apply the chosen quality preset to steps (and guidance for dev)."}),
                "randomize_seed": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No", "tooltip": "If Yes, seed is set to -1 so each run is different."}),
                "vae_tiling": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off", "tooltip": "Enable VAE tiling to reduce peak memory for very large images (may introduce seams)."}),
                "vae_tiling_split": (["horizontal", "vertical"], {"default": "horizontal", "tooltip": "When vae_tiling is enabled, choose split orientation for tiling."}),
            },
            "hidden": {
                "full_prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "generate"

    def generate(self, prompt, model, seed, width, height, steps, guidance, quantize="None", metadata=True, Local_model="", img2img=None, Loras=None, ControlNet=None, base_model="dev", low_ram=False, full_prompt=None, extra_pnginfo=None, size_preset="Custom", apply_size_preset=True, quality_preset="Balanced (25 steps)", apply_quality_preset=True, randomize_seed=True, vae_tiling=False, vae_tiling_split="horizontal"):
        # Apply user-friendly presets without breaking existing graphs
        final_width, final_height = width, height
        if apply_size_preset and isinstance(size_preset, str) and size_preset != "Custom" and "x" in size_preset:
            try:
                w_str, h_str = size_preset.split("x")
                final_width, final_height = int(w_str), int(h_str)
            except Exception:
                pass

        final_steps, final_guidance = steps, guidance
        if apply_quality_preset and isinstance(quality_preset, str) and quality_preset != "Custom":
            if "Fast" in quality_preset:
                final_steps = 12
                final_guidance = 2.5 if model == "dev" else guidance
            elif "High Quality" in quality_preset:
                final_steps = 35
                final_guidance = 5.0 if model == "dev" else guidance
            else:  # Balanced
                final_steps = 25
                final_guidance = 3.5 if model == "dev" else guidance

        final_seed = -1 if randomize_seed else seed

        # Sanitize Local_model: only pass it if it's a valid existing path
        local_model_path = Local_model if isinstance(Local_model, str) and Local_model and os.path.exists(Local_model) else None

        generated_images = generate_image(
            prompt,
            model,
            final_seed,
            final_width,
            final_height,
            final_steps,
            final_guidance,
            quantize,
            metadata,
            local_model_path or "",
            img2img,
            Loras,
            ControlNet,
            base_model=base_model,
            low_ram=low_ram,
            vae_tiling=vae_tiling,
            vae_tiling_split=vae_tiling_split,
        )

        image_path = img2img.image_path if img2img else None
        image_strength = img2img.image_strength if img2img else None
        lora_paths, lora_scales = get_lora_info(Loras)

        if metadata:
            # Resolve effective model alias for metadata display
            try:
                lm_low = (local_model_path or "").lower()
            except Exception:
                lm_low = ""
            model_alias = model
            if "dev" in lm_low:
                model_alias = "dev"
            elif "schnell" in lm_low:
                model_alias = "schnell"
            # Determine effective quantization for metadata visibility
            if local_model_path:
                detected = infer_quant_bits(local_model_path)
                quantize_effective = f"{detected}-bit" if detected is not None else "local_model_precision"
            else:
                quantize_effective = quantize
            save_images_params = {
                "images": generated_images,
                "prompt": prompt,
                "model": model_alias,
                "quantize": quantize,
                "quantize_effective": quantize_effective,
                "Local_model": local_model_path or "",
                "seed": final_seed,
                "height": final_height,
                "width": final_width,
                "steps": final_steps,
                "guidance": final_guidance,
                "image_path": image_path,
                "image_strength": image_strength,
                "lora_paths": lora_paths,
                "lora_scales": lora_scales,
                "filename_prefix": "Mflux",
                "base_model": base_model,
                "low_ram": low_ram,
                "vae_tiling": vae_tiling,
                "vae_tiling_split": vae_tiling_split,
                "full_prompt": full_prompt,
                "extra_pnginfo": extra_pnginfo,
            }
            # Add ControlNet metadata if provided
            if ControlNet is not None:
                try:
                    save_images_params.update({
                        "control_image_path": getattr(ControlNet, "control_image_path", None),
                        "control_strength": getattr(ControlNet, "control_strength", None),
                        "control_model": getattr(ControlNet, "model_selection", None),
                    })
                except Exception:
                    pass

            result = save_images_with_metadata(**save_images_params)
            counter = result["counter"]

        return generated_images

    @classmethod
    def VALIDATE_INPUTS(cls, prompt, model, seed, width, height, steps, guidance, quantize="None", metadata=True, Local_model="", img2img=None, Loras=None, ControlNet=None, base_model="dev", low_ram=False, full_prompt=None, extra_pnginfo=None, size_preset="Custom", apply_size_preset=True, quality_preset="Balanced (25 steps)", apply_quality_preset=True, randomize_seed=True):
        # Third-party HF repo ids require base_model
        third_party_prefixes = ("filipstrand/", "akx/", "Freepik/", "shuttleai/")
        if any(str(model).startswith(p) for p in third_party_prefixes) and not base_model:
            return "base_model parameter required for HuggingFace models (dev/schnell)"

        # Quantization validation
        try:
            if quantize not in (None, "None"):
                q = int(quantize)
                if q not in (3, 4, 5, 6, 8):
                    return "Quantization must be one of 3, 4, 5, 6, 8 or None"
        except Exception:
            return "Invalid quantize value"

        # LoRA + quantize < 8 limitation
        if Loras is not None and quantize not in (None, "None"):
            try:
                if int(quantize) < 8:
                    return "LoRAs are not compatible with quantization < 8 in current backend"
            except Exception:
                pass

        # width/height validation: multiples of 8
        # Note: when width/height come from linked nodes, ComfyUI may pass None at validate-time.
        # Only enforce the check when both are concrete integers.
        try:
            if isinstance(width, int) and isinstance(height, int):
                if (width % 8 != 0) or (height % 8 != 0):
                    return "Width and Height must be multiples of 8"
        except Exception:
            # Defer detailed validation to runtime; avoid raising during graph validation
            pass

        # Optional presets sanity
        if apply_size_preset and isinstance(size_preset, str) and size_preset != "Custom":
            if "x" not in size_preset:
                return "Invalid size preset format"

        return True
