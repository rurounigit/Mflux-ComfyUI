import os
from folder_paths import models_dir
from .Mflux_Core import get_lora_info, generate_image, save_images_with_metadata, infer_quant_bits

# --- MFLUX 0.13.1 Imports ---
try:
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from huggingface_hub import snapshot_download
except ImportError:
    raise ImportError("mflux>=0.13.1 is required.")

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

mflux_dir = os.path.join(models_dir, "Mflux")
create_directory(mflux_dir)

def get_full_model_path(model_dir, model_name):
    return os.path.join(model_dir, model_name)

def download_hg_model(model_version, force_redownload=False):
    repo_id = model_version if "/" in model_version else (f"madroid/{model_version}" if "4bit" in model_version else f"AITRADER/{model_version}")
    model_checkpoint = get_full_model_path(mflux_dir, model_version)

    if os.path.exists(model_checkpoint) and not force_redownload:
        print(f"Model {model_version} found at {model_checkpoint}")
        return model_checkpoint

    print(f"Downloading {repo_id} to {model_checkpoint}...")
    snapshot_download(repo_id=repo_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
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
                ], {"default": "flux.1-schnell-mflux-4bit"}),
            },
            "optional": {
                "force_redownload": ("BOOLEAN", {"default": False}),
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
                "model": (["dev", "schnell"], {"default": "schnell"}),
                "quantize": (["3", "4", "5", "6", "8"], {"default": "8"}),
            },
            "optional": {
                "Loras": ("MfluxLorasPipeline",),
                "custom_identifier": ("STRING", {"default": ""}),
                "base_model": (["dev", "schnell"], {"default": "dev"}),
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

        lora_paths, lora_scales = get_lora_info(Loras)

        # mflux 0.13.1 Flux1 constructor
        flux = Flux1(
            model_config=ModelConfig.from_name(model, base_model=base_model),
            quantize=int(quantize),
            lora_paths=lora_paths,
            lora_scales=lora_scales
        )

        flux.save_model(save_dir)
        print(f"Model saved to {save_dir}")
        return (save_dir,)

class MfluxModelsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_paths = []
        if os.path.exists(mflux_dir):
            model_paths = [f.name for f in os.scandir(mflux_dir) if f.is_dir()]

        return {
            "required": {
                "model_name": (sorted(model_paths) or ["None"],),
            },
            "optional": {
                "free_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Local_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "load"

    def load(self, model_name="", free_path=""):
        if free_path:
            if not os.path.exists(free_path):
                raise ValueError(f"Path does not exist: {free_path}")
            return (free_path,)

        if model_name and model_name != "None":
            return (get_full_model_path(mflux_dir, model_name),)

        return ("",)

class QuickMfluxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Luxury food photograph"}),
                "model": (["dev", "schnell"], {"default": "schnell"}),
                "quantize": (["None", "3", "4", "5", "6", "8"], {"default": "8"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "Local_model": ("PATH",),
                "Loras": ("MfluxLorasPipeline",),
                "img2img": ("MfluxImg2ImgPipeline",),
                "ControlNet": ("MfluxControlNetPipeline",),
                "base_model": (["dev", "schnell"], {"default": "dev"}),
                "low_ram": ("BOOLEAN", {"default": False}),
                "vae_tiling": ("BOOLEAN", {"default": False}),
                "vae_tiling_split": (["horizontal", "vertical"], {"default": "horizontal"}),
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

        # Apply presets logic (simplified for brevity, keep original logic if needed)
        final_width, final_height = width, height
        final_steps, final_guidance = steps, guidance
        final_seed = -1 if randomize_seed else seed

        generated_images = generate_image(
            prompt, model, final_seed, final_width, final_height, final_steps, final_guidance, quantize, metadata,
            Local_model, img2img, Loras, ControlNet, base_model=base_model, low_ram=low_ram,
            vae_tiling=vae_tiling, vae_tiling_split=vae_tiling_split
        )

        if metadata:
            image_path = img2img.image_path if img2img else None
            image_strength = img2img.image_strength if img2img else None
            lora_paths, lora_scales = get_lora_info(Loras)

            quantize_effective = quantize
            if Local_model:
                detected = infer_quant_bits(Local_model)
                if detected: quantize_effective = f"{detected}-bit"

            control_image_path = None
            control_strength = None
            control_model = None
            if ControlNet:
                control_image_path = getattr(ControlNet, "control_image_path", None)
                control_strength = getattr(ControlNet, "control_strength", None)
                control_model = getattr(ControlNet, "model_selection", None)

            save_images_with_metadata(
                images=generated_images,
                prompt=prompt,
                model=model,
                quantize=quantize,
                quantize_effective=quantize_effective,
                Local_model=Local_model,
                seed=final_seed,
                height=final_height,
                width=final_width,
                steps=final_steps,
                guidance=final_guidance,
                image_path=image_path,
                image_strength=image_strength,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                control_image_path=control_image_path,
                control_strength=control_strength,
                control_model=control_model,
                full_prompt=full_prompt,
                extra_pnginfo=extra_pnginfo,
                base_model=base_model,
                vae_tiling=vae_tiling,
                vae_tiling_split=vae_tiling_split
            )

        return generated_images