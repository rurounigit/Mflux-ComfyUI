# Comfy Registry — Submission Notes

Short description:
> ComfyUI nodes for mflux 0.13.1 (Apple Silicon/MLX). Quick txt2img/img2img with LoRA and ControlNet canny, HF repo support, and metadata saving; keeps legacy graph compatibility. Introduces support for the high-speed Z-Image Turbo model.

PublisherId: rurounigit
DisplayName: Mflux-ComfyUI 2.1.0
Icon: assets/icon.svg

Screenshots (recommended):
- examples/Air.png (text2img)
- examples/Air_img2img.png (img2img)
- examples/Pro_Loras.png (LoRA)
- examples/Pro_ControlNet.png (ControlNet)
- examples/Mflux_Metadata.png (metadata)
- examples/Air_Mflux_Z-Image_Turbo.png (Z-Image Turbo)
- examples/Air_Mflux_Z-Image_Turbo_img2img_lora.png (Z-Image Turbo with img2img and LoRA)


Notes:
- Backend requires mflux >= 0.13.1
- Recommend MLX >= 0.27.0 on Apple Silicon
- Recommend huggingface_hub>=0.26.0
- Quantize choices: None, 3, 4, 5, 6, 8 (default 8)
- LoRAs require quantize=8
