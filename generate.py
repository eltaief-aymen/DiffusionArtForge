"""
DiffusionArtForge – AI Image Generation with Stable Diffusion
Author: Eltaief Aymen
Date: 2025-08-31

Description:
-------------
This script generates AI images from text prompts using Hugging Face Diffusers.
It supports:
- Negative prompts to avoid unwanted artifacts.
- Adjustable inference steps and guidance scale.
- Deterministic output using a fixed random seed.
- GPU (CUDA), Apple Silicon (MPS), and CPU execution.
"""

# ==============================
# 1. IMPORTS
# ==============================
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# ==============================
# 2. DEVICE CONFIGURATION
# ==============================
# Prioritize CUDA, then MPS (Apple Silicon), else CPU
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ==============================
# 3. MODEL LOADING
# ==============================
model_id = "runwayml/stable-diffusion-v1-5"

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Replace default scheduler with Euler Ancestral for improved image quality
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Move model to the selected device and optimize memory usage
if device == "cuda":
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # reduces GPU memory usage
else:
    pipe.to(device)

# ==============================
# 4. PROMPT DEFINITION
# ==============================
prompt = "ultra-detailed portrait of a white wolf wearing a tiny red scarf, cinematic lighting, 35mm"
negative_prompt = "blurry, lowres, jpeg artifacts, extra fingers, text, watermark"

# ==============================
# 5. IMAGE GENERATION
# ==============================
# Configuration:
# - num_inference_steps: number of denoising steps (20–35 recommended)
# - guidance_scale: how strongly the image should follow the prompt (5–9 typical)
# - height / width: resolution of the output image (must be multiple of 8)
# - generator: ensures deterministic output using a fixed seed

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=512,
    width=512,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

# ==============================
# 6. SAVE OUTPUT
# ==============================
output_path = "generated_image.png"
image.save(output_path)
print(f"[INFO] Image successfully saved to: {output_path}")
