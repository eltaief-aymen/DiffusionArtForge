
"""
DiffusionArtForge – AI Image Generation with Stable Diffusion
Author: Eltaief Aymen
Date: 2025-08-31

This script allows you to generate AI images from text prompts using Hugging Face Diffusers.
It supports negative prompts, adjustable inference steps, guidance scale, and deterministic seeds.
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

def load_pipeline(device: str):
    """
    Load Stable Diffusion pipeline with optimized settings.
    """
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Replace default scheduler with Euler Ancestral for sharper results
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Move model to device
    pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    return pipe

def generate_image(pipe, prompt, negative_prompt, steps, guidance, height, width, seed, output):
    """
    Generate an image from text using Stable Diffusion.
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=generator
    ).images[0]
    
    image.save(output)
    print(f"[✔] Image saved to {output}")

def main():
    parser = argparse.ArgumentParser(description="DiffusionArtForge – Stable Diffusion Image Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the image")
    parser.add_argument("--negative", type=str, default="", help="Negative prompt to avoid artifacts")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (higher = stricter to prompt)")
    parser.add_argument("--height", type=int, default=512, help="Image height (must be multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Image width (must be multiple of 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="output.png", help="Output file name")
    args = parser.parse_args()
    
    # Select best device available
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    pipe = load_pipeline(device)
    generate_image(pipe, args.prompt, args.negative, args.steps, args.guidance, args.height, args.width, args.seed, args.output)

if __name__ == "__main__":
    main()
