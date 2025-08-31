# DiffusionArtForge – AI Image Generation with Stable Diffusion

DiffusionArtForge is a **lightweight image generation pipeline** built with [Hugging Face Diffusers](https://github.com/huggingface/diffusers).  
It enables you to generate **high-quality AI images** from text prompts using **Stable Diffusion v1.5**, with flexible configuration for:
- 🎨 Prompt engineering
- 🛑 Negative prompts (avoid artifacts like blur, watermarks, or distortions)
- ⚡ Choice of schedulers (Euler Ancestral, DDIM, PNDM, etc.)
- 🔄 Reproducible results with seeding
- 🖼️ Adjustable resolution, steps, and guidance scale

---

## 🚀 Features
- **Text-to-Image** generation in a few lines of code.
- Support for **CUDA (NVIDIA GPUs)**, **MPS (Apple Silicon)**, and **CPU** fallback.
- **Negative prompts** for higher-quality images.
- **Deterministic output** with fixed seeds.
- Configurable **height, width, steps, and guidance scale**.

---

## 📦 Installation
Make sure you have Python 3.9+ installed, then run:

```bash
pip install diffusers transformers accelerate torch torchvision safetensors

---

## ⚙️ Usage

### 1. Clone this repo
```bash
git clone https://github.com/eltaief-aymen/DiffusionArtForge.git
cd DiffusionArtForge

---

### 2. Run the script
```bash
python generate.py --prompt "A cyberpunk cityscape at night, neon lights, cinematic" \
                   --negative "blurry, lowres, watermark, text" \
                   --steps 30 \
                   --guidance 8.0 \
                   --seed 123 \
                   --output cyberpunk.png
