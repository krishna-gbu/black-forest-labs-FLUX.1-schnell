Here’s the full detailed GitHub README including the **setup steps** to get your improved FLUX.1 FastAPI image generation API up and running — covering everything and skipping Celery/broker parts:

---

# FLUX.1 \[schnell] FastAPI Text-to-Image Generation API

This repository hosts a **FastAPI** web service wrapping the powerful **FLUX.1 \[schnell]** 12-billion parameter flow transformer model for generating high-quality images from text prompts.

---

## Features

* **Prompt Length Handling:**
  Automatically truncates long prompts using `CLIPTokenizer` to fit the 77-token limit, avoiding indexing errors.

* **Multi-Image Generation:**
  Generates **4 unique images per prompt** with varied seeds for diverse outputs.

* **Batch Download via ZIP:**
  Bundles generated images into a downloadable ZIP archive streamed to clients.

* **Memory Optimization:**
  Uses `low_cpu_mem_usage=True` and `enable_model_cpu_offload()` for efficient GPU memory use and avoiding OOM errors.

* **Adjustable Parameters:**
  Supports `guidance_scale`, `num_inference_steps`, `max_sequence_length`, and `seed` for fine-tuning output.

* **Simple Deployment:**
  Run with Uvicorn on any GPU-enabled server or cloud VM.

---

## Setup Instructions

### 1. Prerequisites

* Python 3.8 or later
* NVIDIA GPU with CUDA (optional but recommended)
* Hugging Face account with access to `black-forest-labs/FLUX.1-schnell`
* Hugging Face token (get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

---

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

---

### 3. Create and Activate Python Virtual Environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

---

### 4. Install Dependencies

```bash
pip install fastapi uvicorn torch diffusers huggingface_hub accelerate sentencepiece protobuf transformers
```

---

### 5. Set Your Hugging Face Token

You can export the token as an environment variable or set it in code (`app.py`):

```bash
export HF_TOKEN="hf_your_token_here"  # Linux/macOS
set HF_TOKEN=hf_your_token_here       # Windows
```

Or modify `app.py` directly with your token in:

```python
HF_TOKEN = "hf_your_token_here"
```

---

### 6. Run the API Server

```bash
python app.py
```

The server will start at `http://0.0.0.0:8000`.

---

### 7. Use the API

Send POST requests to `/generate-image` with JSON payload:

```json
{
  "prompt": "A cinematic portrait of a woman with sharp green eyes, photorealistic, 8K",
  "guidance_scale": 7.5,
  "num_inference_steps": 20,
  "max_sequence_length": 77,
  "num_images": 10,
  "seed": 42
}
```

You will receive a ZIP file containing 4 generated images.

---

## Code Overview

* **Prompt truncation:** Ensures max 77 tokens per prompt with `CLIPTokenizer`.
* **Image generation:** Calls the FLUX.1 pipeline 4 times with different seeds.
* **ZIP packaging:** Uses Python standard `zipfile` and `io.BytesIO`.
* **Memory management:** Uses `accelerate` and CPU offloading to optimize VRAM.

---

## Notes

* Make sure to install `accelerate` for best performance and low memory usage.
* The Hugging Face model is gated; you must authenticate before use.
* Adjust `guidance_scale` and `num_inference_steps` to tune quality vs speed.

---

## License

Licensed under Apache 2.0 in accordance with model and dependencies.

---

If you want, I can generate a markdown-ready `README.md` file or help you write example Postman collections!
