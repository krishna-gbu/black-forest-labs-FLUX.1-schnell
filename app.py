import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import FluxPipeline
from fastapi.responses import FileResponse, StreamingResponse
import io
from zipfile import ZipFile
from transformers import CLIPTokenizer
from huggingface_hub import login

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Hugging Face token
# Retrieve the token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running the app.")

# Login to Hugging Face CLI programmatically
login(token=HF_TOKEN)

# Clear CUDA cache before loading model
torch.cuda.empty_cache()

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load FLUX.1-schnell model WITHOUT use_auth_token param
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # Requires `accelerate` installed
)

# Enable CPU offloading to reduce GPU memory usage
pipe.enable_model_cpu_offload()
# Do NOT call pipe.to(device) when using CPU offload!

class GenerateRequest(BaseModel):
    prompt: str
    guidance_scale: float = 0.0
    num_inference_steps: int = 4
    max_sequence_length: int = 256
    seed: int = 0

@app.post("/generate-image")
async def generate_image(req: GenerateRequest):
    tokens = tokenizer.tokenize(req.prompt)
    if len(tokens) > 77:
        truncated_tokens = tokens[:77]
        prompt = tokenizer.convert_tokens_to_string(truncated_tokens)
    else:
        prompt = req.prompt

    images = []
    for i in range(4):
        generator = torch.Generator(device).manual_seed(req.seed + i)
        img = pipe(
            prompt,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            max_sequence_length=req.max_sequence_length,
            generator=generator,
        ).images[0]
        images.append(img)

    # Create a zip file in memory containing all 4 images
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        for idx, image in enumerate(images):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            zip_file.writestr(f"image_{idx+1}.png", img_bytes.read())
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=images.zip"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
