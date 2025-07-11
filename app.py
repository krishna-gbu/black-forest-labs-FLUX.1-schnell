import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from diffusers import FluxPipeline
from fastapi.responses import FileResponse, StreamingResponse
import io
from zipfile import ZipFile
from transformers import CLIPTokenizer
from huggingface_hub import login

# Optional: Suppress specific warnings from transformers/diffusers
# You can set the logging level to ERROR or CRITICAL if you don't want to see warnings
# import logging
# logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
# logging.getLogger("diffusers.pipelines.flux.pipeline_flux").setLevel(logging.ERROR)
# logging.getLogger("huggingface_hub.utils._deprecation").setLevel(logging.ERROR) # Example for deprecation warnings

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running the app.")

login(token=HF_TOKEN)

torch.cuda.empty_cache()

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

pipe.enable_model_cpu_offload()

class GenerateRequest(BaseModel):
    prompt: str
    guidance_scale: float = 0.0
    num_inference_steps: int = 4
    max_sequence_length: int = Field(default=77, ge=1, le=77) # Adjusted default and max to 77
    seed: int = 0
    num_images: int = Field(default=1, ge=1, le=10)

@app.post("/generate-image")
async def generate_image(req: GenerateRequest):
    # Your current truncation logic is good for explicit control
    tokens = tokenizer.tokenize(req.prompt)
    if len(tokens) > 77:
        truncated_tokens = tokens[:77]
        # It's better to convert back to string if you want the truncated text
        # to be exactly what CLIP receives.
        prompt = tokenizer.convert_tokens_to_string(truncated_tokens)
        print(f"Warning: Prompt truncated to 77 tokens. Original length: {len(tokens)}. Truncated prompt: '{prompt}'")
    else:
        prompt = req.prompt
        
    # If the original prompt contained characters not in CLIP's vocabulary,
    # the 'prompt' variable might still contain <unk> after conversion.
    # You might want to log this for debugging if image quality is poor.
    if "<unk>" in prompt:
        print(f"Warning: Prompt contains unknown tokens (<unk>) after tokenization: '{prompt}'")


    images = []
    for i in range(req.num_images):
        generator = torch.Generator(device).manual_seed(req.seed + i)
        img = pipe(
            prompt,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            max_sequence_length=req.max_sequence_length, # Now this is consistently 77
            generator=generator,
        ).images[0]
        images.append(img)

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
