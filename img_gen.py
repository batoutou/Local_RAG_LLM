import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download

# Define the text prompt
prompt = "a group of ducks dancing on top of the eiffel tower"

# Initialize the Stable Diffusion model
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
# Take 2-steps lora as an example
ckpt_name = "Hyper-SDXL-1step-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()

# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Generate the image
image=pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0).images[0]

# Display the generated image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()