import matplotlib.pyplot as plt

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download


class img_gen:
    def __init__(self) -> None: 
        # self.small_sd()
        self.sd_xl_base()
        
    def sd_xl_base(self):
        # Initialize the Stable Diffusion model
        model_id = "stabilityai/stable-diffusion-2-1"
        
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Load model
        # self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
        # self.pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        # self.pipe.fuse_lora()

        # Ensure ddim scheduler timestep spacing set as trailing !!!
        # self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        
    def small_sd(self):
        # Initialize the Stable Diffusion model
        base_model_id = "runwayml/stable-diffusion-v1-5"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SD15-8steps-lora.safetensors"

        # Load model.
        self.pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipe.fuse_lora()
        # Use TCD scheduler to achieve better image quality
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
    
    def generate_img(self, prompt):
        # Generate the image
        # self.image = self.pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0).images[0]
        # self.image = self.pipe(prompt=prompt, num_inference_steps=8, guidance_scale=0).images[0]
        self.image = self.pipe(prompt).images[0]

        return self.image
    
    def display_img(self):
        
        imgplot = plt.imshow(self.image)
        plt.show()
        
    

# imggen = img_gen()

# imggen.generate_img("a pink panther")
# imggen.display_img()