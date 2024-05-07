import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class img_gen:
    def __init__(self) -> None: 
        self.sd_model()
        
    def sd_model(self):
        # Initialize the Stable Diffusion model
        model_id = "stabilityai/stable-diffusion-2-1"
        # load teh pretained weights to the GPU
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        # intanciate a scheduler to improve efficiency 
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
    
    def generate_img(self, prompt):
        # Generate the image
        self.image = self.pipe(prompt=prompt, num_inference_steps=32).images[0]

        return self.image