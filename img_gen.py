import torch
from stable-diffusion import StableDiffusion

# Define the text prompt
prompt = "A beautiful sunset on a beach"

# Initialize the Stable Diffusion model
model = StableDiffusion()

# Set the hyperparameters for the diffusion process
num_steps = 1000
step_size = 0.1
temperature = 0.5

# Generate the image
image = model.generate(prompt, num_steps, step_size, temperature)

# Display the generated image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()