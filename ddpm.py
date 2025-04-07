import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
from PIL import Image

# from torchvision import transforms
from tqdm.auto import tqdm

# warning
logging.basicConfig(level=logging.INFO)


def load_ddpm_pipeline():
    # Load the pipeline
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f"Using device: {device}")

    image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    image_pipe.to(device)

    # Create new scheduler and set num inference steps
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    scheduler.set_timesteps(num_inference_steps=40)
    scheduler.set_timesteps(num_inference_steps=40)
    return image_pipe, scheduler, device


def generate(image_pipe, scheduler, device):
    # return
    # The random starting point
    x = torch.randn(4, 3, 256, 256).to(device)  # Batch of 4, 3-channel 256 x 256 px images

    # Loop through the sampling timesteps
    for i, t in tqdm(enumerate(scheduler.timesteps)):

        # Prepare model input
        model_input = scheduler.scale_model_input(x, t)

        # Get the prediction
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)["sample"]

        # Calculate what the updated sample should look like with the scheduler
        scheduler_output = scheduler.step(noise_pred, t, x)

        # Update x
        x = scheduler_output.prev_sample

        # Occasionally display both x and the predicted denoised images
        # if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
        #     fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        #     grid = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0)
        #     axs[0].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
        #     axs[0].set_title(f"Current x (step {i})")

        #     pred_x0 = scheduler_output.pred_original_sample  # Not available for all schedulers
        #     grid = torchvision.utils.make_grid(pred_x0, nrow=4).permute(1, 2, 0)
        #     axs[1].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
        #     axs[1].set_title(f"Predicted denoised images (step {i})")
        #     plt.show()
    
    images = []
    # save the generated image
    for i in range(x.shape[0]):
        img = x[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.save(f"generated_image_{i}.png")
        print(f"Image {i} saved as generated_image_{i}.png")
        images.append(img)
    return images


if __name__ == "__main__":
    image_pipe, scheduler, device = load_ddpm_pipeline()
    images = generate(image_pipe, scheduler, device)
