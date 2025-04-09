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
    # Pipeline will load the task class based on the model
    image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    image_pipe.to(device)

    # Create new scheduler and set num inference steps
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    scheduler.set_timesteps(num_inference_steps=40)
    return image_pipe, scheduler, device


def generate(image_pipe, scheduler, device):
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

    images = []
    # save the generated image
    for i in range(x.shape[0]):
        img = x[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5) * 255
        img = Image.fromarray(img.astype(np.uint8))
        images.append(img)
    return images


def image_loss(images, target_image):
    # Calculates mean absolute difference between the image pixels and the target color
    # target shape: [1, 3, 256, 256]
    target = torch.tensor(target_image).to(images.device)
    error = torch.abs(images - target).mean()
    return error


def guide(image_pipe, scheduler, device, target_image, guidance_loss_scale=50.0):
    """
    Generates images by iteratively refining random noise using a diffusion model
    and a guidance mechanism based on a target image.
    """
    # start with random noise
    x = torch.randn(5, 3, 256, 256).to(device)
    target_image = image_transform(target_image)

    for i, t in tqdm(enumerate(scheduler.timesteps)):

        # Prepare the model input
        model_input = scheduler.scale_model_input(x, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)["sample"]

        # Set x.requires_grad to True
        x = x.detach().requires_grad_()

        # Get the predicted x0
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample

        # Calculate loss
        loss = image_loss(x0, target_image) * guidance_loss_scale
        if i % 10 == 0:
            print(i, "loss:", loss.item())

        # Get gradient
        cond_grad = -torch.autograd.grad(loss, x)[0]

        # Modify x based on this gradient, detach to stop gradient tracking
        x = x.detach() + cond_grad

        # Now step with scheduler
        x = scheduler.step(noise_pred, t, x).prev_sample

    images = []
    # save the generated image
    for i in range(x.shape[0]):
        img = x[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5) * 255
        img = Image.fromarray(img.astype(np.uint8))
        images.append(img)
    return images


def image_transform(pl_img):
    # Pillow loads images in RGB
    img = pl_img.resize((256, 256))
    img = (np.array(img) / 255.0 - 0.5) / 0.5
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return img


if __name__ == "__main__":
    image_pipe, scheduler, device = load_ddpm_pipeline()
    images = generate(image_pipe, scheduler, device)
    # target_image = Image.open("pretty_woman.png")
    # images = guide(image_pipe, scheduler, device, target_image)
    # save images
    for i, img in enumerate(images):
        img.save(f"generated_image_{i}.png")
