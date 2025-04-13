This repository implements a Flask app for a [Denoising Diffusion Probabilistic Model (DDPM)](https://arxiv.org/pdf/2006.11239).  
Given a user-uploaded selfie, the model generates celebrity faces that resemble the input image. The model is served through a backend and can be accessed from a web frontend hosted on GCP Cloud Run.  
The model comes from Hugging Face's [`diffusers`](https://github.com/huggingface/diffusers) library and is pre-trained on 30,000 celebrity faces, resized to 256×256 pixels.

## Background on DDPM with guidance

Diffusion models are a class of generative models that progressively transform random noise into data samples (such as images). The goal of diffusion models is to generate samples from a complex distribution by simulating a Markov chain of noisy steps—starting from Gaussian noise and gradually moving toward the target distribution (e.g., real images). The model is trained to predict the noise added at each step, which can be interpreted as learning the gradient of the data distribution.  
One limitation of the vanilla DDPM is its lack of control over the generated images. To address this, guidance can be added to steer the model toward a desired output.  
This implementation adds a guidance method by introducing a loss function that measures the difference between the generated image and a target image. This image loss guides the diffusion process, nudging the model toward producing images that resemble the specified target.

## Local development & testing

You can run the app locally using Docker or a Python virtual environment.

### Using docker

#### Build the image:
```bash
cd /path/to/project-root
docker build -t flask-app .
```

#### Run the container:
```bash
docker run --gpus all -p 5000:8080 -v $(pwd):/app -e PORT=8080 flask-app
```

### Using Python virtual environment

#### Set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt 
```

#### Run the Flask app:
```bash
python main.py
```


## GCP deployment

You can deploy the app to Google Cloud Run for scalable hosting. Note that GPU support is not currently enabled in Cloud Run, so generating five images takes roughly 14 minutes!

### Prerequisites

- A Google Cloud account
- The gcloud CLI installed and authenticated

### Deployment steps

#### Set your GCP region:
```bash
gcloud config set run/region europe-west1
```

#### Create a docker artifact repository:
```bash
gcloud artifacts repositories create docker-repo \
  --repository-format=docker \
  --location=europe-west1 \
  --description="Docker repository for guided DDPM"
```

#### Build and push your image:
```bash
gcloud builds submit --tag europe-west1-docker.pkg.dev/<PROJECT-ID>/docker-repo/guided-ddpm:v0
```

#### Deploy to Cloud Run:
```bash
gcloud run deploy guided-ddpm-service \
  --image=europe-west1-docker.pkg.dev/<PROJECT-ID>/docker-repo/guided-ddpm:v0 \
  --region=europe-west1 \
  --cpu=4 \
  --memory=16Gi \
  --port=8080
```

You can also deploy the app directly from the Google Cloud Console. Navigate to **Cloud Run**, click **Create Service**, and follow the prompts to upload your container image and configure the service.

## Generated images

### Unguided generation
Below are four example images generated without any guidance:

<div style="display: flex; gap: 10px;">
  <img src="generated_images/generated_image_0.png" alt="Generated Image Example 0" width="150">
  <img src="generated_images/generated_image_1.png" alt="Generated Image Example 1" width="150">
  <img src="generated_images/generated_image_2.png" alt="Generated Image Example 2" width="150">
  <img src="generated_images/generated_image_4.png" alt="Generated Image Example 3" width="150">
</div>

These samples are quite realistic and diverse.

---

### In-distribution image

<img src="generated_images/id_gen.png" alt="Generated Images - In-Distribution" width="600">

When using in-distribution target images, the model performs quite well! The generated outputs closely match the target in terms of facial pose and hairstyle, showing effective guidance within the learned data manifold. Note how all generated images appear female, consistent with the target image's characteristics.

---

### Out-of-distribution: selfie target

<img src="generated_images/selfie_gen.png" alt="Generated Images - Selfie" width="600">

Using my own selfie as the target gives results that are... let's say, humbling. I think this result makes sense: a target image significantly different from the training distribution pushes the model outside the data manifold, leading to poor generation.

---

### Out-of-distribution: non-Human targets

**Dog image**  
<img src="generated_images/dog_gen.png" alt="Generated Images - Dog" width="600">

**Cat image**  
<img src="generated_images/cat_gen.png" alt="Generated Images - Cat" width="600">

These results are… unsettling. Since the model was trained exclusively on human faces, it can't adapt when guided toward non-human targets like cats and dogs.  
