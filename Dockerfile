# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy only requirements first to leverage Docker's caching
# COPY requirements.txt ./

# Install larger dependencies first to optimize caching
RUN pip install torch==2.6.0 torchvision==0.21.0 diffusers==0.32.2 accelerate==1.6.0

# Install the remaining dependencies
RUN pip install matplotlib==3.9.4 numpy==2.0.2 Flask==3.1.0 pillow==11.1.0 gunicorn pyheif

# Copy the rest of the application code
COPY . ./

# Command to run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app