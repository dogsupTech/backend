#!/bin/bash

# Ensure the builder is running
docker buildx inspect --bootstrap

# Build the Docker image with buildx specifying the platform
docker buildx build --platform linux/amd64 -t gcr.io/vetai1994/pythonserver:latest --push .

# Push the Docker image to GCR (this step can be omitted if included in the buildx command with --push)
docker push gcr.io/vetai1994/pythonserver:latest

# Deploy the image to Cloud Run
gcloud run deploy pythonserver --image gcr.io/vetai1994/pythonserver --platform managed --region europe-west2
