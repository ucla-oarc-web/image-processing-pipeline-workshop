#!/bin/bash

LAYER_NAME="oarc-image-process-layer"
REGION="us-west-2"

echo "Building Lambda layer with PIL..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create layer directory structure
mkdir -p layer/python

# Install dependencies into the layer
pip install -r "${SCRIPT_DIR}/../requirements.txt" -t layer/python/ --platform manylinux2014_x86_64 --implementation cp --python-version 3.12 --only-binary=:all: --upgrade

# Create zip file for the layer
cd layer
zip -r ../oarc-image-process-layer.zip .
cd ..

# Check if layer exists and update or create accordingly
if aws lambda get-layer-version-by-arn --arn "arn:aws:lambda:${REGION}:$(aws sts get-caller-identity --query Account --output text):layer:${LAYER_NAME}:1" --region ${REGION} >/dev/null 2>&1; then
    echo "Updating existing layer..."
else
    echo "Creating new layer..."
fi

# Deploy/update the layer
aws lambda publish-layer-version \
    --layer-name ${LAYER_NAME} \
    --description "PIL/Pillow for image processing" \
    --zip-file fileb://oarc-image-process-layer.zip \
    --compatible-runtimes python3.9 python3.10 python3.11 python3.12 \
    --region ${REGION}

echo "Layer operation completed!"

# Clean up
rm -rf layer oarc-image-process-layer.zip
