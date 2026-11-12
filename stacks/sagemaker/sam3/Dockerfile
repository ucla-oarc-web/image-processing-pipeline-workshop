# PyTorch 2.6 GPU inference image (transformers 5.x needs torch>=2.6)
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker

# Pinned so the workshop builds the same image every time.
# torch/torchvision come from the base image; don't reinstall them.
RUN pip install --no-cache-dir \
    "transformers==5.22.2" \
    "Pillow>=12.0.0,<14.0.0"

# HuggingFace pretrained model files (download weights first; see README)
COPY model/ /opt/ml/model/

# Workshop inference code
COPY code/ /opt/ml/model/code/
