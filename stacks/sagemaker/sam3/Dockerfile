FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker

# Install extra dependencies
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers.git \
    Pillow \
    torchvision

# HuggingFace pretrained model files
COPY model/ /opt/ml/model/

# Workshop inference code
COPY code/ /opt/ml/model/code/
