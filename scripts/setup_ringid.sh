#!/usr/bin/env bash
set -e  # exit on error

ENV_NAME="ringid"
PYTHON_VERSION="3.10"

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Check if environment already exists
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

echo "Installing core dependencies..."
conda install -y -n "$ENV_NAME" -c conda-forge \
    numpy=1.23.5 pandas=1.5.3 pyarrow=11.0.0 dill=0.3.6

echo "Installing PyTorch with CUDA 11.8..."
conda install -y -n "$ENV_NAME" pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

echo "Installing HuggingFace ecosystem..."
conda install -y -n "$ENV_NAME" -c conda-forge \
    transformers=4.23.1 tokenizers=0.13.3 accelerate=0.14.0 \
    huggingface_hub=0.15.1 datasets=2.14.5

echo "Installing additional utilities..."
conda install -y -n "$ENV_NAME" -c conda-forge ftfy scikit-image scikit-learn rich

echo "Installing pip-based packages (OpenCLIP, Diffusers, ipykernel)..."
conda run -n "$ENV_NAME" pip install open_clip_torch==2.20.0 diffusers==0.12.0 ipykernel


DIFFUSERS_UTILS="/usr/local/envs/${ENV_NAME}/lib/python${PYTHON_VERSION}/site-packages/diffusers/utils/dynamic_modules_utils.py"
if [[ -f "$DIFFUSERS_UTILS" ]]; then
    echo "🔧 Patching diffusers utils..."
    sed -i 's/cached_download,//' "$DIFFUSERS_UTILS"
    sed -i 's/cached_download(/hf_hub_download(/g' "$DIFFUSERS_UTILS"
fi


echo "Verifying GPU availability..."
conda run -n "$ENV_NAME" python -c "import torch; print('GPU available:', torch.cuda.is_available(), 'CUDA version:', torch.version.cuda)"

echo "Environment '${ENV_NAME}' setup completed!"

