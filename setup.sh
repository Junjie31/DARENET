#!/bin/bash

# Check if a mode argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {conda|uv}"
    exit 1
fi

MODE=$1

conda_setup() {
    echo "Setting up using conda..."
    conda remove --name DareNet --all -y || exit 1
    conda create -n DareNet python=3.10 -y || exit 1
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate DareNet || exit 1
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt || exit 1
}

uv_setup() {
    echo "Setting up using uv..."
    rm -rf .venv || exit 1
    uv venv --python 3.10 || exit 1
    source .venv/bin/activate || exit 1
    uv pip install setuptools wheel || exit 1
    uv pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121 || exit 1
    uv pip install -r requirements.txt || exit 1
}

case $MODE in
    conda)
        conda_setup
        ;;
    uv)
        uv_setup
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 {conda|uv}"
        exit 1
        ;;
esac
