#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "FineTuneIT Light v6.3 Stable - RunPod GPU Setup"

python -m pip install --upgrade pip setuptools wheel

echo "Checking existing PyTorch/CUDA..."
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Installing project requirements..."
pip install -r requirements.txt

echo "Installing Unsloth..."
pip install unsloth

echo "RunPod GPU environment setup complete."
