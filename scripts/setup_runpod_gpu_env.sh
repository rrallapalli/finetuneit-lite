#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "FineTuneIT Light v6.3 Stable - RunPod GPU Setup"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

echo "Installing project requirements..."
pip install -r requirements.txt

echo "Installing Unsloth..."
pip install unsloth

echo "Verifying setup..."
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA is not available to PyTorch.")

from unsloth import FastLanguageModel
print("Unsloth OK")
PY

echo "RunPod GPU environment setup complete."
