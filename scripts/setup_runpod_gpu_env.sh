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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Installing project requirements..."
pip install -r requirements.txt

echo "Installing Unsloth without changing Torch..."
pip install unsloth_zoo --no-deps
pip install unsloth --no-deps

echo "Verifying setup..."
python - <<'PY'
import torch
from unsloth import FastLanguageModel

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

print("Unsloth OK")
PY

echo "RunPod GPU environment setup complete."
