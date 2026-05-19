#!/bin/bash

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "GPU:"
nvidia-smi || true

echo ""
echo "Torch / CUDA:"
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo ""
echo "TorchAO:"
python - <<'PY'
try:
    import torchao
    print("WARNING: torchao is installed and may cause compatibility issues.")
except ModuleNotFoundError:
    print("torchao is not installed, good.")
PY

echo ""
echo "Unsloth:"
python - <<'PY'
try:
    from unsloth import FastLanguageModel
    print("Unsloth OK")
except Exception as exc:
    print("Unsloth failed:", repr(exc))
PY
