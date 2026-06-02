#!/bin/bash

cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

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
