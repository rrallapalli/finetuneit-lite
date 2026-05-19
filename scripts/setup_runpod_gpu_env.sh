#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "FineTuneIT Light v6.3 Stable - RunPod GPU Setup"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel

echo "Installing CUDA-enabled PyTorch..."
pip uninstall -y torch torchvision torchaudio torchao || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing project requirements..."
pip install -r requirements.txt

echo "Installing fine-tuning dependencies..."
pip install --upgrade accelerate peft trl transformers datasets bitsandbytes sentencepiece protobuf scipy scikit-learn pandas wandb evaluate rouge-score bert-score huggingface_hub

echo "Installing Unsloth last..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo "Removing TorchAO if installed transitively..."
pip uninstall -y torchao || true

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

try:
    import torchao
    raise RuntimeError("torchao is still installed. Run: pip uninstall -y torchao")
except ModuleNotFoundError:
    print("TorchAO: not installed, good")

from unsloth import FastLanguageModel
print("Unsloth OK")
PY

echo "RunPod GPU environment setup complete."
