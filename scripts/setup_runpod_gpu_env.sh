#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "FineTuneIT Light v6.3 Stable - RunPod GPU Setup"

python -m pip install --upgrade pip setuptools wheel

echo "Installing project requirements without touching Torch..."
pip install -r requirements.txt --no-deps

echo "Installing compatible ML dependencies..."
pip install \
  transformers peft trl datasets accelerate bitsandbytes \
  wandb huggingface_hub evaluate bert-score rouge-score nltk sacrebleu \
  sentencepiece tqdm

echo "Installing Unsloth without changing Torch..."
pip install unsloth --no-deps

echo "Verifying setup..."
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))

from unsloth import FastLanguageModel
print("Unsloth OK")
PY

echo "RunPod GPU environment setup complete."
