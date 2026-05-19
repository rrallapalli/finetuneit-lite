#!/bin/bash
set -e

cd "$(dirname "$0")/.."

mkdir -p configs/secrets

read -p "Enter Hugging Face Token: " HF_TOKEN
read -p "Enter W&B API Key: " WANDB_KEY

cat > configs/secrets/secrets.yaml <<EOF
huggingface:
  token: $HF_TOKEN

wandb:
  api_key: $WANDB_KEY

runpod:
  api_key:
EOF

echo "Secrets configured at configs/secrets/secrets.yaml"
