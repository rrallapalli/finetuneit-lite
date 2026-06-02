#!/bin/bash
set -e

cd "$(dirname "$0")/.."

mkdir -p configs/secrets

echo ""
echo "========================================="
echo " FineTuneIT Lite - Token Configuration"
echo "========================================="
echo ""
echo "Tokens are stored locally in:"
echo "  - .env"
echo "  - configs/secrets/secrets.yaml"
echo ""
echo "They are ignored by git."
echo ""

read -s -p "Enter Hugging Face Token: " HF_TOKEN
echo ""
read -s -p "Enter W&B API Key (optional): " WANDB_API_KEY
echo ""
read -s -p "Enter RunPod API Key (optional): " RUNPOD_API_KEY
echo ""

cat > .env <<EOF_ENV
HF_TOKEN=${HF_TOKEN}
HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
WANDB_API_KEY=${WANDB_API_KEY}
RUNPOD_API_KEY=${RUNPOD_API_KEY}
EOF_ENV

cat > configs/secrets/secrets.yaml <<EOF_YAML
huggingface:
  token: ${HF_TOKEN}

wandb:
  api_key: ${WANDB_API_KEY}

runpod:
  api_key: ${RUNPOD_API_KEY}
EOF_YAML

echo ""
echo "Saved tokens to .env and configs/secrets/secrets.yaml"

if [ -n "$HF_TOKEN" ] && command -v huggingface-cli >/dev/null 2>&1; then
    echo "Logging in to Hugging Face CLI..."
    huggingface-cli login --token "$HF_TOKEN" || echo "Hugging Face CLI login skipped/failed. The token is still saved."
fi

if [ -n "$WANDB_API_KEY" ] && command -v wandb >/dev/null 2>&1; then
    echo "Logging in to Weights & Biases..."
    wandb login "$WANDB_API_KEY" || echo "W&B login skipped/failed. The token is still saved."
fi

echo ""
echo "Token setup complete."
echo ""
