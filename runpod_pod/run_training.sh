#!/usr/bin/env bash
set -e
CONFIG_PATH=${1:-configs/custom_jsonl.yaml}
source .venv/bin/activate
python - <<PY
from app.training.config import load_training_config
from app.training.train_lora import run_training_from_config
config = load_training_config("$CONFIG_PATH")
print(run_training_from_config(config))
PY
