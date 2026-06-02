#!/usr/bin/env bash
set -e

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install unsloth_zoo
pip install unsloth

echo "RunPod pod environment setup completed."
