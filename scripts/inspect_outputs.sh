#!/bin/bash

cd "$(dirname "$0")/.."

echo "Output folders:"
find outputs -maxdepth 2 -type d 2>/dev/null || true

echo ""
echo "Adapter files:"
find outputs -type f \( -name "adapter_model.safetensors" -o -name "adapter_config.json" -o -name "*.safetensors" \) 2>/dev/null || true

echo ""
echo "Recent files:"
find outputs -type f 2>/dev/null | tail -50 || true
