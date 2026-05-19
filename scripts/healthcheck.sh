#!/bin/bash

cd "$(dirname "$0")/.."

echo "FastAPI:"
curl -s http://localhost:8000/health || true

echo ""
echo "Streamlit:"
curl -I http://localhost:8501 || true

echo ""
echo "GPU:"
nvidia-smi || true
