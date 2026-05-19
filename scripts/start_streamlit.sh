#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.

streamlit run app/ui/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
