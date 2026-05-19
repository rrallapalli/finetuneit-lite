#!/usr/bin/env bash
set -e
source .venv/bin/activate
streamlit run app/ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
