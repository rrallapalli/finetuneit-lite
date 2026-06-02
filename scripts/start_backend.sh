#!/bin/bash
set -e

cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH=.

uvicorn app.api.main:app --host 0.0.0.0 --port 8000
