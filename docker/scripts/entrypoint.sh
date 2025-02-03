#!/bin/bash
set -e

if [ "$DOWNLOAD_PTH" = "true" ]; then
    python docker/scripts/download_model.py  --type pth
fi

if [ "$DOWNLOAD_ONNX" = "true" ]; then
    python docker/scripts/download_model.py --type onnx
fi

exec uv run python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug