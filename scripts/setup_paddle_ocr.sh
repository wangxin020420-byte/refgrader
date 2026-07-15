#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OCR_ENV="$PROJECT_ROOT/.venv-ocr"
OCR_PYTHON="$OCR_ENV/bin/python"
REQUIREMENTS="$PROJECT_ROOT/ocr/requirements.txt"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-$HOME/.cache/refgrader/paddlex}"
PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-modelscope}"
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
export PADDLE_PDX_CACHE_HOME
export PADDLE_PDX_MODEL_SOURCE
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK

mkdir -p "$PADDLE_PDX_CACHE_HOME"

if [ ! -x "$OCR_PYTHON" ]; then
    "$PYTHON_COMMAND" -m venv "$OCR_ENV"
fi

"$OCR_PYTHON" -m pip install --upgrade pip setuptools wheel
"$OCR_PYTHON" -m pip install -r "$REQUIREMENTS"
"$OCR_PYTHON" -c \
    "import paddle,paddleocr; print('PaddlePaddle', paddle.__version__); print('PaddleOCR', paddleocr.__version__); print('CUDA', paddle.is_compiled_with_cuda())"

echo "PaddleOCR environment ready: $OCR_ENV"
echo "Authoritative PaddleX model cache: $PADDLE_PDX_CACHE_HOME"
