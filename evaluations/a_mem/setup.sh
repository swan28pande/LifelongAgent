#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export HF_HOME="$ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ROOT/.cache"

uv venv --python 3.11 "$ROOT/.venv"
uv pip install --python "$ROOT/.venv/bin/python" 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"

# Cache the upstream retriever model under evaluations/a_mem/ before a paid run.
"$ROOT/.venv/bin/python" -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2")'
"$ROOT/.venv/bin/python" "$ROOT/run.py" --check
