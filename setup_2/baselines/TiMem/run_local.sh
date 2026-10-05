#!/usr/bin/env bash
set -euo pipefail

TIMEM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TIMEM_REPO_ROOT"

# Reuse the existing, validated local environment and Nomic download.
TIMEM_PYTHON="${TIMEM_PYTHON:-$TIMEM_REPO_ROOT/tmp/memory-v3-update-validation/bin/python}"
if [[ ! -x "$TIMEM_PYTHON" ]]; then
    echo 'Set TIMEM_PYTHON to a Python environment containing the TiMem requirements.' >&2
    exit 1
fi
export TMPDIR="$TIMEM_REPO_ROOT/setup_2/tmp/timem-local"
export HF_HOME="${HF_HOME:-$TIMEM_REPO_ROOT/tmp/locomo-v3-update-cache/huggingface}"
export XDG_CACHE_HOME="$TIMEM_REPO_ROOT/setup_2/.cache"
export TORCH_HOME="$TIMEM_REPO_ROOT/setup_2/.cache/torch"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
export GOOGLE_GENAI_USE_VERTEXAI=true
export LANGSMITH_TRACING=false LANGSMITH_TRACING_V2=false
export LANGCHAIN_TRACING=false LANGCHAIN_TRACING_V2=false
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

# Users run in parallel; months stay sequential inside each user. Arguments
# supplied by the caller override these modest per-user concurrency defaults.
exec "$TIMEM_PYTHON" -u -m setup_2.run --method timem --run v2_timem_local \
    --user-workers 5 --workers 1 --judge-concurrency 1 "$@"
