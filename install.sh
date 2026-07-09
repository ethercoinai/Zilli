#!/usr/bin/env bash
set -euo pipefail

# Zilli — one-click development setup
# Usage: bash install.sh  (or chmod +x install.sh && ./install.sh)

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { printf "${GREEN}[✓]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[!]${NC} %s\n" "$*"; }
error() { printf "${RED}[✗]${NC} %s\n" "$*"; }

cd "$(dirname "$0")"

# --- Python ---
PYTHON=""
for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    error "Python 3.10+ is required"
    exit 1
fi

PYVER=$("$PYTHON" --version 2>&1 | grep -oP '\d+\.\d+')
info "Python $PYVER ($PYTHON)"

# --- uv ---
if command -v uv &>/dev/null; then
    INSTALLER="uv"
    info "Package manager: uv"
elif command -v pip &>/dev/null; then
    INSTALLER="pip"
    warn "Using pip (uv recommended: curl -LsSf https://astral.sh/uv/install.sh | sh)"
else
    error "No Python package manager found (pip or uv)"
    exit 1
fi

# --- Virtual env ---
if [ ! -d .venv ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate

# --- Dependencies ---
info "Installing Python dependencies..."
if [ "$INSTALLER" = "uv" ]; then
    uv pip install --quiet --upgrade pip setuptools wheel
    uv pip install --quiet -r requirements.txt 2>/dev/null || \
        uv pip install --quiet -e ".[dev,celery,chroma]"
else
    pip install --quiet --upgrade pip setuptools wheel
    pip install --quiet -r requirements.txt 2>/dev/null || \
        pip install --quiet -e ".[dev,celery,chroma]"
fi

# --- Optional middleware ---
MIDDLEWARE=false
if command -v docker &>/dev/null; then
    MIDDLEWARE=true
    info "Docker found — checking Redis + Chroma"

    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'zilli-redis'; then
        warn "Redis container not running. Start with:"
        warn "  docker run -d --name zilli-redis -p 6379:6379 redis:7-alpine"
    fi

    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'zilli-chroma'; then
        warn "Chroma container not running. Start with:"
        warn "  docker run -d --name zilli-chroma -p 8000:8000 chromadb/chroma"
    fi
fi

# --- Setup directories ---
mkdir -p audit_logs state
info "Setup complete"

# --- Summary ---
echo ""
printf "${BOLD}Zilli development environment ready${NC}\n"
echo "  Activate:  source .venv/bin/activate"
echo "  Test:      python -m pytest tests/"
if [ "$INSTALLER" = "uv" ]; then
    echo "  Dashboard: streamlit run zilli/dashboard_app.py"
    echo "  Celery:    celery -A zilli.workflow.celery_app worker --loglevel=info"
fi
echo "  Config:    cp zilli/config.example.yaml config.yaml  # then edit"
echo "  State:     cat state/STATE.md"
