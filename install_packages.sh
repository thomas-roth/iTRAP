#!/bin/bash

set -euo pipefail

GREEN=$'\033[0;32m'
NC=$'\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_VERSION="3.11.14"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "${GREEN}Installing uv package manager${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "${GREEN}uv already installed, skipping${NC}"
fi

# Create the uv virtual environment if it does not exist
if [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
    echo "${GREEN}Creating uv venv with Python ${PYTHON_VERSION}${NC}"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
cd "$SCRIPT_DIR"

# All third-party dependencies are declared in pyproject.toml
echo "${GREEN}Installing dependencies from pyproject.toml${NC}"
uv sync

# Local packages (their dependencies come from pyproject.toml)
echo "${GREEN}Installing local packages${NC}"
uv pip install --no-deps -e iTRAP/models/flower_vla_calvin/calvin_env/tacto
uv pip install --no-deps -e iTRAP/models/flower_vla_calvin/calvin_env
uv pip install --no-deps -e iTRAP/models/flower_vla_calvin/LIBERO
uv pip install --no-deps -e iTRAP/models/flower_vla_calvin
