#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m'

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "${GREEN}Installing uv package manager${NC}"
    pip install uv
else
    echo "${GREEN}uv already installed, skipping${NC}"
fi

# FLOWER
echo "${GREEN}Installing FLOWER (1/2)${NC}"
echo "${GREEN}Installing calvin (1/4)${NC}"
cd iTRAP/models/flower_vla_calvin/calvin_env/tacto
uv pip install -e .
cd ..
uv pip install -e .
cd ..
echo "${GREEN}Installing pyhash (2/4)${NC}"
uv pip install cmake
uv pip install setuptools==57.5.0
cd pyhash-0.9.3
python setup.py build
python setup.py install
uv pip install --upgrade setuptools
cd ..
echo "${GREEN}Installing LIBERO (3/4)${NC}"
cd LIBERO
uv pip install -r requirements.txt
uv pip install -e .
uv pip install numpy~=1.23
cd ..
echo "${GREEN}Installing miscellaneous packages (4/4)${NC}"
uv pip install ninja
uv pip install seaborn
uv pip install optree
uv pip install natsort
uv pip install moviepy==1.0.3
uv pip install rdp
uv pip install dtw-python
uv pip install -r requirements.txt
uv pip uninstall opencv-python # clashes with opencv-python-headless
uv pip install numpy-quaternion # fix numpy version issues
uv pip install pylineclip
cd ..

# Qwen3-VL
echo "${GREEN}Installing Qwen3-VL (2/2)${NC}"
uv pip install --upgrade "transformers>4.57.0"
uv pip uninstall numpy
uv pip install numpy==1.26.4
uv pip install vllm
