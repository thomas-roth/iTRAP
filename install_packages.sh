#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m'

# FLOWER
echo "${GREEN}Installing FLOWER (1/2)${NC}"
echo "${GREEN}Installing calvin (1/4)${NC}"
cd iTRAP/models/flower_vla_calvin/calvin_env/tacto
pip install -e .
cd ..
pip install -e .
cd ..
echo "${GREEN}Installing pyhash (2/4)${NC}"
pip install cmake
pip install setuptools==57.5.0
cd pyhash-0.9.3
python setup.py build
python setup.py install
cd ..
echo "${GREEN}Installing LIBERO (3/4)${NC}"
cd LIBERO
pip install -r requirements.txt
pip install -e .
pip install numpy~=1.23
cd ..
echo "${GREEN}Installing miscellaneous packages (4/4)${NC}"
pip install ninja
pip install seaborn
pip install optree
pip install natsort
pip install moviepy==1.0.3
pip install rdp
pip install dtw-python
pip install -r requirements.txt
pip uninstall -y opencv-python # clashes with opencv-python-headless
cd ..

# Qwen2.5-VL
echo "${GREEN}Installing Qwen2.5-VL (2/2)${NC}"
pip install vllm
