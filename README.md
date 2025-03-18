# Evaluating Image-Space Trajectory Planning for Vision-Language-Guided Robot Manipulation

This repository contains the code and benchmarks for my Bachelor's thesis titled **"Evaluating Image-Space Trajectory Planning for Vision-Language-Guided Robot Manipulation"**

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup and Installation](#setup-and-installation)
3. [Usage](#usage)

---

## Project Structure

---

## Setup and Installation

### Prerequisites


### Installation Steps
```bash
git clone --recurse-submodules git@github.com:thomas-roth/iTRAP.git
cd iTRAP
conda create -n itrap python=3.9 -y
conda activate itrap
sh install_packages.sh
```

---

## Usage

### Training

### Evaluation
To evaluate the model, two terminal sessions need to be started.
Start the VLM server in the first terminal:
```bash
conda activate itrap
sh iTRAP/models/Qwen2-VL/start_vlm_server.sh
```
Start the evaluation script in the second terminal:
```bash
conda activate itrap
python iTRAP/evaluation/itrap_evaluate.py
```
