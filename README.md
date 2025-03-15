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
conda create -n itrap python=3.9
conda activate itrap
sh install_packages.sh
```

---

## Usage

### Training

### Evaluation
To evaluate the model, execute the evaluation script:
```bash
python iTRAP/evaluation/itrap_evaluate.py
```
