# Evaluating Image-Space Trajectory Planning for Vision-Language-Guided Robot Manipulation

This repository contains the code and benchmarks for my Bachelor's thesis titled **"Evaluating Image-Space Trajectory Planning for Vision-Language-Guided Robot Manipulation"**.


## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup and Installation](#setup-and-installation)
3. [Usage](#usage)


## Project Structure
```bash
📦iTRAP
 ┣ 📂iTRAP                      # (for packaging)
 ┃ ┣ 📂datasets                 # Dataset downloaders & dataset builders for VLM & Policy
 ┃ ┃ ┗ 📂calvin                 # Dataset downloader & dataset builders for CALVIN dataset
 ┃ ┣ 📂evaluation               # Evaluation scripts for iTRAP & VLM
 ┃ ┗ 📂models                   # VLM & Policy of iTRAP
 ┃ ┃ ┣ 📂MoDE_Diffusion_Policy  # Policy of iTRAP
 ┃ ┃ ┃ ┣ 📂LIBERO               # LIBERO benchmark for simulation & real-world tests (not used)
 ┃ ┃ ┃ ┣ 📂calvin_env           # CALVIN simulation environment
 ┃ ┃ ┃ ┣ 📂conf                 # Config files for training & evaluation on CALVIN & LIBERO
 ┃ ┃ ┃ ┣ 📂datasets             # Policy datasets (non-existent at first)
 ┃ ┃ ┃ ┣ 📂mode                 # Model, data modules & training & evaluation scripts
 ┃ ┃ ┃ ┃ ┣ 📂callbacks          # EMA Callback for training & evaluation
 ┃ ┃ ┃ ┃ ┣ 📂datasets           # Data modules for CALVIN & LIBERO
 ┃ ┃ ┃ ┃ ┣ 📂evaluation         # Policy evaluation scripts for CALVIN & LIBERO (not used)
 ┃ ┃ ┃ ┃ ┣ 📂models             # Policy agent, CLIP nets, perceptual encoders & diffusion model
 ┃ ┃ ┃ ┃ ┣ 📂rollout            # Rollout scripts for CALVIN & LIBERO
 ┃ ┃ ┃ ┃ ┣ 📂utils              # Learning rate schedulers, data transforms, model saving, etc.
 ┃ ┃ ┃ ┃ ┗ 📂wrappers           # HULC wrapper for CALVIN
 ┃ ┃ ┃ ┣ 📂preprocess           # Preprocessing for CALVIN to optimize GPU utilization for training
 ┃ ┃ ┃ ┣ 📂pretrained           # Pretrained models (non-existent at first)
 ┃ ┃ ┃ ┗ 📂pyhash-0.9.3         # Custom pyhash package (to mitigate version conflicts)
 ┃ ┃ ┣ 📂Qwen2-VL               # VLM of iTRAP
 ┃ ┃ ┃ ┗ 📂pretrained           # Pretrained models (non-existent at first)
 ┣ 📂logs                       # Logs of training (non-existent at first)
 ┣ 📂outputs                    # Logs of evaluation (non-existent at first)
```


## Setup and Installation

### Prerequisites
Make sure Xlib libraries are installed. They can be installed by executing:
```bash
sudo apt update
sudo apt install -y libx11-dev mesa-common-dev libegl1-mesa-dev libgl1-mesa-dev
```

### Installation Steps
```bash
git clone --recurse-submodules git@github.com:thomas-roth/iTRAP.git
export ITRAP_ROOT=$(pwd)/iTRAP
cd $ITRAP_ROOT
conda create -n itrap python=3.9 -y
conda activate itrap
sh install_packages.sh
```


## Usage
Download CALVIN datsets by choosing a split (`D`, `ABC`, `ABCD`, `debug`) and executing:
```bash
cd $ITRAP_ROOT/iTRAP/datasets/calvin
sh download_data.sh <split>
```

### Training
To train the VLM on CALVIN, follow the steps below:
1. Build the VLM dataset:
    ```bash
    python $ITRAP_ROOT/iTRAP/datasets/calvin/calvin_vlm_dataset_builder.py --dataset-path $ITRAP_ROOT/iTRAP/datasets/calvin/task_<split> --output-dir $ITRAP_ROOT/iTRAP/models/Qwen2-VL/dataset/task_<split>
    ```
2. Train the VLM using LLaMA-Factory

To train the policy on CALVIN, follow the steps below:
1. Build the policy dataset:
    ```bash
    python $ITRAP_ROOT/iTRAP/datasets/calvin/calvin_policy_dataset_builder.py --dataset-path $ITRAP_ROOT/iTRAP/datasets/calvin/task_<split> --output-dir $ITRAP_ROOT/iTRAP/models/MoDE_Diffusion_Policy/dataset/task_<split>
    ```
2. Choose & download a pretrained policy from the [collection](https://huggingface.co/collections/mbreuss/mode-6760239f42bc757093b6de13):
    ```bash
    cd $ITRAP_ROOT/iTRAP/models/MoDE_Diffusion_Policy
    mkdir pretrained
    cd pretrained
    git clone <chosen_pretrained_policy>
    cd <chosen_pretrained_policy>
    mkdir .hydra
    mv config.yaml .hydra/config.yaml
    ```
3. Adjust the training parameters in the config files ([main config](iTRAP/models/MoDE_Diffusion_Policy/conf/config_calvin.yaml), [checkpoint path, optimizer & lr scheduler](iTRAP/models/MoDE_Diffusion_Policy/conf/model/mode_agent.yaml), [rollout config](iTRAP/models/MoDE_Diffusion_Policy/conf/callbacks/rollout_lh/calvin.yaml)).
4. Start the VLM server in a separate terminal (required during rollout):
    ```bash
    conda activate itrap
    sh iTRAP/models/Qwen2-VL/start_vlm_server.sh
    ```
5. Start the training:
    ```bash
    python iTRAP/models/MoDE_Diffusion_Policy/mode/training_calvin.py
    ```

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
