# Installation Guide

## System Requirements

- **Operating System**: Recommended Ubuntu 18.04 or later
- **GPU**: Nvidia GPU
- **Driver Version**: Recommended version 525 or later

---

## 1. Creating a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. Conda is recommended for creating virtual environments. If Conda is already installed on your system, you can skip step 1.1.

### 1.1 Download and Install MiniConda

MiniConda is a lightweight distribution of Conda, suitable for creating and managing virtual environments. Use the following commands to download and install:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installation, initialize Conda:

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 Create a New Environment

Use the following command to create a virtual environment:

```bash
conda create -n rlgpu python=3.11
```

### 1.3 Activate the Virtual Environment

```bash
conda activate rlgpu
```

---

## 2. Installing Dependencies

### 2.1 Install PyTorch

PyTorch is a neural network computation framework used for model training and inference. Install it using the following command:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### 2.2 Install Isaac Lab (Recommended)

Isaac Lab is the successor to Isaac Gym (deprecated). It is recommended for new installations.

Navigate to the Isaac Lab source directory and install the subpackages:

```bash
cd ~/sourcecode/IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_rl
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_mimic
```

### 2.3 Install Isaac Gym (Legacy, Deprecated)

> **Note**: Isaac Gym is deprecated. Use Isaac Lab (Section 2.2) for new installations. This section is kept for reference only.

#### 2.3.1 Download

Download [Isaac Gym](https://developer.nvidia.com/isaac-gym) from Nvidia's official website.

#### 2.3.2 Install

After extracting the package, navigate to the `isaacgym/python` folder and install it using the following commands:

```bash
cd isaacgym/python
pip install -e .
```

#### 2.3.3 Verify Installation

Run the following command. If a window opens displaying 1080 balls falling, the installation was successful:

```bash
cd examples
python 1080_balls_of_solitude.py
```

If you encounter any issues, refer to the official documentation at `isaacgym/docs/index.html`.

### 2.4 Install VideoMimic RL

VideoMimic RL provides the RL algorithms.

```bash
cd videomimic_rl
pip install -e .
cd ..
```

### 2.5 Install videomimic_gym

Navigate to the directory and install it:

```bash
cd videomimic_gym
pip install -e .
cd ..
```

---

## Summary

After completing the above steps, you are ready to run the related programs in the virtual environment. If you encounter any issues, refer to the official documentation of each component or check if the dependencies are installed correctly.
