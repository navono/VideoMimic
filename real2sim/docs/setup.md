# Setup Guide

This guide walks you through setting up the VideoMimic Real-to-Sim pipeline for transforming human motion videos into robot-ready motion data.

## Overview

The pipeline requires two separate conda environments due to dependency conflicts:

| Environment | Name | Python | CUDA | Purpose |
|------------|------|--------|------|---------|
| **Main** | `vm1rs` | 3.12 | 12.4+ | Human preprocessing, optimization, retargeting |
| **Reconstruction** | `vm1recon` | 3.10 | 11.8 | MegaSam, NKSR meshification, GeoCalib |

> **Why two environments?** MegaSam requires xformers ≤0.0.27 (due to deprecated NyquistAttention) which only compiles with CUDA 11.8. NKSR is also tied to CUDA 11.8.

## Test Environment / Prerequisites

- Ubuntu 24.04
- Conda package manager
- ~10GB free disk space for models and data
- NVIDIA GPU with CUDA support (A5000, A6000, A6000 ADA, A100 40GB, A100 80GB)

## Installation

### 1. Main Environment (`vm1rs`)

Create and activate the main environment:

```bash
conda create -n vm1rs python=3.12
conda activate vm1rs
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

#### Human Detection & Pose Estimation

```bash
mkdir third_party

# 1. Grounded-SAM-2 (bounding boxes and segmentation)
cd third_party/
git clone https://github.com/hongsukchoi/Grounded-SAM-2.git
cd Grounded-SAM-2
export CUDA_HOME=/usr/local/cuda-12.4  # Adjust to your CUDA version
pip install -e .                        # Segment Anything 2
pip install --no-build-isolation -e grounding_dino  # Grounding DINO
pip install transformers
cd ../..

# 2. ViTPose (2D pose estimation)
pip install -U openmim
pip install --upgrade setuptools
mim install mmcv==1.3.9  # If error, try: pip install setuptools --upgrade
cd third_party/
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
# If error above, do `pip install numpy cython wheel` first
cd ../..

# 3. VIMO (3D human mesh - primary method)
pip install git+https://github.com/hongsukchoi/VIMO.git

# 4D Humans (deprecated)
# pip install git+https://github.com/hongsukchoi/4D-Humans.git

# 4. BSTRO (contact detection)
cd third_party/
git clone --recursive https://github.com/hongsukchoi/bstro.git
cd bstro
python setup.py build develop
cd ../..
```

<details>
<summary>Troubleshooting: cuda-12.6 and grounding dino installation  </summary>

When you see error for `pip install --no-build-isolation -e grounding_dino`, something like `error: command '/usr/local/cuda-12.6/bin/nvcc' failed with exit code 2`, see the error message and change `value.type()` to ` value.scalar_type()`.

</details>

<details>
<summary>Troubleshooting: g++-11 errors</summary>

If you encounter g++-11 related errors:

```bash
# Install g++-11
sudo apt update
sudo apt install g++-11

# Set environment variables
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# Retry the installation
pip install --no-build-isolation -e grounding_dino
```
</details>

#### MegaHunter + PyRoki 

```bash
# Second order optimization for MegaHunter and PyRoki
pip install -U "jax[cuda12]"
pip install "git+https://github.com/brentyi/jaxls.git"
# PyRoki for robot motion retargeting
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
# pyroki might have updated some variable names; git checkout 70b30a56b1e1ea83fb4c2cac8fe2c63a0624b9ce 
pip install -e .
cd ../..
```

#### Core Dependencies

```bash
# PyTorch (avoid 2.6 - it's unstable)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Viser for visualization
cd third_party/
git clone https://github.com/nerfstudio-project/viser
cd viser
pip install -e .
cd ../..
```


#### Optional: World Reconstruction (Align3r)

```bash
# Monst3r/Align3r (skip if only using MegaSam)
cd third_party/
git clone https://github.com/Junyi42/monst3r-depth-package.git
cd monst3r-depth-package
pip install -e .
cd ../..
pip install git+https://github.com/Junyi42/croco_package.git
```

#### Optional: Neural Meshification (NDC)

```bash
# NDC (skip if only using NKSR)
pip install trimesh h5py cython opencv-python
cd third_party/NDC
python setup.py build_ext --inplace
cd ../..
```

#### Optional: Hand Pose Estimation

```bash
# WiLor (3D hand mesh)
pip install git+https://github.com/warmshao/WiLoR-mini
```

### 2. Reconstruction Environment (`vm1recon`)

This environment handles MegaSam reconstruction, NKSR meshification, and GeoCalib operations.

```bash
cd third_party/
git clone --recursive https://github.com/Junyi42/megasam-package
cd megasam-package

# Create environment from yaml
conda env create -f environment.yml

conda activate vm1recon

# other dependencies
cd ../..
pip install -r requirements.txt
cd third_party/megasam-package

# Additional dependencies
# cuda 11.8 is required
export CUDA_HOME=/usr/local/cuda-11.8

# Install g++-11 if not already installed
# sudo apt update
# sudo apt install g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install torch-scatter==2.1.2

# Install specific xformers version (required for MegaSam)
wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
conda install xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
rm xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2

# Compile DROID-SLAM components
cd base

python setup.py install
cd ../..

# NKSR for fast meshification
conda install -c pyg -c nvidia -c conda-forge pytorch-lightning=1.9.4 tensorboard pybind11 pyg rich pandas omegaconf
pip install -f https://pycg.huangjh.tech/packages/index.html python-pycg[full]==0.5.2 randomname pykdtree plyfile flatten-dict pyntcloud
pip install trimesh tyro h5py rtree

# Install NKSR from source. The nksr wheel index can be unavailable or can
# resolve to an empty 0.0.0 package without nksr.Reconstructor.
git clone --depth 1 https://github.com/nv-tlabs/NKSR.git /tmp/NKSR
cd /tmp/NKSR/package
git clone https://github.com/AcademySoftwareFoundation/openvdb.git external/openvdb
git -C external/openvdb checkout 7edd8cd86f105a01a41ad7b2bf59a81034cd79fb
git clone https://gitlab.com/libeigen/eigen.git external/eigen
git -C external/eigen checkout 3.4
python -m pip install --no-build-isolation /tmp/NKSR/package
python -c "import torch, nksr; r=nksr.Reconstructor(torch.device('cuda')); print(nksr.__version__, type(r).__name__)"
cd ..

# GeoCalib for gravity calibration
git clone https://github.com/hongsukchoi/GeoCalib.git third_party/GeoCalib
cd third_party/GeoCalib
pip install -e .
cd ../..
```

## Environment Quick Reference

Always activate the correct environment before running commands:

```bash
# Most operations
conda activate vm1rs

# For MegaSam reconstruction and postprocessing
conda activate vm1recon
```

See [commands.md](./commands.md) for detailed usage instructions.

## Benchverse HTTP Service on 5051

Benchverse calls this real2sim pipeline through the HTTP service in
`/home/ubuntu22/sourcecode/VideoMimic/real2sim/server/server.py`.

On the 5051 host:

```bash
cd /home/ubuntu22/sourcecode/VideoMimic/real2sim
make serve
curl http://127.0.0.1:8090/api/health
```

The Benchverse skill server at `:5052` can also manage this service:

```bash
curl http://127.0.0.1:5052/api/services/real2sim/status
curl -X POST http://127.0.0.1:5052/api/services/real2sim/restart
```

The HTTP server must pass the uploaded job video to the Makefile with
`VIDEO_PATH`, not `VIDEO_NAME`:

```python
make_cmd = (
    f'{CONDA_EVAL} && '
    f'export HF_TOKEN=${{HF_TOKEN:-}} && '
    f'make pipeline VIDEO_PATH="{video_src}" STRIDE={stride} HEIGHT={height_value} '
    f'ROBOT={robot} GENDER={gender} PROXY="{PROXY_URL}"'
)
```

On the 5051 deployment, the default stride should be `4` for 16GB GPUs:

```makefile
STRIDE ?= 4
```

```python
stride: int = Form(4)
```

`VIDEO_NAME` does not override the Makefile input video. If it is used, the
pipeline can silently fall back to the default `assets/sitting_standing.mp4`
and write outputs under `demo_data/sitting_standing`. Uploaded files should be
stored under the sanitized `video_stem + suffix` so the Makefile's
`VID_STEM := $(basename $(notdir $(VIDEO_PATH)))` matches the server result
collection path.

If postprocessing fails in `meshification.py` with
`TypeError: 'NoneType' object is not callable` at `np.sum(...)`, avoid
`np.sum` in the weighted depth interpolation and use `np.dot` with finite
neighbor guards. This has been verified on the `talented-urban-woman-dancing`
job with `subsample_4`.

Real2Sim output is an h5 file such as `retarget_poses_g1.h5`. Before RL
training, Benchverse skill server converts it in unitree_rl_lab:

```bash
cd /home/ubuntu22/sourcecode/unitree_rl_lab
python scripts/mimic/convert_videomimic_to_npz.py \
  -i "<job>/input/retarget_poses_g1.h5" \
  -o "<job>/motion.npz" \
  --target_fps 50
```

The equivalent Makefile target in unitree_rl_lab is:

```bash
make convert-videomimic \
  VM_H5="<job>/input/retarget_poses_g1.h5" \
  CONVERT_NPZ="<job>/motion.npz" \
  CONVERT_FPS=50
```

Current known-good checks on 5051:

```bash
/home/ubuntu22/miniforge3/envs/vm1recon/bin/python \
  -c "import torch, nksr; r=nksr.Reconstructor(torch.device('cuda')); print(nksr.__version__, type(r).__name__)"

/usr/bin/env -C /home/ubuntu22/sourcecode/VideoMimic/real2sim \
  /home/ubuntu22/miniforge3/envs/vm1recon/bin/python \
  stage3_postprocessing/postprocessing_pipeline.py \
  --megahunter-path demo_data/sitting_standing/output_smpl_and_points/megahunter_megasam_reconstruction_results_input_images_cam01_frame_1_111_subsample_2.h5 \
  --out-dir /tmp/nksr_postprocess_test \
  --gender male \
  --is-megasam
```

If a long uploaded video now fails with CUDA OOM during reconstruction, that is
separate from the `VIDEO_PATH` and NKSR fixes. Shorten the clip, increase
`STRIDE`, or free GPU memory.

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   - Ensure CUDA 12.4+ is available for `vm1rs`
   - Ensure CUDA 11.8 is available for `vm1recon`

2. **Memory Errors**
   - MegaSam: Requires ~24GB+ GPU memory for 300 frames
   - Align3r: Requires ~80GB+ GPU memory for 150 frames
   - Reduce `--end-frame` or use `--stride` to process fewer frames

3. **Import Errors**
   - Verify you're in the correct conda environment
   - Check that all installation steps completed without errors

4. **`AttributeError: module 'nksr' has no attribute 'Reconstructor'`**
   - This usually means the empty PyPI `nksr==0.0.0` package was installed.
   - Reinstall NKSR from `/tmp/NKSR/package` with `--no-build-isolation` as shown above.

### Getting Help

- Check existing issues on GitHub
- Include error messages and system info (GPU, CUDA, etc.) when reporting issues
