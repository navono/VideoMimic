# Real2Sim 完整环境安装指南

本文档是一份**自包含的、可顺序执行的**安装指南，已将所有已知问题和修复融入对应步骤中。新环境下按顺序执行即可完成 Real2Sim 两个 conda 环境的配置。

> 原始安装文档见 `real2sim/docs/setup.md`，本文件补充了原始文档未覆盖的兼容性修复。

## 前置条件

- Ubuntu 22.04 / 24.04
- Conda 包管理器（miniforge / miniconda 均可）
- NVIDIA GPU，已安装驱动
- 需要同时存在 CUDA 12.4+ 和 CUDA 11.8：
  ```bash
  ls /usr/local/cuda-12* /usr/local/cuda-11.8 -d
  ```
- 需要 g++-11（vm1recon 编译 CUDA 扩展用）：
  ```bash
  sudo apt update && sudo apt install -y g++-11
  ```
- 约 10GB+ 磁盘空间（模型 + 数据）

## 工作目录

所有命令默认在 `real2sim` 目录下执行：

```bash
cd /path/to/VideoMimic/real2sim
```

---

## 一、vm1rs 环境（Python 3.12, CUDA 12.4）

用于：人体预处理、MegaHunter 优化、机器人运动重定向。

### 1.1 创建 conda 环境

```bash
conda create -n vm1rs python=3.12
conda activate vm1rs
```

### 1.2 安装 PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 1.3 安装 requirements.txt

```bash
pip install viser tyro supervision transformers warp-lang timm einops scikit-learn boto3 requests pyliblzfse h5py yacs
```

安装 chumpy 和 smplx（需要 `--no-build-isolation`，因为 chumpy 的 setup.py 在隔离环境中会失败）：

```bash
cd /tmp && git clone https://github.com/hongsukchoi/chumpy.git
cd chumpy && pip install --no-build-isolation -e .
cd /tmp && rm -rf chumpy

pip install git+https://github.com/hongsukchoi/smplx
```

### 1.4 Grounded-SAM-2（人体检测和分割）

```bash
mkdir -p third_party
cd third_party/
git clone https://github.com/hongsukchoi/Grounded-SAM-2.git
cd Grounded-SAM-2
export CUDA_HOME=/usr/local/cuda-12.4   # 按实际路径调整
pip install -e .                         # Segment Anything 2
pip install --no-build-isolation -e grounding_dino  # Grounding DINO
pip install transformers
cd ../..
```

> **Troubleshooting**：如果 `pip install --no-build-isolation -e grounding_dino` 报 nvcc 错误（如 `value.type()` 找不到），按报错信息将 `value.type()` 改为 `value.scalar_type()`。

### 1.5 ViTPose（2D 姿态估计）

**关键修复**：ViTPose 依赖 mmcv 1.x API，而 `mim install mmcv==1.3.9` 在 Python 3.12 上不可用。必须使用 mmcv 1.7.2。

```bash
# 不要用 mim install mmcv==1.3.9，在 Python 3.12 上会失败
# 直接安装 mmcv 1.7.2（保留了完整的 1.x Python API）
pip install --no-build-isolation mmcv==1.7.2

cd third_party/
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
# 如果上面报错，先运行：pip install numpy cython wheel
cd ../..
```

**代码修复**：numpy 2.x 对数组赋值更严格，需要修改 ViTPose 一行代码：

```bash
# 修改 third_party/ViTPose/mmpose/models/heads/topdown_heatmap_base_head.py
# 找到这一行（约第71行）：
#   score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
# 替换为：
#   score[i] = float(np.array(img_metas[i]['bbox_score']).reshape(-1)[0])
```

用 Python 一键修改：

```bash
python -c "
import pathlib
f = pathlib.Path('third_party/ViTPose/mmpose/models/heads/topdown_heatmap_base_head.py')
txt = f.read_text()
txt = txt.replace(
    \"score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)\",
    \"score[i] = float(np.array(img_metas[i]['bbox_score']).reshape(-1)[0])\",
)
f.write_text(txt)
print('Patch applied.')
"
```

### 1.6 VIMO（3D 人体网格）

```bash
pip install git+https://github.com/hongsukchoi/VIMO.git
```

### 1.7 BSTRO（接触检测）

```bash
cd third_party/
git clone --recursive https://github.com/hongsukchoi/bstro.git
cd bstro
pip install -e .
cd ../..
```

> **注意**：bstro 的 Python 包名是 `metro`（不是 `bstro`），这是正常的。代码中 `import metro` 即可。

### 1.8 JAX + jaxls + PyRoki（MegaHunter 二阶优化 + 机器人重定向）

**关键修复**：`pip install -U "jax[cuda12]"` 会安装 JAX 0.10+，需要 cuDNN 9.8+。如果系统 cuDNN < 9.8（如 9.1），JAX 运行时会报 `dnn_support != nullptr` 错误。需要安装兼容版本：

```bash
# 先检查 cuDNN 版本
python -c "import torch; print(f'cuDNN: {torch.backends.cudnn.version()}')"
# 如果输出 < 90800（即 < 9.8），使用下面的命令：
pip install "jax[cuda12]>=0.5,<0.6"
# 如果 cuDNN >= 9.8，可以直接用最新版：
# pip install -U "jax[cuda12]"

pip install "git+https://github.com/brentyi/jaxls.git"

cd third_party/
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
# pyroki 可能更新了变量名，建议 checkout 指定版本：
git checkout 70b30a56b1e1ea83fb4c2cac8fe2c63a0624b9ce
pip install -e .
cd ../..
```

### 1.9 viser 可视化

```bash
cd third_party/
git clone https://github.com/nerfstudio-project/viser
cd viser
pip install -e .
cd ../..
```

### 1.10 验证 vm1rs 环境

```bash
python -c "
import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')
import mmcv; print(f'mmcv: {mmcv.__version__}')
from mmpose.apis import inference_top_down_pose_model, init_pose_model
import vimo; print('vimo: OK')
import metro; print('bstro(metro): OK')
import jax; print(f'jax: {jax.__version__}')
import jaxls; print('jaxls: OK')
import pyroki; print('pyroki: OK')
import sam2; print('sam2: OK')
import grounding_dino; print('grounding_dino: OK')
"
```

---

## 二、vm1recon 环境（Python 3.10, CUDA 11.8）

用于：MegaSam 重建、NKSR meshification、GeoCalib 重力校准。

### 2.1 创建 conda 环境

```bash
cd third_party/megasam-package
conda env create -f environment.yml
conda activate vm1recon
cd ../..
```

> 如果下载超时，重新运行同一命令即可，conda 会利用已下载的缓存。

### 2.2 安装 requirements.txt

```bash
pip install viser tyro supervision transformers warp-lang timm einops scikit-learn boto3 requests pyliblzfse h5py yacs
```

安装 chumpy 和 smplx：

```bash
cd /tmp && git clone https://github.com/hongsukchoi/chumpy.git
cd chumpy && pip install --no-build-isolation -e .
cd /tmp && rm -rf chumpy

pip install git+https://github.com/hongsukchoi/smplx
```

### 2.3 修复 numpy 版本

**重要**：放在 requirements.txt 安装之后，因为其他包安装时可能将 numpy 升级到 2.x，导致 PyTorch 2.0.1 的 torchvision 初始化失败：

```bash
pip install "numpy<2"
```

### 2.4 安装 torch-scatter

**关键修复**：需要 `setuptools<70`（提供 `pkg_resources`）和 `--no-build-isolation`：

```bash
pip install setuptools==69.5.1
export CUDA_HOME=/usr/local/cuda-11.8
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install --no-build-isolation torch-scatter==2.1.2
```

### 2.5 安装 xformers

```bash
cd third_party/megasam-package
wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
conda install -n vm1recon xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
rm xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
cd ../..
```

### 2.6 编译 DROID-SLAM

```bash
cd third_party/megasam-package/base
export CUDA_HOME=/usr/local/cuda-11.8
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
python setup.py install
cd ../../..
```

### 2.7 安装 NKSR 依赖

```bash
conda install -y -c pyg -c nvidia -c conda-forge pytorch-lightning=1.9.4 tensorboard pybind11 pyg rich pandas omegaconf
pip install randomname pykdtree plyfile flatten-dict pyntcloud
```

安装 pycg（不装 `[full]`，因为 `usd-core` 在 Python 3.10 上不可用）：

```bash
pip install python-pycg==0.5.2 --no-deps
pip install open3d
```

### 2.8 安装 NKSR

**关键修复**：官方 `pip install nksr -f https://nksr.huangjh.tech/...` 可能因 SSL 问题失败。从 GitHub 源码编译更可靠：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
cd /tmp && rm -rf nksr
git clone https://github.com/nv-tlabs/nksr.git
cd nksr/package
pip install --no-build-isolation -e .
cd /tmp && rm -rf nksr
```

安装 NKSR 剩余依赖：

```bash
pip install trimesh tyro h5py rtree
```

### 2.9 安装 GeoCalib

```bash
git clone https://github.com/hongsukchoi/GeoCalib.git third_party/GeoCalib
cd third_party/GeoCalib
pip install -e .
cd ../..
```

GeoCalib 初始化时会从 GitHub Releases 下载模型权重。如果自动下载失败（代理 SSL / 网络问题），手动下载：

```bash
mkdir -p ~/.cache/torch/hub/geocalib
curl -L -o ~/.cache/torch/hub/geocalib/pinhole.tar \
    https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar
curl -L -o ~/.cache/torch/hub/geocalib/distorted.tar \
    https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-distorted.tar
```

### 2.10 修复 wandb 版本

environment.yml 中的 wandb 0.18.7 与 protobuf 不兼容：

```bash
pip install "wandb>=0.19"
```

### 2.11 确认 numpy 版本

再次确认 numpy 没有被后续安装升级到 2.x：

```bash
pip install "numpy<2"
```

### 2.12 验证 vm1recon 环境

```bash
python -c "
import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')
import xformers; print(f'xformers: {xformers.__version__}')
import torch_scatter; print(f'torch_scatter: {torch_scatter.__version__}')
import lietorch; print('lietorch (DROID-SLAM): OK')
import nksr; print('nksr: OK')
import geocalib; print('geocalib: OK')
import open3d; print('open3d: OK')
import numpy; print(f'numpy: {numpy.__version__}')
"
```

---

## 三、网络预处理（代理环境适用）

以下步骤在非代理环境下通常不需要。如果你的网络环境通过代理访问外网，且遇到 SSL 错误，请执行。

### 3.1 Depth-Anything / dinov2 本地缓存

MegaSam 内部的 Depth-Anything 通过 `torch.hub.load('facebookresearch/dinov2', ...)` 加载 dinov2，需要访问 GitHub API。如果失败，手动克隆并使用 `--localhub`：

```bash
mkdir -p ~/.cache/torch/hub/torchhub
cd ~/.cache/torch/hub/torchhub
git clone https://github.com/facebookresearch/dinov2.git facebookresearch_dinov2_main
cd -

# 在 real2sim 目录下创建软链接（--localhub 使用相对路径 "torchhub/..."）
cd /path/to/VideoMimic/real2sim
mkdir -p torchhub
ln -s ~/.cache/torch/hub/torchhub/facebookresearch_dinov2_main torchhub/
```

运行 MegaSam 时加 `--localhub` 标志：

```bash
python stage1_reconstruction/megasam_reconstruction.py ... --localhub
```

---

## 四、Pipeline 验证

使用 `demo_data/input_images/sitting_standing` 完成全流程测试：

```bash
# Stage 0: 预处理（vm1rs）
conda activate vm1rs
bash preprocess_human.sh sitting_standing 0

# Stage 1: MegaSam 重建（vm1recon）
conda activate vm1recon
python stage1_reconstruction/megasam_reconstruction.py \
    --video-dir ./demo_data/input_images/sitting_standing/cam01 \
    --out-dir ./demo_data/input_megasam \
    --start-frame 0 --end-frame 100 --stride 1 --gsam2 --localhub

# Stage 2: MegaHunter 优化（vm1rs）
conda activate vm1rs
python stage2_optimization/megahunter_optimization.py \
    --world-env-path ./demo_data/input_megasam/megasam_reconstruction_results_sitting_standing_cam01_frame_0_100_subsample_1.h5 \
    --bbox-dir ./demo_data/input_masks/sitting_standing/cam01/json_data \
    --pose2d-dir ./demo_data/input_2d_poses/sitting_standing/cam01 \
    --smpl-dir ./demo_data/input_3d_meshes/sitting_standing/cam01 \
    --out-dir ./demo_data/output_smpl_and_points

# Stage 3: GeoCalib + NKSR meshification（vm1recon）
conda activate vm1recon
python stage3_postprocessing/postprocessing_pipeline.py \
    --megahunter-path ./demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_sitting_standing_cam01_frame_0_100_subsample_1.h5 \
    --out-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_sitting_standing_cam01_frame_0_100_subsample_1 \
    --is-megasam

# Stage 4: 机器人运动重定向（vm1rs）
conda activate vm1rs
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_sitting_standing_cam01_frame_0_100_subsample_1 \
    --contact-dir ./demo_data/input_contacts/sitting_standing/cam01
```

### 预期输出

```
demo_data/output_calib_mesh/.../sitting_standing.../
├── gravity_calibrated_keypoints.h5
├── gravity_calibrated_megahunter.h5
├── background_mesh.obj
├── background_less_filtered_colored_pointcloud.ply
├── background_more_filtered_colored_pointcloud.ply
└── retarget_poses_g1.h5
```

---

## 五、修复清单速查

| # | 环境 | 问题 | 修复 | 影响的阶段 |
|---|------|------|------|-----------|
| 1 | vm1rs | mmcv 2.x 移除了 `mmcv.parallel` 等 ViTPose 依赖的 API | `pip install --no-build-isolation mmcv==1.7.2` | ViTPose |
| 2 | vm1rs | ViTPose `bbox_score` numpy 2.x 赋值报错 | 修改 `topdown_heatmap_base_head.py` 第71行 | ViTPose |
| 3 | vm1rs | JAX 0.10+ 需要 cuDNN 9.8+，系统为 9.1 | `pip install "jax[cuda12]>=0.5,<0.6"` | MegaHunter / Retargeting |
| 4 | vm1rs | bstro `setup.py build develop` 后 import 失败 | `pip install -e third_party/bstro` | BSTRO |
| 5 | vm1recon | numpy 2.x 与 PyTorch 2.0.1 不兼容 | `pip install "numpy<2"` | 全部 |
| 6 | vm1recon | torch-scatter 编译需 pkg_resources | `pip install setuptools==69.5.1` + `--no-build-isolation` | MegaSam |
| 7 | vm1recon | NKSR 官方 whl 站点 SSL 不可达 | 从 GitHub 源码编译 `nv-tlabs/nksr` 的 `package/` 子目录 | NKSR |
| 8 | vm1recon | pycg `[full]` 的 `usd-core` 不可用 | `pip install python-pycg==0.5.2 --no-deps` + `pip install open3d` | NKSR |
| 9 | vm1recon | wandb 0.18.7 与 protobuf 不兼容 | `pip install "wandb>=0.19"` | MegaSam |
| 10 | vm1recon | chumpy setup.py 在隔离构建中失败 | `pip install --no-build-isolation -e .` | chumpy |
| 11 | 网络 | torch.hub 通过代理访问 GitHub API 失败 | 手动克隆 dinov2 到缓存 + `--localhub` 标志 | MegaSam |
| 12 | 网络 | GeoCalib 模型下载 SSL 失败 | `curl -L` 手动下载到 `~/.cache/torch/hub/geocalib/` | GeoCalib |
