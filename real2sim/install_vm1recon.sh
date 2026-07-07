#!/usr/bin/env bash
# vm1recon 环境安装脚本（按 real2sim/docs/setup.md）。
# 遇错即停，全量记日志；每步写 step.done 标记便于定位/续装。
set -eo pipefail
# 注意:不用 -u。conda 的 activate 钩子（libblas_mkl_activate.sh 等）会引用未绑定的
# MKL_INTERFACE_LAYER 等环境变量，set -u 会让 conda install/activate 直接退出。

# HTTP 代理（pip/conda 子进程继承；VM1RECON_PROXY 可在外层覆盖或置空关闭）
VM1RECON_PROXY="${VM1RECON_PROXY:-http://192.168.8.195:18899}"
export http_proxy="$VM1RECON_PROXY" https_proxy="$VM1RECON_PROXY"
export HTTP_PROXY="$VM1RECON_PROXY" HTTPS_PROXY="$VM1RECON_PROXY"
# git 不读 http_proxy env，NKSR/pyroki 等包会 git clone github 依赖，
# 必须用 git config 配代理，否则 gnutls_handshake failed。
git config --global http.proxy "$VM1RECON_PROXY"
git config --global https.proxy "$VM1RECON_PROXY"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD="$ROOT/third_party"
MEGASAM="$THIRD/megasam-package"
LOG="$ROOT/install_vm1recon.log"
MARK_DIR="$ROOT/.install_marks_recon"
mkdir -p "$MARK_DIR"
: > "$LOG"

# CUDA 11.8 toolkit（conda env 内通过 cudatoolkit=11.8 获得，不需要系统级安装）
# 但 DROID-SLAM setup.py 编译需要 nvcc，conda env 的 nvcc 即可。
# xformers / torch-scatter 编译也用 env 内的 nvcc。

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}

log() { echo -e "\n========== [$(date +%H:%M:%S)] $* ==========" | tee -a "$LOG"; }
run() {
    local name="$1"; shift
    if [[ -f "$MARK_DIR/$name.done" ]]; then
        log "SKIP $name (已完成)"; return 0
    fi
    log "STEP $name 开始"
    if "$@" >>"$LOG" 2>&1; then
        touch "$MARK_DIR/$name.done"
        log "STEP $name 完成 ✓"
    else
        local rc=$?
        log "!!!! STEP $name 失败 (rc=$rc) — 见 $LOG 尾部 !!!!"
        echo "FAILED_AT=$name" > "$MARK_DIR/_status"
        exit $rc
    fi
}

PIP="pip install --no-input"

log "vm1recon 安装开始；CC=$CC CXX=$CXX"

# 1. 从 environment.yml 创建环境 ------------------------------------------------
if ! conda env list | grep -qE "^vm1recon\b"; then
    run create_env conda env create -f "$MEGASAM/environment.yml"
else
    log "vm1recon 已存在，跳过创建"
fi
set +u  # conda activate 的 MKL 脚本会引用未绑定变量，临时关闭（此处保留，-u 已全局关闭，幂等）
conda activate vm1recon
PY="$(conda run -n vm1recon which python)"; PIP_BIN="$(conda run -n vm1recon which pip)"
log "激活 vm1recon: python=$PY"; conda run -n vm1recon python --version >>"$LOG" 2>&1

# 用 CONDA_RUN 封装所有后续命令
CONDA_RUN="conda run -n vm1recon"

# 1.5 装 env 内的 cuda-nvcc 11.8 ------------------------------------------------
# 系统默认 CUDA_HOME 指向 cuda-12.8，torch 2.0.1 是 cu118 编译，cpp_extension 的
# _check_cuda_version 检测到 12.8≠11.8 会拒绝编译 NKSR/DROID-SLAM。setup.md 假设
# "conda env 的 nvcc 即可"，但 environment.yml 没装 nvcc，这里补装并把 CUDA_HOME
# 指向 env，使所有编译用 11.8 的 nvcc。
run cuda_nvcc conda install -y -n vm1recon -c nvidia 'cuda-nvcc=11.8' 'cuda-cudart-dev=11.8' 'cuda-version=11.8'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

# 1.6 钉 numpy=1.26.3 -----------------------------------------------------------
# megasam environment.yml 把 numpy==1.26.3 写在 pip 段，但 conda 求解器在解析 conda
# 依赖时已先选了 conda-forge 的 numpy 2.2.6（pytorch/scipy 间接拉入），pip 段的约束
# 无法覆盖 → 环境里最终是 numpy 2.2.6，与 torch 2.0.1（numpy 1.x 编译）冲突，
# import torch 报 _ARRAY_API not found，import nksr 直接崩。故在 conda 级别显式钉 1.26.3。
run pin_numpy conda install -y -n vm1recon -c conda-forge 'numpy=1.26.3'

# 1.7 钉 setuptools<81 -----------------------------------------------------------
# conda 装的 setuptools>=81 移除了 pkg_resources，而 torch.utils.cpp_extension 仍
# `from pkg_resources import packaging` → DROID-SLAM/NKSR 编译 import 即崩。
# 钉 <81 保留 pkg_resources（vm1recon 是 py3.10，80.x 无 ImpImporter 问题）。
run pin_setuptools conda install -y -n vm1recon -c conda-forge 'setuptools<81'

# 2. requirements.txt（real2sim 根目录的）---------------------------------------
run reqs $CONDA_RUN $PIP_BIN install -r "$ROOT/requirements_nogit.txt"

# 2.5 强制钉回 torch 2.0.1 cu118 + numpy/scipy ----------------------------------
# 关键：requirements_nogit.txt 里的包（transformers/timm 等）依赖 torch 且无上限，
# pip 会从 PyPI 拉最新的 torch 2.1.0+cu121，覆盖 conda 装的 2.0.1+cu118 → 之后
# torch_scatter/xformers/DROID-SLAM/NKSR 全部 ABI 错乱。必须卸掉 pip torch，
# 用 conda 强制重装 2.0.1 cu118 精确 build。同时钉 numpy<2、scipy<1.13（scipy>=1.13
# 要 numpy 2.x ABI）。
run pin_torch bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1recon && \
    pip uninstall -y torch torchvision || true && \
    conda install -n vm1recon --force-reinstall -c pytorch \
        'pytorch=2.0.1=py3.10_cuda11.8_cudnn8.7.0_0' 'torchvision=0.15.2=py310_cu118' -y && \
    pip install 'numpy==1.26.3' 'scipy<1.13'"

# 3. torch-scatter (cu118, 匹配 torch 2.0.1) -------------------------------------
run torch_scatter $CONDA_RUN $PIP_BIN install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 4. xformers 0.0.22.post7 (cu118 + py310 + pt2.0.1) ----------------------------
# setup.md 用 conda install 本地 bz2，这里直接 pip 装对应版本更简单
run xformers $CONDA_RUN $PIP_BIN install xformers==0.0.22.post7

# 5. DROID-SLAM 编译 (用 env 内 nvcc 11.8) --------------------------------------
run droid_slam bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1recon && \
    export CUDA_HOME=\$CONDA_PREFIX && export PATH=\$CUDA_HOME/bin:\$PATH && \
    cd '$MEGASAM/base' && python setup.py install"

# 6. NKSR + 依赖 ----------------------------------------------------------------
# conda 依赖（pyg/nvidia 频道会升 numpy，nksr_conda 后重钉 numpy/scipy）
run nksr_conda conda install -y -n vm1recon -c pyg -c nvidia -c conda-forge \
    pytorch-lightning=1.9.4 tensorboard pybind11 pyg rich pandas omegaconf
run nksr_repin $CONDA_RUN $PIP_BIN install 'numpy==1.26.3' 'scipy<1.13'
# pycg：pycg.huangjh.tech 的自定义 wheel index（open3d==0.16.1+c65c7ef）服务端 TLS
# 损坏（SSL EOF），代理也救不了。python-pycg 本体 wheel 在 PyPI 上有，去掉 [full]
# extra（那个 extra 只是为了拉 .tech 上的自定义 open3d），改用 PyPI 正式版 open3d。
# NKSR 网格化核心不依赖 pycg 的 [full] extra。
run nksr_pycg $CONDA_RUN $PIP_BIN install 'python-pycg==0.5.2' randomname pykdtree plyfile flatten-dict pyntcloud
# nksr 本体：nksr.huangjh.tech wheel index 同样 TLS 损坏，且 wheel 会解成空的 0.0.0
# 包（无 nksr.Reconstructor）。按 setup.md 从源码编译（用 env 内 nvcc 11.8）。
if [[ ! -d /tmp/NKSR ]]; then
    run clone_nksr git clone --depth 1 https://github.com/nv-tlabs/NKSR.git /tmp/NKSR
fi
run nksr_submods bash -c "cd /tmp/NKSR/package && \
    git clone https://github.com/AcademySoftwareFoundation/openvdb.git external/openvdb && \
    git -C external/openvdb checkout 7edd8cd86f105a01a41ad7b2bf59a81034cd79fb && \
    git clone https://gitlab.com/libeigen/eigen.git external/eigen && \
    git -C external/eigen checkout 3.4"
run nksr_core bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1recon && \
    export CUDA_HOME=\$CONDA_PREFIX && export PATH=\$CUDA_HOME/bin:\$PATH && \
    rm -rf /tmp/NKSR/package/build /tmp/NKSR/package/*.egg-info && \
    cd /tmp/NKSR/package && python -m pip install --no-build-isolation ."
run nksr_verify_build bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1recon && \
    python -c \"import torch, nksr; r=nksr.Reconstructor(torch.device('cuda')); print(nksr.__version__, type(r).__name__)\""
run nksr_extras $CONDA_RUN $PIP_BIN install trimesh tyro h5py rtree open3d

# 7. GeoCalib -------------------------------------------------------------------
if [[ ! -d "$THIRD/GeoCalib" ]]; then
    run clone_geocalib git clone https://github.com/hongsukchoi/GeoCalib.git "$THIRD/GeoCalib"
fi
run geocalib bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1recon && cd '$THIRD/GeoCalib' && pip install -e ."

# 7.5 最终钉 numpy/scipy/plyfile ------------------------------------------------
# geocalib 的 pip install -e . 会把 numpy 升到 2.x（与 nksr 的 numpy 1.x ABI 冲突），
# open3d/extras 也会拉新 plyfile（要 numpy>=2）。最后强制钉回 numpy<2、scipy<1.13、
# plyfile<1.0，保证 nksr 可导入。
run final_pin $CONDA_RUN $PIP_BIN install 'numpy==1.26.3' 'scipy<1.13' 'plyfile<1.0'

# 7.6 chumpy + smplx (stage3 postprocessing 跑在 vm1recon，gravity_calibration import smplx)
# smplx 的 setup.py 构建时 import pip，--no-build-isolation 用 env 内 pip。
run smplx bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1recon && \
    pip install --no-build-isolation 'git+https://github.com/hongsukchoi/chumpy' && \
    pip install --no-build-isolation 'git+https://github.com/hongsukchoi/smplx'"

# 8. 验证关键导入（nksr 失败则本步失败，不再静默通过）----------------------------
run verify bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1recon && python -c '
import torch, numpy, scipy
assert torch.__version__.startswith(\"2.0.1\"), torch.__version__
assert torch.version.cuda == \"11.8\", torch.version.cuda
assert numpy.__version__.startswith(\"1.26\"), numpy.__version__
import torch_scatter, nksr, geocalib, h5py, cv2
r = nksr.Reconstructor(torch.device(\"cuda\"))
print(\"verify OK:\", torch.__version__, numpy.__version__, nksr.__version__, type(r).__name__)
'"

log "全部步骤完成 ✓✓✓"
echo "ALL_DONE" > "$MARK_DIR/_status"
