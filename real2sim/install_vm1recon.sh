#!/usr/bin/env bash
# vm1recon 环境安装脚本（按 real2sim/docs/setup.md）。
# 遇错即停，全量记日志；每步写 step.done 标记便于定位/续装。
set -euo pipefail

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
set +u  # conda activate 的 MKL 脚本会引用未绑定变量，临时关闭
conda activate vm1recon
set -u
PY="$(conda run -n vm1recon which python)"; PIP_BIN="$(conda run -n vm1recon which pip)"
log "激活 vm1recon: python=$PY"; conda run -n vm1recon python --version >>"$LOG" 2>&1

# 用 CONDA_RUN 封装所有后续命令
CONDA_RUN="conda run -n vm1recon"

# 2. requirements.txt（real2sim 根目录的）---------------------------------------
run reqs $CONDA_RUN $PIP_BIN install -r "$ROOT/requirements_nogit.txt"

# 3. torch-scatter ---------------------------------------------------------------
run torch_scatter $CONDA_RUN $PIP_BIN install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 4. xformers 0.0.22.post7 (cu118 + py310 + pt2.0.1) ----------------------------
# setup.md 用 conda install 本地 bz2，这里直接 pip 装对应版本更简单
run xformers $CONDA_RUN $PIP_BIN install xformers==0.0.22.post7

# 5. DROID-SLAM 编译 ------------------------------------------------------------
run droid_slam bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1recon && cd '$MEGASAM/base' && python setup.py install"

# 6. NKSR + 依赖 ----------------------------------------------------------------
# conda 依赖
run nksr_conda conda install -y -n vm1recon -c pyg -c nvidia -c conda-forge \
    pytorch-lightning=1.9.4 tensorboard pybind11 pyg rich pandas omegaconf
# pycg 等
run nksr_pycg $CONDA_RUN $PIP_BIN install -f https://pycg.huangjh.tech/packages/index.html 'python-pycg[full]==0.5.2' randomname pykdtree plyfile flatten-dict pyntcloud
# nksr 本体
run nksr_core $CONDA_RUN $PIP_BIN install -f https://nksr.huangjh.tech/whl/torch-2.0.0+cu118.html nksr
run nksr_extras $CONDA_RUN $PIP_BIN install trimesh tyro h5py rtree open3d

# 7. GeoCalib -------------------------------------------------------------------
if [[ ! -d "$THIRD/GeoCalib" ]]; then
    run clone_geocalib git clone https://github.com/hongsukchoi/GeoCalib.git "$THIRD/GeoCalib"
fi
run geocalib bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1recon && cd '$THIRD/GeoCalib' && pip install -e ."

# 8. 验证关键导入 ----------------------------------------------------------------
run verify $CONDA_RUN python -c "
pkgs = ['torch', 'torchvision', 'raft', 'droid', 'lietorch', 'depth_anything', 'unidepth', 'nksr', 'geocalib', 'h5py', 'cv2', 'numpy']
for p in pkgs:
    try:
        m = __import__(p)
        v = getattr(m, '__version__', '?')
        print(f'  ✓ {p:18s} {v}')
    except Exception as e:
        print(f'  ✗ {p:18s} {e}')
"

log "全部步骤完成 ✓✓✓"
echo "ALL_DONE" > "$MARK_DIR/_status"
