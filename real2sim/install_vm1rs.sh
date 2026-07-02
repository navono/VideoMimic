#!/usr/bin/env bash
# vm1rs 环境安装脚本（按 real2sim/docs/setup.md，torch 提前到最前避免 CPU-torch 陷阱）。
# 遇错即停（set -euo pipefail），全量记日志；每步写 step.done 标记便于定位/续装。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$ROOT/install_vm1rs.log"
MARK_DIR="$ROOT/.install_marks"
mkdir -p "$MARK_DIR"
: > "$LOG"   # 清空旧日志

export CUDA_HOME="/usr/local/cuda-12.4"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}

log() { echo -e "\n========== [$(date +%H:%M:%S)] $* ==========" | tee -a "$LOG"; }
run() {  # run <step_name> <cmd...> ; 失败则记并退出
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
PYBin=""  # 占位，激活后即 conda env 的 python

log "vm1rs 安装开始；CUDA_HOME=$CUDA_HOME CC=$CC"

# 1. 建 env -------------------------------------------------------------
if ! conda env list | grep -qE "^vm1rs\b"; then
    run create_env conda create -y -n vm1rs python=3.12
else
    log "vm1rs 已存在，跳过创建"
fi
conda activate vm1rs
PY="$(which python)"; PIP_BIN="$(which pip)"
log "激活 vm1rs: python=$PY"; "$PY" --version >>"$LOG" 2>&1

# 2. torch（cu124）先装 -------------------------------------------------
run torch "$PIP_BIN" install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. requirements.txt ---------------------------------------------------
# chumpy/smplx 的 setup.py 在构建时 import pip，但 PEP517 隔离环境里没有 pip → 构建崩。
# 拆开：先装去掉这两个 git 包的其余依赖，再用 --no-build-isolation 单独装它们（用 env 内的 pip）。
run reqs "$PIP_BIN" install -r "$ROOT/requirements_nogit.txt"
run reqs_chumpy "$PIP_BIN" install --no-build-isolation "git+https://github.com/hongsukchoi/chumpy"
run reqs_smplx "$PIP_BIN" install --no-build-isolation "git+https://github.com/hongsukchoi/smplx"

# 4. Grounded-SAM-2 + grounding_dino -----------------------------------
mkdir -p "$ROOT/third_party"
if [[ ! -d "$ROOT/third_party/Grounded-SAM-2" ]]; then
    run clone_sam2 git clone https://github.com/hongsukchoi/Grounded-SAM-2.git "$ROOT/third_party/Grounded-SAM-2"
fi
run sam2_core bash -c "cd '$ROOT/third_party/Grounded-SAM-2' && $PIP_BIN install -e ."
run grounding_dino bash -c "cd '$ROOT/third_party/Grounded-SAM-2' && $PIP_BIN install --no-build-isolation -e grounding_dino"
run transformers "$PIP_BIN" install transformers

# 5. ViTPose ------------------------------------------------------------
# 注意：setuptools>=81 (2025) 移除了 pkg_resources，而 mim 仍 import pkg_resources →
# mim 启动即崩。故装完 openmim 后把 setuptools 钉到 <81（保留 pkg_resources）。
run openmim bash -c "$PIP_BIN install -U openmim && $PIP_BIN install 'setuptools<81'"
run mmcv bash -c "source ${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh} && conda activate vm1rs && $PIP_BIN install 'setuptools<81' && $PIP_BIN install --no-build-isolation mmcv==1.3.9"
if [[ ! -d "$ROOT/third_party/ViTPose" ]]; then
    run clone_vitpose git clone https://github.com/ViTAE-Transformer/ViTPose.git "$ROOT/third_party/ViTPose"
fi
run vitpose bash -c "cd '$ROOT/third_party/ViTPose' && $PIP_BIN install numpy cython wheel && $PIP_BIN install -v --no-build-isolation -e ."

# 6. VIMO ---------------------------------------------------------------
run vimo "$PIP_BIN" install git+https://github.com/hongsukchoi/VIMO.git

# 7. BSTRO --------------------------------------------------------------
if [[ ! -d "$ROOT/third_party/bstro" ]]; then
    run clone_bstro git clone --recursive https://github.com/hongsukchoi/bstro.git "$ROOT/third_party/bstro"
fi
run bstro bash -c "cd '$ROOT/third_party/bstro' && $PY setup.py build develop"

# 8. jax + jaxls + pyroki + viser --------------------------------------
run jax "$PIP_BIN" install -U "jax[cuda12]"
run jaxls "$PIP_BIN" install "git+https://github.com/brentyi/jaxls.git"
if [[ ! -d "$ROOT/third_party/pyroki" ]]; then
    run clone_pyroki git clone https://github.com/chungmin99/pyroki.git "$ROOT/third_party/pyroki"
fi
run pyroki bash -c "cd '$ROOT/third_party/pyroki' && git checkout 70b30a56b1e1ea83fb4c2cac8fe2c63a0624b9ce && $PIP_BIN install -e ."
if [[ ! -d "$ROOT/third_party/viser" ]]; then
    run clone_viser git clone https://github.com/nerfstudio-project/viser "$ROOT/third_party/viser"
fi
run viser bash -c "cd '$ROOT/third_party/viser' && $PIP_BIN install -e ."

# 9. 服务依赖（real2sim/server）----------------------------------------
run server_reqs "$PIP_BIN" install -r "$ROOT/server/requirements.txt"

log "全部步骤完成 ✓✓✓"
echo "ALL_DONE" > "$MARK_DIR/_status"
