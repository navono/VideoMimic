"""Configuration: host paths, env-derived settings, stage definitions.

All host-absolute paths and env-tunable knobs live here. Mirrors the top of the
single-file server.py 1:1 (values unchanged). CONDA_SH auto-detects the conda
installation (CONDA_PREFIX, then `which conda`, then ~/miniforge3) so it works
whether conda lives in ~/miniforge3 (5051 ubuntu22), /data/cache/<user>/miniforge3
(this dev box), or elsewhere; override explicitly via the CONDA_SH env.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #

SERVER_DIR = Path(__file__).resolve().parent


def _find_real2sim_dir() -> Path:
    """Find the real2sim project root whether this file is in ./ or ./server/."""
    for candidate in (SERVER_DIR, *SERVER_DIR.parents):
        if (candidate / "Makefile").exists() and (candidate / "demo_data").exists():
            return candidate
    return SERVER_DIR


REAL2SIM_DIR = _find_real2sim_dir()
JOBS_DIR = SERVER_DIR / "jobs"
DEMO_DATA_DIR = REAL2SIM_DIR / "demo_data"
LOG_DIR = SERVER_DIR / "logs"
SERVER_LOG_PATH = LOG_DIR / "server.log"

# --------------------------------------------------------------------------- #
# Logging / job retention tunables
# --------------------------------------------------------------------------- #

LOG_MAX_BYTES = int(os.environ.get("VIDEOMIMIC_LOG_MAX_BYTES", str(50 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.environ.get("VIDEOMIMIC_LOG_BACKUP_COUNT", "7"))
JOB_RETENTION_DAYS = int(os.environ.get("VIDEOMIMIC_JOB_RETENTION_DAYS", "7"))

# --------------------------------------------------------------------------- #
# Conda / proxy / viser
# --------------------------------------------------------------------------- #

def _default_conda_sh() -> str:
    """Locate conda.sh for the default CONDA_SH.

    conda 可能装在 ~/miniforge3（5051 的 ubuntu22）、/data/cache/<user>/miniforge3
    （本机 pqx，~ 不指向那里），或任意自定义位置。按优先级探测，避免写死单一路径：
      1. 环境变量 CONDA_SH（调用方显式指定，最高优先级，在下方 os.environ.get 处理）
      2. 已激活 conda 的 CONDA_PREFIX/etc/profile.d/conda.sh
      3. `conda` 可执行文件反推：$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
      4. ~/miniforge3/etc/profile.d/conda.sh（兼容旧默认）
    """
    # 2. 已激活的 conda
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        cand = Path(prefix) / "etc" / "profile.d" / "conda.sh"
        if cand.exists():
            return str(cand)
    # 3. 从 `conda` 可执行文件反推
    conda_exe = shutil.which("conda")
    if conda_exe:
        # conda exe 在 <prefix>/bin/conda → conda.sh 在 <prefix>/etc/profile.d/conda.sh
        cand = Path(conda_exe).resolve().parent.parent / "etc" / "profile.d" / "conda.sh"
        if cand.exists():
            return str(cand)
    # 4. 旧默认
    return os.path.expanduser("~/miniforge3/etc/profile.d/conda.sh")


CONDA_SH = os.environ.get("CONDA_SH", _default_conda_sh())
CONDA_EVAL = f"source {CONDA_SH} && conda activate"
DEFAULT_PORT = 8090
DEFAULT_PROXY = "http://192.168.8.195:18899"
PROXY_URL = os.environ.get("VIDEOMIMIC_PROXY", DEFAULT_PROXY)
# server 内部 viser 端口（benchverse 预览走 /api/viser proxy 到这里）。
# H20 服务器限定外部可访问端口 10000~20000，故默认 18091（与 SERVER_PORT=18090、
# 手动 make visualize 的 VIS_PORT=18089 错开，避免冲突）。
VISER_PORT = int(os.environ.get("VIDEOMIMIC_VISER_PORT", "18091"))
VISER_URL = os.environ.get("VIDEOMIMIC_VISER_URL", f"http://127.0.0.1:{VISER_PORT}")

# 默认 GPU 编号（CUDA_VISIBLE_DEVICES）。请求未显式传 device 时用它，避免落到
# 满载卡上 OOM。默认 "6"（本机空闲卡）；环境变量 VIDEOMIMIC_DEVICE 可覆盖。
DEFAULT_DEVICE = os.environ.get("VIDEOMIMIC_DEVICE", "6")

# Conda envs used by the pipeline
CONDA_VM1RS = "vm1rs"
CONDA_VM1RECON = "vm1recon"

# Post-pipeline：把 retarget h5 落到 benchverse jobs/ 并转成 motion.npz（RL 训练输入）。
# Post-pipeline：把 retarget h5 落到 benchverse jobs/ 并转成 motion.npz（RL 训练输入）。
# 默认按 ~/sourcecode/ 布局解析（本机 pqx 与 5051 ubuntu22 布局一致），可由环境变量覆盖。
_HOME = Path(os.environ.get("BENCHVERSE_HOME", str(Path.home())))
BENCHVERSE_JOBS_DIR = Path(os.environ.get("BENCHVERSE_JOBS_DIR", str(_HOME / "sourcecode/benchverse-skill-server/jobs")))
RL_LAB_DIR = Path(os.environ.get("RL_LAB_DIR", str(_HOME / "sourcecode/unitree_rl_lab")))

# --------------------------------------------------------------------------- #
# Stage progression (order matters — _run_logged_command indexes into this)
# --------------------------------------------------------------------------- #

STAGES = [
    "extracting_frames",
    "preprocessing",
    "reconstruction",
    "optimization",
    "postprocessing",
    "retargeting",
]

# Regex to detect stage transitions in make output
STAGE_PATTERNS = {
    "extracting_frames": r"extract.*frame|Extracting frames",
    "preprocessing": r"Step 0.*Preprocessing|Running Grounding-SAM|Running ViTPose|Running VIMO|Running BSTRO",
    "reconstruction": r"Step 1.*Reconstruction|MegaSAM",
    "optimization": r"Step 2.*Optimization|MegaHunter",
    "postprocessing": r"Step 3.*Postprocessing|postprocessing",
    "retargeting": r"Step 4.*Retargeting|robot_motion_retargeting",
}