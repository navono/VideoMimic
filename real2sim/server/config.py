"""Configuration: host paths, env-derived settings, stage definitions.

All host-absolute paths and env-tunable knobs live here. Mirrors the top of the
single-file server.py 1:1 (values unchanged). CONDA_SH defaults to
~/miniforge3/etc/profile.d/conda.sh (expands per-user: /home/ubuntu22 on the
5051 deployment, /home/pingqixing on this dev box); override via the CONDA_SH env.
"""

from __future__ import annotations

import os
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

CONDA_SH = os.environ.get(
    "CONDA_SH",
    os.path.expanduser("~/miniforge3/etc/profile.d/conda.sh"),
)
CONDA_EVAL = f"source {CONDA_SH} && conda activate"
DEFAULT_PORT = 8090
DEFAULT_PROXY = "http://192.168.8.195:18899"
PROXY_URL = os.environ.get("VIDEOMIMIC_PROXY", DEFAULT_PROXY)
VISER_PORT = int(os.environ.get("VIDEOMIMIC_VISER_PORT", "8089"))
VISER_URL = os.environ.get("VIDEOMIMIC_VISER_URL", f"http://127.0.0.1:{VISER_PORT}")

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