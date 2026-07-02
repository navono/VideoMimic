"""Job state: Pydantic model + filesystem-backed status CRUD + helpers.

Single source of truth for job status; mirrors the single-file server.py 1:1.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from .config import DEMO_DATA_DIR, JOBS_DIR, JOB_RETENTION_DAYS, REAL2SIM_DIR


class JobStatus(BaseModel):
    job_id: str
    task_id: str | None = None  # benchverse client 期望响应里有 task_id（= job_id）
    status: str  # pending, running, completed, failed
    stage: str | None = None
    progress: float = 0.0  # 0.0 - 1.0
    error_message: str | None = None
    video_stem: str | None = None
    result_files: list[dict] = []
    stage_timestamps: dict[str, str] = {}


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def status_path(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def read_status(job_id: str) -> dict:
    p = status_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return json.loads(p.read_text())


def write_status(job_id: str, data: dict) -> None:
    p = status_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(UTC).isoformat()
    p.write_text(json.dumps(data, indent=2))


def mark_stage(job_id: str, stage: str, progress: float) -> None:
    data = read_status(job_id)
    if data.get("status") == "cancelled":
        return
    stage_timestamps = dict(data.get("stage_timestamps") or {})
    stage_timestamps.setdefault(stage, datetime.now(UTC).isoformat())
    write_status(job_id, {
        **data,
        "stage": stage,
        "progress": progress,
        "stage_timestamps": stage_timestamps,
    })


def append_job_log(job_id: str, line: str) -> None:
    log_path = job_dir(job_id) / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def safe_stem(name: str | None, fallback: str = "skill_job") -> str:
    """benchverse 风格的 job 目录前缀：视频名 stem，非 [A-Za-z0-9._-] 替成 _。"""
    raw = Path(name or fallback).name
    stem = Path(raw).stem or raw or fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe or fallback


def new_job_id(video_filename: str) -> str:
    """Create a stable, readable job id used as both API job id and VideoMimic VID_STEM."""
    prefix = safe_stem(video_filename, fallback="real2sim_job")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{prefix}_{stamp}"
    for _ in range(10):
        candidate = base if _ == 0 else f"{base}_{uuid.uuid4().hex[:4]}"
        if not job_dir(candidate).exists() and not (DEMO_DATA_DIR / candidate).exists():
            return candidate
    return f"{base}_{uuid.uuid4().hex[:8]}"


# benchverse client 可传入 task_id 作 job_id；它同时拼进 JOBS_DIR/{id}/ 与
# DEMO_DATA_DIR/{id}/，必须防路径穿越（../、绝对路径、隐藏文件名等）。
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_task_id(task_id: str) -> str:
    """校验外部传入的 task_id，合法则原样返回（作 job_id）。不合法抛 400。"""
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id must not be empty")
    if task_id in (".", ".."):
        raise HTTPException(status_code=400, detail="task_id must not be '.' or '..'")
    if not _TASK_ID_RE.match(task_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid task_id: must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}",
        )
    return task_id


def collect_result_files(video_stem: str) -> list[dict]:
    """Scan output_calib_mesh for retargeting results."""
    calib_dir = DEMO_DATA_DIR / video_stem / "output_calib_mesh"
    results = []
    if not calib_dir.exists():
        return results
    for subdir in sorted(calib_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for f in subdir.iterdir():
            if f.is_file() and f.suffix in (".h5", ".obj", ".ply"):
                results.append({
                    "name": f.name,
                    "path": str(f.relative_to(REAL2SIM_DIR)),
                    "size": f.stat().st_size,
                })
    return results


def cleanup_old_jobs() -> None:
    """Delete old per-job directories so pipeline logs do not accumulate forever."""
    if JOB_RETENTION_DAYS <= 0 or not JOBS_DIR.exists():
        return
    import shutil

    cutoff = datetime.now(UTC) - timedelta(days=JOB_RETENTION_DAYS)
    for job_dir_path in JOBS_DIR.iterdir():
        if not job_dir_path.is_dir():
            continue
        sp = job_dir_path / "status.json"
        try:
            if sp.exists():
                data = json.loads(sp.read_text())
                timestamp = data.get("updated_at") or data.get("created_at")
                job_time = datetime.fromisoformat(timestamp) if timestamp else None
            else:
                job_time = datetime.fromtimestamp(job_dir_path.stat().st_mtime, tz=UTC)
            if job_time and job_time < cutoff:
                shutil.rmtree(job_dir_path)
                # logger 导入放局部避免循环
                import logging
                logging.getLogger("videomimic-server").info(
                    "Removed expired VideoMimic job directory: %s", job_dir_path)
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("videomimic-server").warning(
                "Failed to cleanup VideoMimic job directory %s: %s", job_dir_path, exc)


def postprocessed_dir_from_status(data: dict) -> Path | None:
    for result_file in data.get("result_files", []):
        path = result_file.get("path")
        if path:
            return (REAL2SIM_DIR / path).parent
    return None


def is_cancelled(job_id: str) -> bool:
    try:
        return read_status(job_id).get("status") == "cancelled"
    except HTTPException:
        return False