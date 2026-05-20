"""VideoMimic real2sim HTTP API server.

Wraps the real2sim pipeline as an async FastAPI service so that Benchverse
(or any client) can submit videos, track progress, and download results over
HTTP instead of SSH.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _find_real2sim_dir() -> Path:
    """Find the real2sim project root whether this file is copied to ./ or ./server/."""
    for candidate in (SERVER_DIR, *SERVER_DIR.parents):
        if (candidate / "Makefile").exists() and (candidate / "demo_data").exists():
            return candidate
    return SERVER_DIR


SERVER_DIR = Path(__file__).resolve().parent
REAL2SIM_DIR = _find_real2sim_dir()
JOBS_DIR = SERVER_DIR / "jobs"
DEMO_DATA_DIR = REAL2SIM_DIR / "demo_data"
LOG_DIR = SERVER_DIR / "logs"
SERVER_LOG_PATH = LOG_DIR / "server.log"
LOG_MAX_BYTES = int(os.environ.get("VIDEOMIMIC_LOG_MAX_BYTES", str(50 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.environ.get("VIDEOMIMIC_LOG_BACKUP_COUNT", "7"))
JOB_RETENTION_DAYS = int(os.environ.get("VIDEOMIMIC_JOB_RETENTION_DAYS", "7"))
CONDA_SH = os.environ.get("CONDA_SH", "/home/ubuntu22/miniforge3/etc/profile.d/conda.sh")
CONDA_EVAL = f"source {CONDA_SH} && conda activate"
DEFAULT_PORT = 8090
DEFAULT_PROXY = "http://127.0.0.1:18899"
PROXY_URL = os.environ.get("VIDEOMIMIC_PROXY", DEFAULT_PROXY)
VISER_URL = os.environ.get("VIDEOMIMIC_VISER_URL", "http://127.0.0.1:8081")

# Conda envs used by the pipeline
CONDA_VM1RS = "vm1rs"
CONDA_VM1RECON = "vm1recon"

# Stage progression (order matters)
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("videomimic-server")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        SERVER_LOG_PATH,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    stage: str | None = None
    progress: float = 0.0  # 0.0 - 1.0
    error_message: str | None = None
    video_stem: str | None = None
    result_files: list[dict] = []
    stage_timestamps: dict[str, str] = {}
    created_at: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _read_status(job_id: str) -> dict:
    p = _status_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return json.loads(p.read_text())


def _write_status(job_id: str, data: dict) -> None:
    p = _status_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(UTC).isoformat()
    p.write_text(json.dumps(data, indent=2))


def _mark_stage(job_id: str, stage: str, progress: float) -> None:
    data = _read_status(job_id)
    stage_timestamps = dict(data.get("stage_timestamps") or {})
    stage_timestamps.setdefault(stage, datetime.now(UTC).isoformat())
    _write_status(job_id, {
        **data,
        "stage": stage,
        "progress": progress,
        "stage_timestamps": stage_timestamps,
    })


def _detect_stage(line: str) -> str | None:
    for stage, pattern in STAGE_PATTERNS.items():
        if re.search(pattern, line, re.IGNORECASE):
            return stage
    return None


def _collect_result_files(video_stem: str) -> list[dict]:
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


def _height_arg(height: float) -> str:
    if height == -1:
        return "-1"
    if height == 0:
        return "0"
    if not 1.0 <= height <= 2.2:
        raise HTTPException(status_code=400, detail="height must be -1, 0, or between 1.0 and 2.2 meters")
    return f"{height:g}"


def _proxy_headers(headers) -> dict[str, str]:
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "content-encoding",
    }
    return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}


def _rewrite_viser_html(content: bytes) -> bytes:
    text = content.decode("utf-8", errors="replace")
    text = text.replace('href="/', 'href="/api/viser/')
    text = text.replace('src="/', 'src="/api/viser/')
    text = text.replace('action="/', 'action="/api/viser/')
    text = text.replace('url(/', 'url(/api/viser/')
    return text.encode("utf-8")


def _websocket_subprotocols(websocket: WebSocket) -> list[str]:
    value = websocket.headers.get("sec-websocket-protocol", "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _append_job_log(job_id: str, line: str) -> None:
    log_path = _job_dir(job_id) / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def _cleanup_old_jobs() -> None:
    """Delete old per-job directories so pipeline logs do not accumulate forever."""
    if JOB_RETENTION_DAYS <= 0 or not JOBS_DIR.exists():
        return

    cutoff = datetime.now(UTC) - timedelta(days=JOB_RETENTION_DAYS)
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue

        status_path = job_dir / "status.json"
        try:
            if status_path.exists():
                data = json.loads(status_path.read_text())
                timestamp = data.get("updated_at") or data.get("created_at")
                job_time = datetime.fromisoformat(timestamp) if timestamp else None
            else:
                job_time = datetime.fromtimestamp(job_dir.stat().st_mtime, tz=UTC)

            if job_time and job_time < cutoff:
                import shutil

                shutil.rmtree(job_dir)
                logger.info("Removed expired VideoMimic job directory: %s", job_dir)
        except Exception as exc:
            logger.warning("Failed to cleanup VideoMimic job directory %s: %s", job_dir, exc)


def _postprocessed_dir_from_status(data: dict) -> Path | None:
    for result_file in data.get("result_files", []):
        path = result_file.get("path")
        if path:
            return (REAL2SIM_DIR / path).parent
    return None


def _start_final_viser(job_id: str, data: dict) -> None:
    postprocessed_dir = _postprocessed_dir_from_status(data)
    if not postprocessed_dir or not postprocessed_dir.exists():
        raise HTTPException(status_code=404, detail="No completed visualization directory found")

    subprocess.run(
        ["pkill", "-f", "complete_results_egoview_visualization.py"],
        capture_output=True,
        check=False,
        timeout=5,
    )

    log_path = _job_dir(job_id) / "viser.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    robot = data.get("robot") or "g1"
    command = (
        f'{CONDA_EVAL} && conda activate {CONDA_VM1RS} && '
        f'python visualization/complete_results_egoview_visualization.py '
        f'--postprocessed-dir "{postprocessed_dir}" '
        f'--robot-name "{robot}" --is-megasam'
    )
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("ab") as log_file:
        subprocess.Popen(
            command,
            executable="/bin/bash",
            shell=True,
            cwd=str(REAL2SIM_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    logger.info("[%s] Started final Viser for %s", job_id, postprocessed_dir)


async def _run_logged_command(
    job_id: str,
    label: str,
    command: str,
    *,
    stage_start_index: int,
) -> tuple[int, list[str]]:
    logger.info("[%s] Starting %s", job_id, label)
    logger.info("[%s] Command: %s", job_id, command)
    _append_job_log(job_id, f"$ {command}")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if PROXY_URL:
        env.update({
            "http_proxy": PROXY_URL,
            "https_proxy": PROXY_URL,
            "HTTP_PROXY": PROXY_URL,
            "HTTPS_PROXY": PROXY_URL,
        })

    proc = await asyncio.create_subprocess_shell(
        command,
        executable="/bin/bash",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REAL2SIM_DIR),
        env=env,
    )

    if proc.stdout is None:
        raise RuntimeError(f"{label} did not expose stdout")

    current_stage_idx = stage_start_index
    stage_progress_base = {0: 0.0, 1: 0.15, 2: 0.30, 3: 0.50, 4: 0.70, 5: 0.85}
    tail_lines: list[str] = []

    async for line_bytes in proc.stdout:
        line = line_bytes.decode(errors="replace").rstrip()
        tail_lines.append(line)
        tail_lines = tail_lines[-200:]

        logger.info("[%s] %s", job_id, line)
        _append_job_log(job_id, line)

        detected = _detect_stage(line)
        if detected:
            stage_idx = STAGES.index(detected) if detected in STAGES else current_stage_idx
            if stage_idx > current_stage_idx:
                current_stage_idx = stage_idx
                progress = stage_progress_base.get(stage_idx, 0.9)
                _mark_stage(job_id, detected, progress)
                logger.info("[%s] Stage changed to %s (progress %.2f)", job_id, detected, progress)

    await proc.wait()
    logger.info("[%s] Finished %s with exit code %s", job_id, label, proc.returncode)
    _append_job_log(job_id, f"{label} exited with code {proc.returncode}")
    return proc.returncode or 0, tail_lines


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


async def run_pipeline(job_id: str, video_stem: str, stride: int, height: float, robot: str, gender: str) -> None:
    """Execute the full real2sim pipeline for a job, updating status along the way."""
    job_dir = _job_dir(job_id)
    height_value = _height_arg(height)

    try:
        logger.info(
            "[%s] Pipeline submitted: video=%s stride=%s height=%s robot=%s gender=%s",
            job_id,
            video_stem,
            stride,
            height,
            robot,
            gender,
        )
        _write_status(job_id, {**_read_status(job_id), "status": "running"})
        _mark_stage(job_id, "extracting_frames", 0.0)

        # Extract frames first
        cam_dir = DEMO_DATA_DIR / video_stem / "input_images" / "cam01"
        video_src = job_dir / "input" / _read_status(job_id).get("video_filename", "video.mp4")

        # Step 0: Extract frames
        extract_cmd = (
            f'{CONDA_EVAL} && conda activate {CONDA_VM1RS} && '
            f'python utilities/extract_frames_from_video.py '
            f'--video-path "{video_src}" '
            f'--output-dir "{cam_dir}" '
            f'--start-frame 0 --end-frame 99999'
        )
        exit_code, tail_lines = await _run_logged_command(
            job_id,
            "frame extraction",
            extract_cmd,
            stage_start_index=0,
        )
        if exit_code != 0:
            raise RuntimeError(f"Frame extraction failed: {' | '.join(tail_lines[-20:])}")

        _mark_stage(job_id, "preprocessing", 0.15)

        # Build make pipeline command
        make_cmd = (
            f'{CONDA_EVAL} && '
            f'export HF_TOKEN=${{HF_TOKEN:-}} && '
            f'make pipeline VIDEO_NAME="{video_stem}" STRIDE={stride} HEIGHT={height_value} '
            f'ROBOT={robot} GENDER={gender} PROXY="{PROXY_URL}"'
        )

        exit_code, tail_lines = await _run_logged_command(
            job_id,
            "real2sim pipeline",
            make_cmd,
            stage_start_index=1,
        )
        if exit_code != 0:
            raise RuntimeError(f"Pipeline failed (exit {exit_code}): {' | '.join(tail_lines[-20:])}")

        # Collect results
        result_files = _collect_result_files(video_stem)
        _write_status(job_id, {
            **_read_status(job_id),
            "status": "completed",
            "stage": "retargeting",
            "progress": 1.0,
            "result_files": result_files,
        })
        logger.info("[%s] Pipeline completed with %s result files", job_id, len(result_files))

    except Exception as e:
        logger.exception("[%s] Pipeline failed: %s", job_id, e)
        _write_status(job_id, {
            **_read_status(job_id),
            "status": "failed",
            "error_message": str(e),
        })


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="VideoMimic Real2Sim API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.api_route("/api/viser", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/api/viser/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_viser(request: Request, path: str = ""):
    """Proxy the transient Viser UI on localhost:8081 through this API server."""
    target = f"{VISER_URL.rstrip('/')}/{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client:
            upstream = await client.request(
                request.method,
                target,
                headers=_proxy_headers(request.headers),
                content=await request.body(),
            )
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail="Viser UI is not running") from exc

    content = upstream.content
    headers = _proxy_headers(upstream.headers)
    content_type = upstream.headers.get("content-type", "")
    if "text/html" in content_type:
        content = _rewrite_viser_html(content)
        headers["content-length"] = str(len(content))

    return Response(
        content=content,
        status_code=upstream.status_code,
        headers=headers,
        media_type=content_type or None,
    )


@app.websocket("/api/viser")
@app.websocket("/api/viser/{path:path}")
async def proxy_viser_websocket(websocket: WebSocket, path: str = ""):
    """Proxy Viser WebSocket traffic through this API server."""
    import websockets

    target = f"{VISER_URL.rstrip('/').replace('http://', 'ws://').replace('https://', 'wss://')}/{path}"
    if websocket.url.query:
        target = f"{target}?{websocket.url.query}"

    subprotocols = _websocket_subprotocols(websocket)
    await websocket.accept(subprotocol=subprotocols[0] if subprotocols else None)
    try:
        async with websockets.connect(
            target,
            subprotocols=subprotocols or None,
            max_size=None,
            proxy=None,
        ) as upstream:
            async def client_to_upstream():
                try:
                    while True:
                        message = await websocket.receive()
                        if "text" in message:
                            await upstream.send(message["text"])
                        elif "bytes" in message:
                            await upstream.send(message["bytes"])
                        elif message.get("type") == "websocket.disconnect":
                            break
                except WebSocketDisconnect:
                    pass

            async def upstream_to_client():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            await asyncio.gather(client_to_upstream(), upstream_to_client())
    except Exception as exc:
        logger.warning("Viser websocket proxy closed: %s", exc)


@app.post("/api/pipeline", response_model=JobStatus)
async def submit_pipeline(
    video: UploadFile = File(...),  # noqa: B008
    stride: int = Form(2),  # noqa: B008
    height: float = Form(-1),  # noqa: B008
    robot: str = Form("g1"),  # noqa: B008
    gender: str = Form("male"),  # noqa: B008
):
    """Upload a video and start the real2sim pipeline."""
    _cleanup_old_jobs()

    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    _height_arg(height)

    # Sanitize video stem
    video_stem = Path(video.filename).stem
    video_stem = re.sub(r"[^\w\-.]", "_", video_stem)

    job_id = uuid.uuid4().hex[:12]
    job_dir = _job_dir(job_id)
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_path = input_dir / video.filename
    content = await video.read()
    video_path.write_bytes(content)
    logger.info(
        "[%s] Received video %s (%s bytes), stride=%s height=%s robot=%s gender=%s",
        job_id,
        video.filename,
        len(content),
        stride,
        height,
        robot,
        gender,
    )

    # Create initial status
    now = datetime.now(UTC).isoformat()
    _write_status(job_id, {
        "job_id": job_id,
        "status": "pending",
        "stage": None,
        "progress": 0.0,
        "error_message": None,
        "video_stem": video_stem,
        "video_filename": video.filename,
        "robot": robot,
        "gender": gender,
        "result_files": [],
        "stage_timestamps": {},
        "created_at": now,
        "updated_at": now,
    })

    # Start pipeline in background
    asyncio.create_task(
        run_pipeline(job_id, video_stem, stride, height, robot, gender)
    )

    return JobStatus(**_read_status(job_id))


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    data = _read_status(job_id)
    return JobStatus(**data)


@app.get("/api/jobs/{job_id}/result/{filename}")
async def download_result(job_id: str, filename: str):
    data = _read_status(job_id)
    if data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    # Find the file in result_files
    for rf in data.get("result_files", []):
        if rf["name"] == filename:
            file_path = REAL2SIM_DIR / rf["path"]
            if file_path.exists():
                return FileResponse(
                    str(file_path),
                    filename=filename,
                    media_type="application/octet-stream",
                )
    raise HTTPException(status_code=404, detail=f"File {filename} not found")


@app.get("/api/jobs/{job_id}/log")
async def get_job_log(job_id: str, tail: int = 100):
    log_path = _job_dir(job_id) / "pipeline.log"
    if not log_path.exists():
        return {"log": ""}
    lines = log_path.read_text(errors="replace").splitlines()
    return {"log": "\n".join(lines[-tail:])}


@app.post("/api/jobs/{job_id}/viser/start")
async def start_job_viser(job_id: str):
    data = _read_status(job_id)
    if data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    _start_final_viser(job_id, data)
    return {"url": "/api/viser/"}


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    data = _read_status(job_id)
    if data.get("status") not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in status: {data.get('status')}")

    # Kill any running make process for this job
    try:
        result = subprocess.run(
            ["pkill", "-f", f"make pipeline.*{data.get('video_stem', job_id)}"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        logger.info("[%s] Cancel pkill exited with %s", job_id, result.returncode)
    except Exception as e:
        logger.warning("[%s] Cancel pkill failed: %s", job_id, e)

    _write_status(job_id, {
        **data,
        "status": "failed",
        "error_message": "Cancelled by user",
    })
    return {"detail": "Job cancelled"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("VIDEOMIMIC_PORT", DEFAULT_PORT))
    _cleanup_old_jobs()
    logger.info("Starting VideoMimic API on 0.0.0.0:%s; server log: %s", port, SERVER_LOG_PATH)
    logger.info(
        "VideoMimic log rotation: max_bytes=%s backup_count=%s job_retention_days=%s",
        LOG_MAX_BYTES,
        LOG_BACKUP_COUNT,
        JOB_RETENTION_DAYS,
    )
    logger.info("VideoMimic proxy: %s", PROXY_URL or "disabled")
    logger.info("VideoMimic Viser proxy: /api/viser -> %s", VISER_URL)
    uvicorn.run(app, host="0.0.0.0", port=port)
