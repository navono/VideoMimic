"""FastAPI app: routes + main(). Behaves 1:1 like the single-file server.py.

Routes/params/defaults preserved exactly (benchverse client contract):
  GET  /api/health
  ANY  /api/viser, /api/viser/{path}            (HTTP proxy to transient viser)
  WS   /api/viser, /api/viser/{path}            (WebSocket proxy)
  POST /api/pipeline   video(!) stride=4 height=-1 robot=g1 gender=male
  GET  /api/jobs/{job_id}
  GET  /api/jobs/{job_id}/result/{filename}
  GET  /api/jobs/{job_id}/log
  POST /api/jobs/{job_id}/viser/start
  DELETE /api/jobs/{job_id}
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .config import (
    DEFAULT_PORT,
    REAL2SIM_DIR,
    VISER_URL,
)
from .jobs import (
    JobStatus,
    cleanup_old_jobs,
    new_job_id,
    read_status,
    safe_stem,
    write_status,
)
from .logging_setup import setup_logging
from .pipeline import _height_arg, run_pipeline
from .runtime import (
    proxy_headers,
    rewrite_viser_html,
    start_final_viser,
    terminate_job_processes,
    websocket_subprotocols,
)

logger = setup_logging()

app = FastAPI(title="VideoMimic Real2Sim API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Viser proxy (HTTP + WebSocket)
# --------------------------------------------------------------------------- #


@app.api_route("/api/viser", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/api/viser/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_viser(request: Request, path: str = ""):
    """Proxy the transient Real2Sim Viser UI through this API server."""
    target = f"{VISER_URL.rstrip('/')}/{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client:
            upstream = await client.request(
                request.method,
                target,
                headers=proxy_headers(request.headers),
                content=await request.body(),
            )
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail="Viser UI is not running") from exc

    content = upstream.content
    headers = proxy_headers(upstream.headers)
    content_type = upstream.headers.get("content-type", "")
    if "text/html" in content_type:
        content = rewrite_viser_html(content)
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

    subprotocols = websocket_subprotocols(websocket)
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Viser websocket proxy closed: %s", exc)


# --------------------------------------------------------------------------- #
# Pipeline submission
# --------------------------------------------------------------------------- #


@app.post("/api/pipeline", response_model=JobStatus)
async def submit_pipeline(
    video: UploadFile = File(...),  # noqa: B008
    stride: int = Form(4),  # noqa: B008  # 16G 显存防 OOM：默认每 4 帧取 1 帧
    height: float = Form(-1),  # noqa: B008
    robot: str = Form("g1"),  # noqa: B008
    gender: str = Form("male"),  # noqa: B008
):
    """Upload a video and start the real2sim pipeline."""
    from .jobs import job_dir

    cleanup_old_jobs()

    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    # 早期校验 height（_height_arg 抛 400）
    _height_arg(height)

    original_video_stem = safe_stem(video.filename, fallback="real2sim_job")
    job_id = new_job_id(video.filename)
    # VideoMimic's Makefile writes to demo_data/$(VID_STEM). Use the unique
    # job id so repeated uploads of the same filename cannot reuse old outputs.
    video_stem = job_id
    jdir = job_dir(job_id)
    input_dir = jdir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_filename = f"{video_stem}{Path(video.filename).suffix}"
    video_path = input_dir / video_filename
    content = await video.read()
    video_path.write_bytes(content)
    logger.info(
        "[%s] Received video %s (%s bytes), stride=%s height=%s robot=%s gender=%s",
        job_id, video.filename, len(content), stride, height, robot, gender,
    )

    # Create initial status
    now = datetime.now(UTC).isoformat()
    write_status(job_id, {
        "job_id": job_id,
        "status": "pending",
        "stage": None,
        "progress": 0.0,
        "error_message": None,
        "video_stem": video_stem,
        "original_video_stem": original_video_stem,
        "video_filename": video_filename,
        "original_video_filename": video.filename,
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

    return JobStatus(**read_status(job_id))


# --------------------------------------------------------------------------- #
# Job status / results / log / cancel
# --------------------------------------------------------------------------- #


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    data = read_status(job_id)
    return JobStatus(**data)


@app.get("/api/jobs/{job_id}/result/{filename}")
async def download_result(job_id: str, filename: str):
    data = read_status(job_id)
    if data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
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
    from .jobs import job_dir
    log_path = job_dir(job_id) / "pipeline.log"
    if not log_path.exists():
        return {"log": ""}
    lines = log_path.read_text(errors="replace").splitlines()
    return {"log": "\n".join(lines[-tail:])}


@app.post("/api/jobs/{job_id}/viser/start")
async def start_job_viser(job_id: str):
    data = read_status(job_id)
    if data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    start_final_viser(job_id, data)
    return {"url": "/api/viser/"}


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    data = read_status(job_id)
    if data.get("status") == "cancelled":
        return {"detail": "Job already cancelled"}
    if data.get("status") not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in status: {data.get('status')}")

    write_status(job_id, {
        **data,
        "status": "cancelled",
        "error_message": "Cancelled by user",
    })
    await terminate_job_processes(job_id, data)
    return {"detail": "Job cancelled"}


def main():
    """CLI entry point for `python -m server` and `python server/server.py`."""
    import uvicorn
    from .config import JOB_RETENTION_DAYS, LOG_BACKUP_COUNT, LOG_MAX_BYTES, SERVER_LOG_PATH
    port = int(os.environ.get("VIDEOMIMIC_PORT", DEFAULT_PORT))
    cleanup_old_jobs()
    logger.info("Starting VideoMimic API on 0.0.0.0:%s; server log: %s", port, SERVER_LOG_PATH)
    logger.info(
        "VideoMimic log rotation: max_bytes=%s backup_count=%s job_retention_days=%s",
        LOG_MAX_BYTES, LOG_BACKUP_COUNT, JOB_RETENTION_DAYS,
    )
    logger.info("VideoMimic proxy: %s", os.environ.get("VIDEOMIMIC_PROXY", "http://127.0.0.1:18899") or "disabled")
    logger.info("VideoMimic Viser proxy: /api/viser -> %s", VISER_URL)
    uvicorn.run(app, host="0.0.0.0", port=port)