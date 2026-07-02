"""Subprocess / process-group / command-runner / stage-detection / Viser proxy helpers.

Houses everything that spawns or signals OS processes, the logged command
runner (with stage progression), cancellation, and the Viser HTTP/WS proxy
helpers. Behaves 1:1 like the single-file server.py, with one fix: the stdout
read loop now uses readline() with a LimitOverrunError fallback and a 64 MB
StreamReader limit, so megasam's multi-MB single-line weight-mismatch warnings
no longer kill the task with "Separator is not found, and chunk exceed the
limit".
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess

from fastapi import WebSocket

from .config import (
    CONDA_EVAL,
    CONDA_VM1RS,
    PROXY_URL,
    REAL2SIM_DIR,
    STAGES,
    STAGE_PATTERNS,
    VISER_PORT,
)
from .jobs import (
    append_job_log,
    is_cancelled,
    mark_stage,
    read_status,
    write_status,
)
from .logging_setup import setup_logging

logger = setup_logging()

# Active subprocess group leaders by job. Each pipeline command is launched in
# its own process group so cancellation can terminate the shell, make, and all
# Python children instead of only matching one command line with pkill.
RUNNING_PROCESS_GROUPS: dict[str, set[int]] = {}


def detect_stage(line: str) -> str | None:
    for stage, pattern in STAGE_PATTERNS.items():
        if re.search(pattern, line, re.IGNORECASE):
            return stage
    return None


def register_process_group(job_id: str, pid: int) -> None:
    RUNNING_PROCESS_GROUPS.setdefault(job_id, set()).add(pid)
    try:
        data = read_status(job_id)
        write_status(job_id, {**data, "process_groups": sorted(RUNNING_PROCESS_GROUPS[job_id])})
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Failed to record process group %s: %s", job_id, pid, exc)


def unregister_process_group(job_id: str, pid: int) -> None:
    groups = RUNNING_PROCESS_GROUPS.get(job_id)
    if not groups:
        return
    groups.discard(pid)
    if groups:
        RUNNING_PROCESS_GROUPS[job_id] = groups
    else:
        RUNNING_PROCESS_GROUPS.pop(job_id, None)


def _terminate_process_group(job_id: str, pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
        logger.info("[%s] Sent %s to process group %s", job_id, sig.name, pgid)
    except ProcessLookupError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Failed to send %s to process group %s: %s", job_id, sig.name, pgid, exc)


async def terminate_job_processes(job_id: str, data: dict | None = None) -> None:
    from .jobs import job_dir, read_status as _rs  # noqa: F401 (kept for parity)
    data = data or read_status(job_id)
    process_groups = set(RUNNING_PROCESS_GROUPS.get(job_id, set()))
    process_groups.update(int(p) for p in data.get("process_groups", []) if str(p).isdigit())

    for pgid in process_groups:
        _terminate_process_group(job_id, pgid, signal.SIGTERM)

    # Fallback for jobs launched before process-group tracking, or for children
    # whose command line still carries the job-specific paths.
    patterns = {
        job_id,
        data.get("video_stem"),
        data.get("video_filename"),
        str(job_dir(job_id)),
    }
    for pattern in sorted(p for p in patterns if p):
        try:
            result = subprocess.run(
                ["pkill", "-TERM", "-f", pattern],
                capture_output=True, check=False, timeout=5,
            )
            if result.returncode == 0:
                logger.info("[%s] Fallback pkill TERM matched pattern %s", job_id, pattern)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Fallback pkill TERM failed for %s: %s", job_id, pattern, exc)

    await asyncio.sleep(1.0)

    for pgid in process_groups:
        _terminate_process_group(job_id, pgid, signal.SIGKILL)
    for pattern in sorted(p for p in patterns if p):
        try:
            subprocess.run(
                ["pkill", "-KILL", "-f", pattern],
                capture_output=True, check=False, timeout=5,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Fallback pkill KILL failed for %s: %s", job_id, pattern, exc)


def start_final_viser(job_id: str, data: dict) -> None:
    from .jobs import job_dir, postprocessed_dir_from_status
    postprocessed_dir = postprocessed_dir_from_status(data)
    if not postprocessed_dir or not postprocessed_dir.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No completed visualization directory found")

    subprocess.run(
        ["pkill", "-f", "complete_results_egoview_visualization.py"],
        capture_output=True, check=False, timeout=5,
    )

    log_path = job_dir(job_id) / "viser.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    robot = data.get("robot") or "g1"
    is_megasam_flag = "--is-megasam" if data.get("reconstruction_method", "megasam") == "megasam" else ""
    command = (
        f'{CONDA_EVAL} && conda activate {CONDA_VM1RS} && '
        f'python visualization/complete_results_egoview_visualization.py '
        f'--postprocessed-dir "{postprocessed_dir}" '
        f'--robot-name "{robot}" {is_megasam_flag} --port {VISER_PORT}'
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


async def run_logged_command(
    job_id: str,
    label: str,
    command: str,
    *,
    stage_start_index: int,
) -> tuple[int, list[str]]:
    logger.info("[%s] Starting %s", job_id, label)
    logger.info("[%s] Command: %s", job_id, command)
    append_job_log(job_id, f"$ {command}")

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
        start_new_session=True,
        # megasam 加载大模型时打出超长单行（权重 mismatch 列表，单行可达数 MB），
        # asyncio StreamReader 默认 limit=64KB，readline() 遇无换行超长行抛
        # LimitOverrunError("Separator is not found, and chunk exceed the limit") →
        # task 误判 failed（子进程其实还活着）。提到 64MB 并加 readline 兜底。
        limit=64 * 1024 * 1024,
    )
    register_process_group(job_id, proc.pid)

    if proc.stdout is None:
        raise RuntimeError(f"{label} did not expose stdout")

    current_stage_idx = stage_start_index
    stage_progress_base = {0: 0.0, 1: 0.15, 2: 0.30, 3: 0.50, 4: 0.70, 5: 0.85}
    tail_lines: list[str] = []

    try:
        # 用 readline() 而非 `async for proc.stdout`：后者遇超长无换行行会抛
        # LimitOverrunError 直接终止读取。这里显式捕获：超长行读出已缓冲部分
        # 记一行后继续，绝不让一行日志输出搞垮整个 stage。
        while True:
            try:
                line_bytes = await proc.stdout.readline()
            except asyncio.LimitOverrunError as e:
                chunk = await proc.stdout.read(e.consumed)
                line = chunk.decode(errors="replace").rstrip()
                tail_lines.append(line)
                tail_lines = tail_lines[-200:]
                logger.info("[%s] %s", job_id, line)
                append_job_log(job_id, line)
                continue
            if not line_bytes:
                break
            if is_cancelled(job_id):
                append_job_log(job_id, f"{label} cancellation requested; terminating process group")
                await terminate_job_processes(job_id)
                break

            line = line_bytes.decode(errors="replace").rstrip()
            tail_lines.append(line)
            tail_lines = tail_lines[-200:]

            logger.info("[%s] %s", job_id, line)
            append_job_log(job_id, line)

            detected = detect_stage(line)
            if detected:
                stage_idx = STAGES.index(detected) if detected in STAGES else current_stage_idx
                if stage_idx > current_stage_idx:
                    current_stage_idx = stage_idx
                    progress = stage_progress_base.get(stage_idx, 0.9)
                    mark_stage(job_id, detected, progress)
                    logger.info("[%s] Stage changed to %s (progress %.2f)", job_id, detected, progress)

        await proc.wait()
        logger.info("[%s] Finished %s with exit code %s", job_id, label, proc.returncode)
        append_job_log(job_id, f"{label} exited with code {proc.returncode}")
        return proc.returncode or 0, tail_lines
    finally:
        if is_cancelled(job_id) and proc.returncode is None:
            await terminate_job_processes(job_id)
            await proc.wait()
        unregister_process_group(job_id, proc.pid)


# --------------------------------------------------------------------------- #
# Viser proxy helpers (HTTP + WebSocket, HTML rewrite)
# --------------------------------------------------------------------------- #


def proxy_headers(headers) -> dict[str, str]:
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


def rewrite_viser_html(content: bytes) -> bytes:
    text = content.decode("utf-8", errors="replace")
    text = text.replace('href="/', 'href="/api/viser/')
    text = text.replace('src="/', 'src="/api/viser/')
    text = text.replace('action="/', 'action="/api/viser/')
    text = text.replace('url(/', 'url(/api/viser/')
    return text.encode("utf-8")


def websocket_subprotocols(websocket: WebSocket) -> list[str]:
    value = websocket.headers.get("sec-websocket-protocol", "")
    return [item.strip() for item in value.split(",") if item.strip()]