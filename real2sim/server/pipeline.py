"""Pipeline orchestration: run_pipeline (extract -> make pipeline -> collect) +
_stage_rl_lab_input (copy retarget h5 to benchverse jobs/ + convert to motion.npz).

Mirrors the single-file server.py 1:1, preserving the `make pipeline` reuse
(over VideoMimic's Makefile, with STRIDE/HEIGHT/ROBOT/GENDER/PROXY) and the
post-pipeline RL-training integration.
"""

from __future__ import annotations

import asyncio
import os
import shutil

from fastapi import HTTPException

from .config import (
    BENCHVERSE_JOBS_DIR,
    CONDA_EVAL,
    CONDA_VM1RS,
    DEMO_DATA_DIR,
    PROXY_URL,
    RL_LAB_DIR,
    REAL2SIM_DIR,
)
from .jobs import (
    append_job_log,
    collect_result_files,
    is_cancelled,
    mark_stage,
    read_status,
    write_status,
)
from .logging_setup import setup_logging
from .runtime import run_logged_command

logger = setup_logging()


def _height_arg(height: float) -> str:
    if height == -1:
        return "-1"
    if height == 0:
        return "0"
    if not 1.0 <= height <= 2.2:
        raise HTTPException(status_code=400, detail="height must be -1, 0, or between 1.0 and 2.2 meters")
    return f"{height:g}"


async def stage_rl_lab_input(job_id: str, video_filename: str, video_stem: str) -> dict:
    """real2sim 成功后：把 retarget h5 落到 benchverse jobs/ 并转成 motion.npz（RL 训练输入）。

    best-effort 分两步：
      1) 拷 h5 到 benchverse jobs/<job_id>/input/ —— 纯文件操作，不依赖 env_isaaclab。
         成功即写入 status['rl_lab']['h5_staged']，后续训练/部署都能拿到。
      2) convert h5 -> motion.npz（调 unitree_rl_lab 的 make convert-videomimic，需 env_isaaclab）。
         失败只记 convert_error，不影响 real2sim job 翻车，也不掩盖第 1 步的成果。
    """
    result: dict = {"job_dir": None, "h5_staged": False, "motion_npz": None, "convert_error": None}

    # ---- Step 1: 拷 h5（必须成功，否则后续无从谈起）----
    try:
        h5s = sorted((DEMO_DATA_DIR / video_stem / "output_calib_mesh").glob("*/retarget_poses_g1.h5"))
        if not h5s:
            raise RuntimeError("retarget_poses_g1.h5 not found under output_calib_mesh")
        h5_src = h5s[-1]

        bv_job = BENCHVERSE_JOBS_DIR / job_id
        (bv_job / "input").mkdir(parents=True, exist_ok=True)
        h5_dst = bv_job / "input" / "retarget_poses_g1.h5"
        shutil.copy2(h5_src, h5_dst)
        result["job_dir"] = str(bv_job)
        result["h5_staged"] = True
        logger.info("[%s] staged retarget h5 -> %s", job_id, h5_dst)
        append_job_log(job_id, f"staged retarget h5 -> {h5_dst}")
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] rl_lab stage h5 failed: %s", job_id, e)
        result["stage_error"] = str(e)
        result["status"] = "error"
        return result

    # ---- Step 2: convert h5 -> motion.npz（best-effort，失败不掩盖 h5 已拷）----
    try:
        npz_dst = bv_job / "motion.npz"
        cmd = (
            f'make -C "{RL_LAB_DIR}" convert-videomimic '
            f'VM_H5="{h5_dst}" CONVERT_NPZ="{npz_dst}"'
        )
        logger.info("[%s] rl_lab convert: %s", job_id, cmd)
        append_job_log(job_id, f"$ {cmd}")
        proc = await asyncio.create_subprocess_shell(
            cmd, executable="/bin/bash",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            cwd=str(REAL2SIM_DIR), env={**os.environ, "PYTHONUNBUFFERED": "1"},
            limit=64 * 1024 * 1024,
        )
        tail: list[str] = []
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            tail.append(line)
            tail = tail[-200:]
            logger.info("[%s] %s", job_id, line)
            append_job_log(job_id, line)
        await proc.wait()
        rc = proc.returncode or 0
        if rc != 0:
            raise RuntimeError(f"convert-videomimic exit {rc}: {' | '.join(tail[-10:])}")
        result["motion_npz"] = str(npz_dst)
        result["status"] = "ok"
        logger.info("[%s] rl_lab convert ok -> %s", job_id, npz_dst)
    except Exception as e:  # noqa: BLE001
        # convert 失败：h5 已拷成功（第1步），记 convert_error 但不判整个 rl_lab 为 error
        logger.exception("[%s] rl_lab convert failed (h5 already staged): %s", job_id, e)
        result["convert_error"] = str(e)
        result["status"] = "partial"  # h5 已就位，npz 待 5051/有 env_isaaclab 时再转
    return result


async def run_pipeline(
    job_id: str, video_stem: str, stride: int, height: float, robot: str, gender: str,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    reconstruction_method: str = "megasam",
) -> None:
    """Execute the full real2sim pipeline for a job, updating status along the way."""
    from .jobs import job_dir
    height_value = _height_arg(height)

    try:
        logger.info(
            "[%s] Pipeline submitted: video=%s stride=%s height=%s robot=%s gender=%s "
            "start_frame=%s end_frame=%s reconstruction_method=%s",
            job_id, video_stem, stride, height, robot, gender,
            start_frame, end_frame, reconstruction_method,
        )
        write_status(job_id, {**read_status(job_id), "status": "running"})
        mark_stage(job_id, "extracting_frames", 0.0)

        # Extract frames first
        cam_dir = DEMO_DATA_DIR / video_stem / "input_images" / "cam01"
        video_src = job_dir(job_id) / "input" / read_status(job_id).get("video_filename", "video.mp4")

        # Step 0: Extract frames
        # None = 抽全帧（交给 Makefile 自动检测处理范围）
        sf = 0 if start_frame is None else start_frame
        ef = 99999 if end_frame is None else end_frame
        extract_cmd = (
            f'{CONDA_EVAL} && conda activate {CONDA_VM1RS} && '
            f'python utilities/extract_frames_from_video.py '
            f'--video-path "{video_src}" '
            f'--output-dir "{cam_dir}" '
            f'--start-frame {sf} --end-frame {ef}'
        )
        exit_code, tail_lines = await run_logged_command(
            job_id, "frame extraction", extract_cmd, stage_start_index=0,
        )
        if is_cancelled(job_id):
            logger.info("[%s] Pipeline cancelled during frame extraction", job_id)
            return
        if exit_code != 0:
            raise RuntimeError(f"Frame extraction failed: {' | '.join(tail_lines[-20:])}")

        mark_stage(job_id, "preprocessing", 0.15)

        # Build make pipeline command
        is_megasam = "1" if reconstruction_method == "megasam" else "0"
        # start_frame/end_frame 为 None 时不传，交给 Makefile 的 ?= 自动检测（ls cam_dir 首尾帧）
        frame_args = ""
        if start_frame is not None:
            frame_args += f" START_FRAME={start_frame}"
        if end_frame is not None:
            frame_args += f" END_FRAME={end_frame}"
        make_cmd = (
            f'{CONDA_EVAL} && '
            f'export HF_TOKEN=${{HF_TOKEN:-}} && '
            f'make pipeline VIDEO_PATH="{video_src}" VID_STEM="{video_stem}" STRIDE={stride} HEIGHT={height_value} '
            f'ROBOT={robot} GENDER={gender} PROXY="{PROXY_URL}"{frame_args} IS_MEGASAM={is_megasam}'
        )

        exit_code, tail_lines = await run_logged_command(
            job_id, "real2sim pipeline", make_cmd, stage_start_index=1,
        )
        if is_cancelled(job_id):
            logger.info("[%s] Pipeline cancelled during real2sim pipeline", job_id)
            return
        if exit_code != 0:
            raise RuntimeError(f"Pipeline failed (exit {exit_code}): {' | '.join(tail_lines[-20:])}")

        # Collect results
        result_files = collect_result_files(video_stem)
        write_status(job_id, {
            **read_status(job_id),
            "status": "completed",
            "stage": "retargeting",
            "progress": 1.0,
            "result_files": result_files,
        })
        logger.info("[%s] Pipeline completed with %s result files", job_id, len(result_files))

        # Post-pipeline：落 h5 到 benchverse jobs/ 并转 motion.npz（RL 训练输入）
        rl_lab = await stage_rl_lab_input(
            job_id,
            read_status(job_id).get("video_filename", f"{video_stem}.mp4"),
            video_stem,
        )
        write_status(job_id, {**read_status(job_id), "rl_lab": rl_lab})

    except Exception as e:
        if is_cancelled(job_id):
            logger.info("[%s] Pipeline stopped after cancellation", job_id)
            return
        logger.exception("[%s] Pipeline failed: %s", job_id, e)
        write_status(job_id, {
            **read_status(job_id),
            "status": "failed",
            "error_message": str(e),
        })