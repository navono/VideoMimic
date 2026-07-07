"""端到端冒烟测试:真实启动 HTTP 服务,跑完整 real2sim 流水线。

区别于 test_helpers.py(纯函数 + mock run_pipeline,不碰 GPU/conda),本测试:
  - 真起 uvicorn(`python -m server`,随机端口)
  - 真上传 assets/talented-urban-woman-dancing.mp4 + device=6
  - 真跑 run_pipeline → make pipeline 全流程(SAM2/ViTPose/MegaSAM/MegaHunter/postprocess/retarget)
  - 轮询 /api/tasks/{id} 直到 completed/failed
  - 断言 device 全链路流转:status.json.device == "6"、pipeline.log 里 make 命令含 DEVICE=6、
    完成则 retarget_poses_g1.h5 存在

前置条件重(GPU + vm1rs/vm1recon + 模型权重 + ffprobe),耗时数十分钟,故默认 skip:
  RUN_E2E=1 python -m pytest server/tests/test_device_e2e.py -s
或加 marker:python -m pytest -m e2e -s

环境旋钮(均有默认):
  E2E_DEVICE        GPU 编号,默认 6
  E2E_SUBSAMPLE     subsample_factor,默认 4(控制耗时;16GB 卡可降到 2)
  E2E_VIDEO         输入视频,默认 assets/talented-urban-woman-dancing.mp4
  E2E_TIMEOUT       单任务超时秒,默认 1800(30min)
  E2E_PORT          固定端口(默认随机挑一个空闲的)
  VIDEOMIMIC_PROXY  代理;默认置空(靠本地 HF cache + --localhub,避免死代理 18899)

已知阻塞(2026-07-07,均已修):
  - wandb 0.18.7 + protobuf 5.x 冲突 → UniDepth import wandb 即 ImportError。
    修:visualization.py 加 try/except stub(推理不用 wandb)。
  - xformers 0.0.22 无 CUDA op → DINOv2 memory_efficient_attention NotImplementedError。
    修:UniDepth attention.py 加 CUDA 探测回退;Makefile 设 XFORMERS_DISABLED=1
    让官方 dinov2 走 torch SDPA。
  - torchhub/facebookresearch_dinov2_main 空目录 → hubconf.py 缺失。修:git clone。
  见 memory: real2sim-install-pitfalls 第 17-19 条。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# real2sim 根目录(server/tests/ 的上两级)
REAL2SIM_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = REAL2SIM_DIR / "assets" / "talented-urban-woman-dancing.mp4"


# --------------------------------------------------------------------------- #
# markers / skip 门槛
# --------------------------------------------------------------------------- #

e2e = pytest.mark.e2e
skip_unless_run = pytest.mark.skipif(
    not os.environ.get("RUN_E2E"),
    reason="E2E 默认 skip:需 GPU+权重+conda,设 RUN_E2E=1 启用(耗时数十分钟)",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(base: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/api/health", timeout=5.0)
            if r.status_code == 200 and r.json() == {"status": "ok"}:
                return
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(0.5)
    raise RuntimeError(f"server 未在 {timeout}s 内健康启动: {last}")


@pytest.fixture(scope="module")
def server_url() -> str:
    """真起 uvicorn HTTP 服务,跑完整 module 后关停。"""
    if not DEFAULT_VIDEO.exists():
        pytest.skip(f"测试视频不存在: {DEFAULT_VIDEO}")
    port = int(os.environ.get("E2E_PORT") or _free_port())
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "VIDEOMIMIC_PORT": str(port),
        # 默认禁用死代理 18899,靠本地 HF cache + --localhub;要联网下载则显式设 VIDEOMIMIC_PROXY
        "VIDEOMIMIC_PROXY": os.environ.get("VIDEOMIMIC_PROXY", ""),
        "PYTHONUNBUFFERED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "server"],
        cwd=str(REAL2SIM_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_health(base)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _submit_task(
    base: str,
    video: Path,
    *,
    device: str,
    subsample: int,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict:
    """POST /api/tasks 上传视频,返回响应 JSON(含 task_id/job_id)。"""
    with video.open("rb") as f:
        files = {"video": (video.name, f, "video/mp4")}
        data: dict[str, str] = {
            "subsample_factor": str(subsample),
            "robot_name": "g1",
            "height": "-1",
            "reconstruction_method": "megasam",
            "device": device,
        }
        if start_frame is not None:
            data["start_frame"] = str(start_frame)
        if end_frame is not None:
            data["end_frame"] = str(end_frame)
        r = httpx.post(f"{base}/api/tasks", files=files, data=data, timeout=60.0)
    assert r.status_code == 200, f"提交失败 {r.status_code}: {r.text}"
    return r.json()


def _poll_until_terminal(
    base: str, task_id: str, *, timeout: float, poll: float = 5.0
) -> dict:
    """轮询 /api/tasks/{id} 直到 status ∈ {completed, failed, cancelled}。

    GET read timeout 放宽到 120s:server 单线程 uvicorn,跑 MegaSAM 时 event loop 被
    子进程密集日志读取占用,GET 响应可能延迟数十秒。用持久 client 避免反复建连,
    read 超时则继续轮询(server 还活着,只是忙)。
    """
    deadline = time.monotonic() + timeout
    last: dict = {}
    with httpx.Client(timeout=httpx.Timeout(10.0, read=120.0, write=120.0, pool=30.0)) as cli:
        while time.monotonic() < deadline:
            try:
                r = cli.get(f"{base}/api/tasks/{task_id}")
            except httpx.ReadTimeout:
                continue  # event loop 被 MegaSAM 占着,读超时但 server 还活着
            assert r.status_code == 200, f"查状态失败 {r.status_code}: {r.text}"
            last = r.json()
            if last.get("status") in ("completed", "failed", "cancelled"):
                return last
            time.sleep(poll)
    # 超时:返回最后状态,让调用方打印 + 断言失败
    return last


def _status_json(task_id: str) -> dict:
    """直接读 JOBS_DIR/{task_id}/status.json(绕过 JobStatus 模型,拿 device 等额外字段)。"""
    p = REAL2SIM_DIR / "server" / "jobs" / task_id / "status.json"
    return json.loads(p.read_text())


def _pipeline_log(task_id: str) -> str:
    p = REAL2SIM_DIR / "server" / "jobs" / task_id / "pipeline.log"
    return p.read_text(errors="replace") if p.exists() else ""


def _wait_for_log(task_id: str, needle: str, *, timeout: float = 120.0) -> str:
    """轮询 pipeline.log 直到出现 needle(extract 阶段先跑,make_cmd 后写)。"""
    deadline = time.monotonic() + timeout
    log = ""
    while time.monotonic() < deadline:
        log = _pipeline_log(task_id)
        if needle in log:
            return log
        time.sleep(1.0)
    return log  # 超时返回最后内容,让调用方断言失败时打印


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #


@e2e
@skip_unless_run
def test_device_e2e_full_pipeline(server_url: str) -> None:
    """device=6 全链路:上传 → make pipeline 含 DEVICE=6 → 完成 → retarget h5 存在。

    这是端到端冒烟:真起服务、真跑 GPU 流水线。耗时长,需 RUN_E2E=1。
    """
    device = os.environ.get("E2E_DEVICE", "6")
    subsample = int(os.environ.get("E2E_SUBSAMPLE", "4"))
    timeout = float(os.environ.get("E2E_TIMEOUT", "1800"))

    # 1) 提交(只跑前 24 帧 ≈ 1s @23.976fps,够触发全 stage 又控制耗时)
    body = _submit_task(
        server_url, DEFAULT_VIDEO,
        device=device, subsample=subsample,
        start_frame=0, end_frame=23,
    )
    task_id = body["task_id"]
    assert body["status"] == "pending"

    # 2) device 已落 status.json(立即校验,不等跑完)
    st = _status_json(task_id)
    assert st["device"] == device, f"status.json device 期望 {device!r},实得 {st.get('device')!r}"

    # 3) make 命令含 DEVICE=6(轮询等待:extract 阶段先跑,make_cmd 在 extract 完成后才写入日志)
    log = _wait_for_log(task_id, f" DEVICE={device}", timeout=180.0)
    assert f" DEVICE={device}" in log, (
        f"pipeline.log 未含 ' DEVICE={device}';make_cmd 没把 device 传下去。\n日志尾部:\n"
        + "\n".join(log.splitlines()[-15:])
    )

    # 4) 轮询到终态
    final = _poll_until_terminal(server_url, task_id, timeout=timeout)
    status = final.get("status")
    if status != "completed":
        # 失败也要打印日志尾部,方便定位(权重缺失 / OOM / proxy 死 等)
        log = _pipeline_log(task_id)
        pytest.fail(
            f"任务未完成(status={status}): {final.get('error_message')}\n"
            f"=== pipeline.log 尾部 ===\n" + "\n".join(log.splitlines()[-40:])
        )

    # 5) 完成则核心产物 retarget_poses_g1.h5 必须存在
    calib = REAL2SIM_DIR / "demo_data" / task_id / "output_calib_mesh"
    h5s = sorted(calib.glob("*/retarget_poses_g1.h5"))
    assert h5s, f"完成但找不到 retarget_poses_g1.h5 于 {calib}"
    assert h5s[-1].stat().st_size > 0, f"retarget h5 空文件: {h5s[-1]}"

    # 6) result_files 应含该 h5(collect_result_files 扫 output_calib_mesh)
    result_names = {rf["name"] for rf in final.get("result_files", [])}
    assert "retarget_poses_g1.h5" in result_names, (
        f"result_files 未含 retarget_poses_g1.h5: {result_names}"
    )


@e2e
@skip_unless_run
def test_device_empty_means_unrestricted(server_url: str) -> None:
    """device=""(空,不限)→ status.json.device == "" 且 make 命令含 ' DEVICE='(尾部空)。

    不跑完整流水线(避免双倍耗时);只校验 device 流转的前半段(提交 + status + make_cmd),
    因为 device="" 的语义在 make_cmd 构造时就已体现,无需等 GPU 跑完。
    """
    subsample = int(os.environ.get("E2E_SUBSAMPLE", "4"))

    body = _submit_task(
        server_url, DEFAULT_VIDEO,
        device="", subsample=subsample,
        start_frame=0, end_frame=3,
    )
    task_id = body["task_id"]

    st = _status_json(task_id)
    assert st["device"] == "", f"空 device 期望持久化为 '',实得 {st.get('device')!r}"

    # 等 make_cmd 写入(extract 先跑),再取;empty 时 DEVICE= 后面接换行/空格,不是数字
    log = _wait_for_log(task_id, " DEVICE=", timeout=180.0)
    assert " DEVICE=" in log, f"pipeline.log 未含 ' DEVICE=';\n{log[-500:]}"
    # 确认是空值:DEVICE= 后面不能跟数字(否则就是 device=6 之类)
    import re
    m = re.search(r" DEVICE=(\S*)", log)
    assert m is not None and m.group(1) == "", (
        f"device='' 应产出 ' DEVICE='(尾部空),实得 DEVICE={m.group(1) if m else '?'}\n{log[-500:]}"
    )

    # 顺手取消,避免后台继续跑空 device 的全流程
    try:
        httpx.delete(f"{server_url}/api/tasks/{task_id}", timeout=15.0)
    except Exception:  # noqa: BLE001
        pass
