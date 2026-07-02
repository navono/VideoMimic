"""Unit tests for the pure helpers — no subprocess, no GPU.

这些测试锁定 server 各模块的纯函数行为 + FastAPI 路由契约（路径/参数/默认值），
"就地分层重构"前后共用：全绿即代表行为不变。
测试不触碰 subprocess 编排核心（run_pipeline/run_logged_command），那部分
依赖真实 GPU/conda，靠端到端真跑验证。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.config import STAGES, STAGE_PATTERNS
from server.jobs import (
    JobStatus,
    collect_result_files,
    new_job_id,
    safe_stem,
)
from server.pipeline import _height_arg
from server.runtime import (
    detect_stage,
    proxy_headers,
    rewrite_viser_html,
    websocket_subprotocols,
)


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def test_safe_stem_sanitizes_and_falls_back():
    assert safe_stem("foo bar.mp4") == "foo_bar"
    assert safe_stem(None) == "skill_job"
    assert safe_stem("", fallback="x") == "x"
    # 非 [A-Za-z0-9._-] 都替成 _,再 strip 首尾 . _ -
    assert safe_stem("a-b_c.d e!.mp4") == "a-b_c.d_e"


def test_new_job_id_is_safe_and_unique(monkeypatch, tmp_path):
    monkeypatch.setattr("server.jobs.JOBS_DIR", tmp_path)
    monkeypatch.setattr("server.jobs.DEMO_DATA_DIR", tmp_path)  # 让 (DEMO_DATA_DIR/candidate).exists() 也命中 tmp_path(jobs 用的是 jobs 模块级的名字)
    jid = new_job_id("dance clip.mov")
    assert jid.startswith("dance_clip_")
    # 占用后再取应不同(走 uuid 分支)
    (tmp_path / jid).mkdir()
    jid2 = new_job_id("dance clip.mov")
    assert jid2 != jid
    assert jid2.startswith("dance_clip_")


def test_detect_stage_matches_each_stage():
    # STAGE_PATTERNS 覆盖 6 个 stage 的典型 make 输出
    samples = {
        "extracting_frames": "Extracting frames from video",
        "preprocessing": "Step 0: Preprocessing",
        "reconstruction": "Step 1: Reconstruction (MegaSAM)",
        "optimization": "Step 2: Optimization (MegaHunter)",
        "postprocessing": "Step 3: Postprocessing",
        "retargeting": "Step 4: Retargeting (robot_motion_retargeting)",
    }
    for stage, line in samples.items():
        assert detect_stage(line) == stage, f"{stage} 未被识别: {line}"
    assert detect_stage("random unrelated log line") is None


def test_detect_stage_is_case_insensitive():
    assert detect_stage("running megasam reconstruction") == "reconstruction"
    assert detect_stage("RUNNING VIMO") == "preprocessing"


def test_height_arg_accepts_special_and_valid():
    assert _height_arg(-1) == "-1"
    assert _height_arg(0) == "0"
    assert _height_arg(1.8) == "1.8"


def test_height_arg_rejects_invalid():
    with pytest.raises(Exception):  # HTTPException
        _height_arg(0.5)
    with pytest.raises(Exception):
        _height_arg(2.5)


def test_collect_result_files_empty_when_no_dir(tmp_path):
    # video_stem 对应的 demo_data/<stem>/output_calib_mesh 不存在 -> []
    assert collect_result_files("nonexistent_stem_xyz") == []


def test_rewrite_viser_html_rewrites_paths():
    html = b'<a href="/foo">x</a> <script src="/bar.js"></script> <link action="/baz">'
    out = rewrite_viser_html(html)
    assert b'href="/api/viser/foo"' in out
    assert b'src="/api/viser/bar.js"' in out
    assert b'action="/api/viser/baz"' in out


def test_proxy_headers_strips_hop_by_hop():
    h = {
        "connection": "keep-alive",
        "transfer-encoding": "chunked",
        "content-length": "123",
        "x-custom": "keep-me",
        "Authorization": "Bearer x",
    }
    out = proxy_headers(h)
    assert "x-custom" in out
    assert "Authorization" in out
    for hop in ("connection", "transfer-encoding", "content-length"):
        assert hop.lower() not in {k.lower() for k in out}


def test_websocket_subprotocols_parses_header():
    class FakeWS:
        headers = {"sec-websocket-protocol": "a, b ,c"}
    assert websocket_subprotocols(FakeWS()) == ["a", "b", "c"]
    class FakeWSEmpty:
        headers = {}
    assert websocket_subprotocols(FakeWSEmpty()) == []


# --------------------------------------------------------------------------- #
# STAGES / 模型
# --------------------------------------------------------------------------- #


def test_stages_order():
    # 顺序被 _run_logged_command 的 stage_progress_base 依赖(index 必须固定)
    assert STAGES == [
        "extracting_frames",
        "preprocessing",
        "reconstruction",
        "optimization",
        "postprocessing",
        "retargeting",
    ]


def test_stage_patterns_keys_match_stages():
    assert set(STAGE_PATTERNS) == set(STAGES)


def test_job_status_model_defaults():
    s = JobStatus(job_id="x", status="pending")
    assert s.stage is None
    assert s.progress == 0.0
    assert s.result_files == []
    assert s.stage_timestamps == {}


# --------------------------------------------------------------------------- #
# 路由契约(重构必须逐字保留:路径/参数/默认值)
# --------------------------------------------------------------------------- #


def _route_specs():
    """返回 {path: {methods, params}} 便于断言契约。"""
    specs = {}
    for r in app.routes:
        path = getattr(r, "path", None)
        if not path:
            continue
        methods = sorted(getattr(r, "methods", set()) or [])
        # 依赖参数:从 endpoint 签名取 Form/File 的名字与默认值(较脆,这里只断言 path 存在)
        specs[path] = methods
    return specs


def test_routes_present():
    specs = _route_specs()
    # 这些是 benchverse 客户端依赖的路由,重构后必须仍在
    for must in [
        "/api/health",
        "/api/pipeline",
        "/api/jobs/{job_id}",
        "/api/jobs/{job_id}/result/{filename}",
        "/api/jobs/{job_id}/log",
        "/api/jobs/{job_id}/viser/start",
    ]:
        assert must in specs, f"路由丢失: {must}"
    # DELETE /api/jobs/{job_id}
    assert "DELETE" in specs["/api/jobs/{job_id}"]
    # viser 代理路由
    assert any(p.startswith("/api/viser") for p in specs), "Viser 代理路由丢失"


def test_submit_pipeline_defaults_and_required():
    """POST /api/pipeline 默认值: stride=4, height=-1, robot=g1, gender=male; video 必传。"""
    from server.app import submit_pipeline
    import inspect

    sig = inspect.signature(submit_pipeline)

    def _default(p):
        # Form(...) / File(...) 默认值是 fastapi 的 Form/File 对象,真实默认在其 .default
        d = p.default
        return getattr(d, "default", d)

    # video 必传(File(...),无默认值)
    assert sig.parameters["video"].default.__class__.__name__ in ("File", "UploadFile") or sig.parameters["video"].default is ...
    assert _default(sig.parameters["stride"]) == 4, "stride 默认必须是 4(16G 防 OOM)"
    assert _default(sig.parameters["height"]) == -1
    assert _default(sig.parameters["robot"]) == "g1"
    assert _default(sig.parameters["gender"]) == "male"


def test_submit_pipeline_rejects_missing_video():
    """无 video 文件 -> 422(FastAPI 必传字段),不是 500。"""
    with TestClient(app) as client:
        resp = client.post("/api/pipeline", data={"stride": "4"})
    assert resp.status_code == 422


def test_health_ok():
    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_job_404_for_unknown():
    with TestClient(app) as client:
        resp = client.get("/api/jobs/does-not-exist-xyz")
    assert resp.status_code == 404


def test_cancel_unknown_job_404():
    with TestClient(app) as client:
        resp = client.delete("/api/jobs/does-not-exist-xyz")
    # _read_status 抛 404 在 cancel 之前
    assert resp.status_code == 404