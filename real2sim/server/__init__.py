"""VideoMimic real2sim HTTP API server package.

分层（行为与单文件 server.py 逐字等价，重构为本布局）：
  config.py          — 路径 / env / STAGES / STAGE_PATTERNS
  logging_setup.py   — RotatingFileHandler + console
  jobs.py            — JobStatus + 文件系统状态 CRUD + new_job_id / safe_stem / collect_results
  runtime.py         — 子进程 / 进程组 / 命令运行器 / stage 检测 / Viser 代理辅助 + limit 兜底
  pipeline.py        — run_pipeline + _stage_rl_lab_input（训练集成）
  app.py             — FastAPI app + 路由 + main
  server.py / __main__.py — 入口（python server/server.py / python -m server）

运行：`cd <real2sim> && python -m server`（Makefile serve target）。
"""