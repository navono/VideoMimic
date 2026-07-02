"""Thin entry point kept for `python server/server.py` (legacy invocation).

The package layout lives in siblings (config/app/jobs/runtime/pipeline/...);
this file bootstraps the import path so the legacy command still works, then
defers to `server.app.main()`. Prefer `python -m server` for new code.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `from server.xxx import ...` resolve: real2sim/ must be on sys.path.
_REAL2SIM_DIR = str(Path(__file__).resolve().parent.parent)
if _REAL2SIM_DIR not in sys.path:
    sys.path.insert(0, _REAL2SIM_DIR)

from server.app import main  # noqa: E402  (sys.path bootstrap must precede)

if __name__ == "__main__":
    main()