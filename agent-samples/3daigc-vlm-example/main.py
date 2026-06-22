# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
3daigc-vlm-example orchestrator — voice VLM plus 3DAIGC mesh generation.

Prerequisites (start before this sample):
  - 3DAIGC-API on :7842
  - 3daigc-mcp-http on :8260  (bash /home/sifr/3DAIGC-API/mcp/scripts/run_http.sh)

How to run (from agent-samples/3daigc-vlm-example/):
    uv sync && uv run daigc_vlm_example

Or use the one-command stack script on DGX:
    bash /home/sifr/3DAIGC-API/mcp/scripts/run_xr_ai_3daigc_stack.sh

Open https://<host>:8088, start mic, say e.g. "make a 3D model of this".
"""
import re
from pathlib import Path

from xr_ai_launcher import Process, ensure_credentials, run_stack, warn_if_missing
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
_WORKER_CONFIG = "yaml/3daigc_vlm_example_worker.yaml"
_DAIGC_MCP_PROJECT = Path("/home/sifr/3DAIGC-API/mcp")

_BACKEND_RE = re.compile(r"^\s*model_backend\s*:\s*[\"']?(\w+)[\"']?", re.MULTILINE)


def _model_backend() -> str:
    try:
        m = _BACKEND_RE.search((_BASE / _WORKER_CONFIG).read_text())
    except OSError:
        return "local"
    return m.group(1).lower() if m else "local"


def _build_processes(backend: str) -> list[Process]:
    procs = [
        Process(
            "3daigc-mcp",
            _DAIGC_MCP_PROJECT,
            "3daigc-mcp-http",
            launch_mode="reuse",
            port=8260,
        ),
        Process("hub", "../../server-runtime", "xr_media_hub",
                config="yaml/xr_media_hub.yaml"),
    ]
    if backend != "nim":
        procs.append(
            Process("vlm", "../../ai-services/vlm-server", "vlm_server",
                    config="yaml/vlm_server.yaml"),
        )
    procs += [
        Process("stt", "../../ai-services/stt-server", "stt_server",
                config="yaml/stt_server.yaml"),
        Process("tts", "../../ai-services/tts/piper", "piper_tts_server",
                config="yaml/piper_tts_server.yaml"),
        Process("worker", "worker", "daigc_vlm_example_worker",
                config=_WORKER_CONFIG),
    ]
    return procs


def run() -> None:
    setup_logging("orchestrator", namespace="3daigc-vlm-example")
    backend = _model_backend()
    warn_if_missing("HF_TOKEN")
    if backend == "nim":
        ensure_credentials("NGC_API_KEY")
    run_stack(_build_processes(backend), _BASE)


if __name__ == "__main__":
    run()
