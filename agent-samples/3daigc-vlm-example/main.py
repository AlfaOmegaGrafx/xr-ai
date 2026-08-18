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
    bash /home/sifr/3DAIGC-API/mcp/scripts/start_xr_voice_full.sh

Open https://<host>:8088, start mic, say e.g. "make a 3D model of this".
"""
import argparse
import os
from dataclasses import replace
from pathlib import Path

from xr_ai_launcher import (
    Process,
    ensure_credentials,
    load_model_deployment,
    require_credentials,
    run_stack,
)
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
_WORKER_CONFIG = "yaml/3daigc_vlm_example_worker.yaml"
_DAIGC_MCP_PROJECT = Path(
    os.environ.get("DAIGC_MCP_ROOT", "/home/sifr/3DAIGC-API/mcp")
)

_MODEL_PROCESSES = {
    "vlm": Process(
        "vlm", "../../services/vlm-server", "vlm_server",
        config="yaml/vlm_server.yaml",
    ),
    "vlm-omni": Process(
        "vlm-omni",
        "../../services/nemotron-omni-llm",
        "nemotron_omni_llm_server",
    ),
    "stt": Process(
        "stt", "../../services/stt-server", "stt_server",
        config="yaml/stt_server.yaml",
    ),
    "tts": Process(
        "tts", "../../services/piper-tts", "piper_tts_server",
        config="yaml/piper_tts_server.yaml",
    ),
}


def _build_processes() -> tuple[list[Process], tuple[str, ...]]:
    deployment = load_model_deployment(_BASE / _WORKER_CONFIG)
    unknown_services = deployment.services.keys() - _MODEL_PROCESSES.keys()
    if unknown_services:
        raise ValueError(
            f"model profile declares unknown services: {sorted(unknown_services)}"
        )
    procs = [
        Process(
            "3daigc-mcp",
            _DAIGC_MCP_PROJECT,
            "3daigc-mcp-http",
            launch_mode="reuse",
            port=8260,
        ),
        Process("hub", "../../services/xr-media-hub", "xr_media_hub",
                config="yaml/xr_media_hub.yaml"),
    ]
    for service, process in _MODEL_PROCESSES.items():
        launch_mode = deployment.launch_mode(service)
        if launch_mode is not None:
            procs.append(replace(process, launch_mode=launch_mode))
    procs.append(
        Process(
            "worker", "worker", "daigc_vlm_example_worker",
            config=_WORKER_CONFIG,
        )
    )
    return procs, deployment.required_credentials


def run() -> None:
    setup_logging("orchestrator", namespace="3daigc-vlm-example")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--allow-anonymous", action="store_true",
                   help="Start without HF_TOKEN (unauthenticated checkpoint "
                        "downloads may stall indefinitely).")
    ns, _ = p.parse_known_args()

    processes, credentials = _build_processes()
    require_credentials("HF_TOKEN", allow_missing=ns.allow_anonymous)
    for credential in credentials:
        ensure_credentials(credential)
    run_stack(processes, _BASE)


if __name__ == "__main__":
    run()
