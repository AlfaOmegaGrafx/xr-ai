# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
3daigc-vlm-example worker — voice VLM plus 3DAIGC mesh generation.

Launched by ``uv run daigc_vlm_example``. Do not run directly.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import signal

from fastmcp import Client as McpClient
from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_stt, make_tts, make_vlm
from xr_ai_pipecat import VadConfig, make_voice_pipeline
from xr_ai_pipecat.services import mcp_probe, wait_for_services
from xr_ai_pipecat.transport import XRMediaHubTransport
from xr_ai_voicegate import load_voice_gate_config

from agent import DEFAULT_SYSTEM_PROMPT, DaigcVlmBrain
from config import WorkerConfig, load_config


async def main(
    cfg: WorkerConfig,
    config_path: pathlib.Path | None = None,
    ready_file: pathlib.Path | None = None,
) -> None:
    setup_logging("worker")

    models_cfg = load_models_config(cfg.models_yaml)
    stt = make_stt(models_cfg, "stt")
    vlm = make_vlm(models_cfg, "vlm")
    tts = make_tts(models_cfg, "tts")

    daigc_mcp = cfg.daigc_mcp_url.rstrip("/") + "/mcp"
    probes = {
        "stt": stt.health,
        "vlm": vlm.health,
        "tts": tts.health,
        "3daigc-mcp": mcp_probe(daigc_mcp),
    }
    await wait_for_services(probes)

    if ready_file:
        ready_file.touch()

    voice_gate_cfg = load_voice_gate_config(pathlib.Path(cfg.voice_gate_yaml))
    transport = XRMediaHubTransport()

    async with McpClient(daigc_mcp) as daigc_client:
        brain = DaigcVlmBrain(
            transport=transport,
            vlm=vlm,
            daigc_client=daigc_client,
            daigc_mesh_model=cfg.daigc_mesh_model,
            daigc_job_timeout_sec=cfg.daigc_job_timeout_sec,
            default_prompt=cfg.default_prompt,
            system_prompt=cfg.system_prompt or DEFAULT_SYSTEM_PROMPT,
            frame_max_age_s=cfg.frame_max_age_s,
            camera_on_timeout_s=cfg.camera_on_timeout_s,
            camera_grace_s=cfg.camera_grace_s,
        )

        _, task = make_voice_pipeline(
            transport=transport,
            stt=stt,
            tts=tts,
            brain=brain,
            vad_cfg=VadConfig(
                silence_duration=cfg.silence_duration,
                min_speech=cfg.min_speech,
                silero_threshold=cfg.silero_threshold,
            ),
            voice_gate_cfg=voice_gate_cfg,
            text_topic="vlm.response",
            idle_timeout_secs=cfg.idle_timeout_secs,
        )

        loop = asyncio.get_running_loop()
        cancel_requested = False

        def _request_cancel() -> None:
            nonlocal cancel_requested
            if cancel_requested:
                return
            cancel_requested = True
            asyncio.create_task(task.cancel())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _request_cancel)

        logger.info("3daigc-vlm-example starting pipecat pipeline")
        try:
            await PipelineRunner().run(task)
        finally:
            transport.shutdown()
            for svc in (stt, vlm, tts):
                try:
                    await svc.close()  # type: ignore[attr-defined]
                except Exception:
                    logger.opt(exception=True).warning("service close failed")
        logger.info("3daigc-vlm-example stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()
    cfg = load_config(ns.config)
    asyncio.run(main(cfg, config_path=ns.config, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
