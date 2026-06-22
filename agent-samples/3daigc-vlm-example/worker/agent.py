# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DaigcVlmBrain — simple-vlm voice UX plus 3DAIGC mesh generation.

When the user asks to make a 3D model from the camera view, the brain
captures a frame and drives the 3daigc-mcp pipeline (upload_image →
image_to_textured_mesh → wait_for_job). All other queries use the VLM path
from :class:`SimpleVlmBrain`.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

from fastmcp import Client as McpClient
from loguru import logger

from daigc_tools import (
    run_image_to_mesh,
    spoken_mesh_summary,
    wants_mesh_generation,
)
from pixels import encode_image, frame_to_pil
from vlm_brain_base import DEFAULT_SYSTEM_PROMPT, SimpleVlmBrain, _now_us
from xr_ai_agent import DataMessage
from xr_ai_logging import print_task_done_banner


class DaigcVlmBrain(SimpleVlmBrain):
    """VLM brain with optional 3DAIGC mesh generation via HTTP MCP."""

    def __init__(
        self,
        *,
        daigc_client: McpClient | None = None,
        daigc_mesh_model: str | None = None,
        daigc_job_timeout_sec: float = 600.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._daigc = daigc_client
        self._daigc_mesh_model = daigc_mesh_model
        self._daigc_job_timeout_sec = daigc_job_timeout_sec

    async def _stream_query(self, pid: str, query: str) -> AsyncIterator[str]:
        if self._daigc is not None and wants_mesh_generation(query):
            async for token in self._stream_3daigc_mesh(pid, query):
                yield token
            return

        async for token in super()._stream_query(pid, query):
            yield token

    async def _stream_3daigc_mesh(self, pid: str, query: str) -> AsyncIterator[str]:
        old_timer = self._camera_off_timers.pop(pid, None)
        if old_timer and not old_timer.done():
            old_timer.cancel()

        self._camera_held.add(pid)
        t0 = time.monotonic()
        status = "done"
        try:
            sig = self._latest_signal(pid)
            if not (sig and self._is_fresh(sig)):
                await self._ensure_camera_on(pid)
                sig = await self._wait_for_camera_frame(
                    pid, self._camera_on_timeout,
                )
                if sig is None:
                    self._camera_on[pid] = False
                    yield "Camera unavailable, please try again."
                    return

            frame = await self._transport.endpoint.request_frame(sig)
            if frame is None:
                yield "Frame data unavailable — please retry."
                return

            loop = asyncio.get_running_loop()
            image_url = await loop.run_in_executor(
                None, lambda: encode_image(frame_to_pil(frame)),
            )
            logger.info(
                "3daigc mesh  pid={!r}  {}x{}  query={!r}",
                pid, frame.width, frame.height, query[:60],
            )

            yield (
                "Starting 3D mesh generation from your camera view. "
                "This may take several minutes."
            )

            await self._transport.endpoint.set_status("processing", pid)
            try:
                result = await run_image_to_mesh(
                    self._daigc,
                    image_data_url=image_url,
                    model_preference=self._daigc_mesh_model,
                    timeout_sec=self._daigc_job_timeout_sec,
                )
                summary = spoken_mesh_summary(result)
                yield summary

                await self._transport.send_return_data(DataMessage(
                    participant_id=pid,
                    topic="3daigc.meshResult",
                    pts_us=_now_us(),
                    data=json.dumps({
                        "query": query,
                        "result": result,
                    }).encode(),
                ))
            except Exception as exc:
                logger.exception("3daigc mesh pipeline failed")
                yield f"3D generation failed: {exc}"
            finally:
                await self._transport.endpoint.set_status("idle", pid)
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            self._camera_held.discard(pid)
            self._schedule_camera_off(pid)
            print_task_done_banner(
                "3daigc-vlm-example",
                status=status,
                detail=f"pid={pid!r}  query={query[:60]!r}",
                duration_s=time.monotonic() - t0,
            )


__all__ = ["DEFAULT_SYSTEM_PROMPT", "DaigcVlmBrain"]
