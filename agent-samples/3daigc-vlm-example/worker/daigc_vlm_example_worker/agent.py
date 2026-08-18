# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped streaming orchestration for the 3DAIGC VLM sample."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import nemo_relay
from xr_ai_hub import DataMessage, FrameUnavailable
from xr_ai_runtime import (
    Agent,
    RuntimeClosedError,
    RuntimeContext,
    Topic,
    subscribe,
)
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool, ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, StreamingImageQueryTool
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
)

from .daigc_tools import image_to_data_url, spoken_mesh_summary, wants_mesh_generation

USER_QUERY_TOPIC = Topic("3daigc-vlm.user-query", UserQuery)
PARTICIPANT_LEFT_TOPIC = Topic(
    "3daigc-vlm.participant-left",
    VoiceParticipantLeft,
)
INTERRUPTED_TOPIC = Topic("3daigc-vlm.interrupted", VoiceInterrupted)
MESH_RESULT_TOPIC = "3daigc.meshResult"

_MeshRunner = Callable[[str], Awaitable[dict[str, Any]]]
_SendReturnData = Callable[[DataMessage], Awaitable[None]]


class DaigcVlmAgent(Agent):
    """Own streamed user turns, mesh generation, and cancellation."""

    def __init__(
        self,
        vision_factory: Callable[
            [],
            tuple[CurrentFrameTool, StreamingImageQueryTool],
        ],
        set_status: Callable[[str, str], Awaitable[None]],
        *,
        images: ImageRegistry,
        send_return_data: _SendReturnData,
        run_mesh: _MeshRunner,
    ) -> None:
        super().__init__()
        self._vision_factory = vision_factory
        self._set_status = set_status
        self._images = images
        self._send_return_data = send_return_data
        self._run_mesh = run_mesh
        self._frames: CurrentFrameTool | None = None
        self._vision: StreamingImageQueryTool | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @subscribe(USER_QUERY_TOPIC)
    async def answer_user(self, request: UserQuery, ctx: RuntimeContext) -> None:
        """Supersede and start one participant's streamed response."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("3DAIGC VLM queries require a participant")
        await self._cancel(participant_id)
        task = asyncio.create_task(
            self._stream(request, ctx),
            name=f"3daigc-vlm-query:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: self._discard(pid, completed))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        """Release this agent's work and frame state for a departed participant."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("participant-left events require a participant")
        await self._cancel(participant_id)
        if self._frames is not None:
            self._frames.release(participant_id)

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        ctx: RuntimeContext,
    ) -> None:
        """Cancel participant-scoped or global vision work."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            await self._cancel_all()
            return
        await self._cancel(participant_id)

    async def _stream(self, request: UserQuery, ctx: RuntimeContext) -> None:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            with nemo_relay.scope.scope(
                "3daigc-vlm.turn",
                nemo_relay.ScopeType.Agent,
                input=request.model_dump(mode="json"),
                metadata={
                    "agent": ctx.agent_name,
                    "message_id": ctx.metadata.message_id,
                    "correlation_id": ctx.metadata.correlation_id,
                    "participant_id": ctx.metadata.participant_id,
                },
            ):
                await self._stream_response(request, ctx)

    async def _stream_response(
        self,
        request: UserQuery,
        ctx: RuntimeContext,
    ) -> None:
        response_id = ctx.metadata.message_id
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("3DAIGC VLM queries require a participant")
        first = True
        opened = False
        cancelled = False
        processing = False
        try:
            if self._vision is None or self._frames is None:
                self._frames, self._vision = self._vision_factory()
            try:
                frame = await self._frames.execute(
                    CurrentFrameRequest(participant_id=participant_id)
                )
            except (FrameUnavailable, RuntimeError) as exc:
                unavailable = _frame_unavailable_message(exc)
                if unavailable is None:
                    raise
                await ctx.publish(
                    VOICE_OUTPUT_TOPIC,
                    VoiceOutput(
                        text=unavailable,
                        response_id=response_id,
                        final=False,
                        interrupt=True,
                        timestamp_us=request.timestamp_us,
                    ),
                )
                opened = True
                return
            await self._set_status("processing", participant_id)
            processing = True
            if wants_mesh_generation(request.text):
                opened = await self._stream_mesh(
                    request,
                    ctx,
                    frame,
                    first=first,
                )
                return
            stream = self._vision.stream(ImageQueryRequest(image=frame.image, query=request.text))
            try:
                async for chunk in stream:
                    await ctx.publish(
                        VOICE_OUTPUT_TOPIC,
                        VoiceOutput(
                            text=chunk.text,
                            response_id=response_id,
                            final=False,
                            interrupt=first,
                            timestamp_us=request.timestamp_us,
                        ),
                    )
                    first = False
                    opened = True
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if processing:
                await self._set_status("idle", participant_id)
            if opened and not cancelled:
                with suppress(RuntimeClosedError):
                    await ctx.publish(
                        VOICE_OUTPUT_TOPIC,
                        VoiceOutput(
                            response_id=response_id,
                            timestamp_us=request.timestamp_us,
                        ),
                    )

    async def _stream_mesh(
        self,
        request: UserQuery,
        ctx: RuntimeContext,
        frame: ImageFrame,
        *,
        first: bool,
    ) -> bool:
        response_id = ctx.metadata.message_id
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("3DAIGC VLM queries require a participant")
        opened = False

        async def speak(text: str, *, interrupt: bool) -> None:
            nonlocal opened
            await ctx.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(
                    text=text,
                    response_id=response_id,
                    final=False,
                    interrupt=interrupt,
                    timestamp_us=request.timestamp_us,
                ),
            )
            opened = True

        image_url = image_to_data_url(self._images, frame.image)
        await speak(
            "Starting 3D mesh generation from your camera view. "
            "This may take several minutes.",
            interrupt=first,
        )
        try:
            result = await self._run_mesh(image_url)
            await speak(spoken_mesh_summary(result), interrupt=False)
            await self._send_return_data(
                DataMessage(
                    participant_id=participant_id,
                    topic=MESH_RESULT_TOPIC,
                    pts_us=time.time_ns() // 1000,
                    data=json.dumps(
                        {"query": request.text, "result": result},
                        default=str,
                    ).encode(),
                )
            )
        except Exception as exc:
            await speak(f"3D generation failed: {exc}", interrupt=False)
        return opened

    async def _cancel(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _cancel_all(self) -> None:
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Cancel all image-query turns owned by this agent."""

        await self._cancel_all()

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)


def _frame_unavailable_message(error: BaseException) -> str | None:
    """Recover a camera error from native or Relay-scrubbed exceptions."""

    relay_prefix = "internal error: FrameUnavailable:"
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, FrameUnavailable):
            return str(current)
        if isinstance(current, RuntimeError):
            message = str(current)
            if message.startswith(relay_prefix):
                return message.removeprefix(relay_prefix).strip()
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


__all__ = [
    "INTERRUPTED_TOPIC",
    "MESH_RESULT_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "DaigcVlmAgent",
    "USER_QUERY_TOPIC",
]
