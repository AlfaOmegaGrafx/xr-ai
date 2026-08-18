# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the packaged 3daigc-vlm-example worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tomllib
import yaml
from xr_ai_hub import DataMessage, FrameUnavailable
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest, ImageFrame
from xr_ai_tools.image import ImageReference, ImageRegistry
from xr_ai_voice import UserQuery, VoiceOutput

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = _REPO_ROOT / "agent-samples" / "3daigc-vlm-example"
_WORKER_DIR = _SAMPLE_DIR / "worker"
sys.path.insert(0, str(_WORKER_DIR))

from daigc_vlm_example_worker import __main__ as worker_main  # noqa: E402  # pyright: ignore[reportMissingImports]
from daigc_vlm_example_worker.agent import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    MESH_RESULT_TOPIC,
    DaigcVlmAgent,
)
from daigc_vlm_example_worker.config import load_config  # noqa: E402  # pyright: ignore[reportMissingImports]
from daigc_vlm_example_worker.daigc_tools import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    image_to_data_url,
    spoken_mesh_summary,
    wants_mesh_generation,
)


class _SelectedFrameTool:
    def __init__(self, image: ImageReference) -> None:
        self.image = image
        self.released: list[str] = []

    async def execute(self, request: CurrentFrameRequest) -> ImageFrame:
        return ImageFrame(
            image=self.image,
            width=2,
            height=2,
            timestamp_us=123,
            sequence=1,
            participant_id=request.participant_id,
        )

    def release(self, participant_id: str) -> None:
        self.released.append(participant_id)


class _StreamingImageQueryTool:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def stream(self, request):
        self.requests.append(request)
        for text in ("a ", "blue square"):
            yield SimpleNamespace(text=text)


async def _ignore_status(_status: str, _participant_id: str) -> None:
    pass


def test_worker_is_a_package_with_module_and_console_entry_points() -> None:
    project = tomllib.loads((_WORKER_DIR / "pyproject.toml").read_text())
    package = _WORKER_DIR / "daigc_vlm_example_worker"
    dependencies = set(project["project"]["dependencies"])

    assert project["project"]["scripts"]["daigc_vlm_example_worker"] == (
        "daigc_vlm_example_worker.__main__:run"
    )
    assert "xr-ai-hub-client" in dependencies
    assert "xr-ai-agent-runtime" in dependencies
    assert "xr-ai-voice" in dependencies
    assert "fastmcp>=0.4" in dependencies
    assert "xr-ai-pipecat" not in dependencies
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "daigc_vlm_example_worker"
    ]
    assert {
        "__init__.py",
        "__main__.py",
        "agent.py",
        "app.py",
        "config.py",
        "daigc_tools.py",
        "prompts/system.txt",
    } <= {str(path.relative_to(package)) for path in package.rglob("*") if path.is_file()}
    assert not (_WORKER_DIR / "agent.py").exists()
    assert not (_WORKER_DIR / "daigc_vlm_example_worker.py").exists()


def test_entry_point_loads_config_and_forwards_ready_file(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "worker.yaml"
    ready_file = tmp_path / "ready"
    config = object()
    seen = {}

    def fake_load_config(path):
        seen["config_path"] = path
        return config

    monkeypatch.setattr(worker_main, "load_config", fake_load_config)

    async def fake_run_app(loaded_config, *, ready_file):
        seen["config"] = loaded_config
        seen["ready_file"] = ready_file

    monkeypatch.setattr(worker_main, "run_app", fake_run_app)
    worker_main.run(
        [
            "--config",
            str(config_path),
            "--ready-file",
            str(ready_file),
            "--launcher-option",
            "ignored",
        ]
    )

    assert seen == {
        "config_path": config_path,
        "config": config,
        "ready_file": ready_file,
    }


def test_shipped_config_preserves_hosted_models_and_mesh_settings() -> None:
    config_path = _SAMPLE_DIR / "yaml" / "3daigc_vlm_example_worker.yaml"
    config = load_config(config_path)
    prompt = (_WORKER_DIR / "daigc_vlm_example_worker" / "prompts" / "system.txt").read_text()

    assert config.models_config == _SAMPLE_DIR / "yaml" / "models.hosted.json"
    assert config.voice_gate_yaml == _SAMPLE_DIR / "yaml" / "voice_gate.yaml"
    assert config.system_prompt == prompt
    assert "Speak directly to me in second person" in prompt
    assert 'Never refer to "the user" in the third person.' in prompt
    assert config.daigc_mcp_url == "http://localhost:8260"
    assert config.daigc_mesh_model is None
    assert config.daigc_job_timeout_sec == 600.0
    assert config.frame_max_age_s == 5.0
    assert config.frame_timeout_s == 30.0
    assert config.idle_timeout_secs is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("make a 3D model of this", True),
        ("generate a mesh from the camera", True),
        ("image to 3d", True),
        ("model this", True),
        ("what am I looking at?", False),
        ("describe this", False),
    ],
)
def test_wants_mesh_generation(query: str, expected: bool) -> None:
    assert wants_mesh_generation(query) is expected


def test_image_to_data_url_encodes_registered_jpeg_bytes() -> None:
    images = ImageRegistry()
    reference = images.put(b"jpeg-bytes", owner="alice")

    assert image_to_data_url(images, reference) == (
        "data:image/jpeg;base64,anBlZy1ieXRlcw=="
    )


def test_spoken_mesh_summary_reports_success_and_failure() -> None:
    assert "trellis2" in spoken_mesh_summary(
        {
            "job": {"status": "completed", "model": "trellis2"},
            "urls": {"download_url": "http://example/job"},
        }
    )
    assert "failed" in spoken_mesh_summary(
        {"job": {"status": "failed", "error": "oom"}}
    ).lower()


async def test_daigc_agent_runs_mesh_pipeline_instead_of_vlm() -> None:
    images = ImageRegistry()
    reference = images.put(b"jpeg-bytes", owner="alice")
    vision = _StreamingImageQueryTool()
    mesh_calls: list[str] = []
    published: list[VoiceOutput] = []
    returned: list[DataMessage] = []

    async def run_mesh(image_data_url: str) -> dict[str, Any]:
        mesh_calls.append(image_data_url)
        return {
            "job": {"status": "completed", "model": "trellis2"},
            "urls": {"download_url": "http://example/job"},
        }

    async def send_return_data(message: DataMessage) -> None:
        returned.append(message)

    class Context:
        agent_name = "3daigc-vlm"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, _topic, output) -> None:
            published.append(output)

    agent = DaigcVlmAgent(
        lambda: (_SelectedFrameTool(reference), vision),  # type: ignore[return-value]
        _ignore_status,
        images=images,
        send_return_data=send_return_data,
        run_mesh=run_mesh,
    )
    await agent._stream(  # noqa: SLF001
        UserQuery(text="make a 3D model of this", timestamp_us=123),
        Context(),  # type: ignore[arg-type]
    )

    assert vision.requests == []
    assert mesh_calls == ["data:image/jpeg;base64,anBlZy1ieXRlcw=="]
    assert [output.text for output in published if output.text][0].startswith(
        "Starting 3D mesh generation"
    )
    assert any("trellis2" in (output.text or "") for output in published)
    assert returned[0].topic == MESH_RESULT_TOPIC
    payload = json.loads(returned[0].data)
    assert payload["query"] == "make a 3D model of this"
    assert payload["result"]["job"]["model"] == "trellis2"


async def test_daigc_agent_uses_vlm_for_scene_questions() -> None:
    images = ImageRegistry()
    reference = images.put(b"jpeg-bytes", owner="alice")
    vision = _StreamingImageQueryTool()
    mesh_calls: list[str] = []
    published: list[VoiceOutput] = []

    async def run_mesh(_image_data_url: str) -> dict[str, Any]:
        mesh_calls.append("called")
        raise AssertionError("mesh path must not run for scene questions")

    async def send_return_data(_message: DataMessage) -> None:
        raise AssertionError("mesh data must not publish for scene questions")

    class Context:
        agent_name = "3daigc-vlm"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, _topic, output) -> None:
            published.append(output)

    agent = DaigcVlmAgent(
        lambda: (_SelectedFrameTool(reference), vision),  # type: ignore[return-value]
        _ignore_status,
        images=images,
        send_return_data=send_return_data,
        run_mesh=run_mesh,
    )
    await agent._stream(  # noqa: SLF001
        UserQuery(text="what am I looking at?", timestamp_us=123),
        Context(),  # type: ignore[arg-type]
    )

    assert mesh_calls == []
    assert vision.requests[0].query == "what am I looking at?"
    assert "".join(output.text or "" for output in published) == "a blue square"


async def test_daigc_agent_reports_missing_camera_frame() -> None:
    published: list[VoiceOutput] = []

    async def missing_frame(_request: CurrentFrameRequest) -> ImageFrame:
        raise FrameUnavailable("No camera frame available — please try again.")

    missing_frame_tool = Tool(
        "missing_camera_frame",
        "Reproduce a missing camera frame through the real Relay boundary.",
        CurrentFrameRequest,
        ImageFrame,
        missing_frame,
    )

    class Context:
        agent_name = "3daigc-vlm"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, _topic, output) -> None:
            published.append(output)

    async def unused_mesh(_url: str) -> dict:
        raise AssertionError("mesh should not run without a frame")

    async def unused_data(_message: DataMessage) -> None:
        raise AssertionError("mesh data should not publish without a frame")

    agent = DaigcVlmAgent(
        lambda: (missing_frame_tool, _StreamingImageQueryTool()),  # type: ignore[return-value]
        _ignore_status,
        images=ImageRegistry(),
        send_return_data=unused_data,
        run_mesh=unused_mesh,
    )
    await agent._stream(  # noqa: SLF001
        UserQuery(text="make a 3D model of this", timestamp_us=123),
        Context(),  # type: ignore[arg-type]
    )

    assert [output.text for output in published] == [
        "No camera frame available — please try again.",
        "",
    ]
