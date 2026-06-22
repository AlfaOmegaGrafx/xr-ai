# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3daigc-vlm-example worker configuration."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class WorkerConfig:
    model_backend: str
    models_yaml: str
    voice_gate_yaml: str
    default_prompt: str
    system_prompt: str | None
    frame_max_age_s: float
    camera_on_timeout_s: float
    camera_grace_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    idle_timeout_secs: float | None
    daigc_mcp_url: str
    daigc_mesh_model: str | None
    daigc_job_timeout_sec: float


def _resolve_relative(raw: str, config_path: pathlib.Path | None) -> str:
    p = pathlib.Path(raw)
    if config_path and not p.is_absolute():
        return str(config_path.parent / p)
    return raw


def load_config(path: pathlib.Path | None) -> WorkerConfig:
    data: dict = {}
    if path and path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    backend = str(data.get("model_backend", "local")).lower()
    models_yaml_raw = (
        "models.nim.yaml" if backend == "nim"
        else data.get("models_yaml", "models.yaml")
    )

    return WorkerConfig(
        model_backend=backend,
        models_yaml=_resolve_relative(models_yaml_raw, path),
        voice_gate_yaml=_resolve_relative(
            data.get("voice_gate_yaml", "voice_gate.yaml"), path,
        ),
        default_prompt=data.get("default_prompt", "Describe what you see."),
        system_prompt=data.get("system_prompt"),
        frame_max_age_s=float(data.get("frame_max_age_s", 5.0)),
        camera_on_timeout_s=float(data.get("camera_on_timeout_s", 30.0)),
        camera_grace_s=float(data.get("camera_grace_s", 5.0)),
        silence_duration=float(data.get("silence_duration", 0.4)),
        min_speech=float(data.get("min_speech", 0.1)),
        silero_threshold=float(data.get("silero_threshold", 0.5)),
        idle_timeout_secs=(
            float(data["idle_timeout_secs"])
            if data.get("idle_timeout_secs") else None
        ),
        daigc_mcp_url=data.get("daigc_mcp_url", "http://localhost:8260"),
        daigc_mesh_model=data.get("daigc_mesh_model"),
        daigc_job_timeout_sec=float(data.get("daigc_job_timeout_sec", 600.0)),
    )
