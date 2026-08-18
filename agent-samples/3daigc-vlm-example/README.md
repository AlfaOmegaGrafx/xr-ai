<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# 3DAIGC VLM example

This sample answers voice and text questions against each participant's latest
camera frame, and runs 3DAIGC image-to-mesh when the user asks to generate a
3D model of what they see. Spoken replies stream to Piper TTS and the
`vlm.response` data topic. Completed mesh jobs are also published on
`3daigc.meshResult`.

The worker is a package under `worker/daigc_vlm_example_worker/`:

- `__main__.py` parses launcher arguments.
- `agent.py` owns participant-scoped vision turns, mesh generation, and cancellation.
- `daigc_tools.py` detects mesh intent and drives the 3DAIGC MCP pipeline.
- `config.py` resolves worker, model-profile, voice-gate, MCP, and prompt settings.
- `app.py` composes the native runtime.
- `prompts/system.txt` owns the VLM system prompt.

`VoiceAgent` privately owns STT/TTS/VLM readiness, the hub voice transport,
voice-gate processing, streaming TTS, signals, and cleanup. `DaigcVlmAgent`
subscribes to this sample's `UserQuery` topic, selects the participant's
current image with `CurrentFrameTool`, and either streams a VLM answer or
uploads the frame to 3DAIGC MCP (`upload_image` → `image_to_textured_mesh` →
`wait_for_job`).

## Prerequisites

- 3DAIGC-API on `:7842`
- 3daigc-mcp-http on `:8260`

On DGX Spark:

```bash
bash /home/sifr/3DAIGC-API/mcp/scripts/start_xr_voice_full.sh
```

## Run

```bash
cd agent-samples/3daigc-vlm-example
uv sync
uv run daigc_vlm_example
```

Open the web client on port **8088**, connect, then speak a question or
"make a 3D model of this".

The worker and orchestrator consume the deployment profile selected by
`models_config` in `yaml/3daigc_vlm_example_worker.yaml`:

- `models.hosted.json` (default) uses hosted NVIDIA NIM for VLM.
- `models.local.json` manages local STT, VLM, and TTS services.
- `models.omni.json` reuses the Nemotron-Omni VLM service on port 8108.

Set `DAIGC_MCP_ROOT` if 3DAIGC MCP is not at `/home/sifr/3DAIGC-API/mcp`.
