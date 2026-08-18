# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3DAIGC MCP helpers — intent detection and image-to-mesh pipeline."""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from fastmcp import Client as McpClient
from xr_ai_tools.image import ImageReference, ImageRegistry

_MESH_INTENT = re.compile(
    r"(?:"
    r"\b(?:make|create|generate|turn|build|model)\b.{0,40}\b(?:3d|three.?d|mesh|model)\b"
    r"|\bimage\s+to\s+3d\b"
    r"|\bmodel\s+this\b"
    r"|\b3d\s+(?:model|mesh)\b.{0,20}\b(?:this|what\s+i\s+see|camera|view)\b"
    r")",
    re.IGNORECASE,
)


def mcp_endpoint(url: str) -> str:
    """Normalize a 3DAIGC MCP base URL to the FastMCP HTTP path."""

    base = url.rstrip("/")
    return base if base.endswith("/mcp") else f"{base}/mcp"


def wants_mesh_generation(query: str) -> bool:
    """Return True when the user is asking to generate a 3D mesh from the camera."""

    return bool(_MESH_INTENT.search(query.strip()))


def image_to_data_url(images: ImageRegistry, reference: ImageReference) -> str:
    """Encode a registered camera frame as a JPEG data URL for MCP upload."""

    image = images.resolve(reference)
    if isinstance(image, bytes):
        return f"data:image/jpeg;base64,{base64.b64encode(image).decode()}"
    if isinstance(image, str) and image.startswith("data:"):
        return image
    raise TypeError(f"unsupported image input for MCP upload: {type(image)!r}")


def parse_tool_result(result: Any) -> dict[str, Any]:
    """Unwrap a FastMCP tool result into a dict."""

    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data

    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured

    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not parse MCP tool result: {result!r}")


async def call_tool_json(
    client: McpClient,
    name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    result = await client.call_tool(name, args)
    return parse_tool_result(result)


def _base64_from_data_url(image_data_url: str) -> str:
    if image_data_url.startswith("data:"):
        return image_data_url.split(",", 1)[1]
    return image_data_url


async def mcp_health(url: str) -> bool:
    """Return True when the 3DAIGC MCP server exposes mesh-generation tools."""

    try:
        async with McpClient(mcp_endpoint(url)) as client:
            tools = await client.list_tools()
        listed = getattr(tools, "tools", tools)
        names = {getattr(tool, "name", "") for tool in listed}
        return bool(names & {"health_check", "upload_image", "image_to_textured_mesh"})
    except Exception:
        return False


async def run_image_to_mesh(
    client: McpClient,
    *,
    image_data_url: str,
    model_preference: str | None = None,
    timeout_sec: float = 600.0,
) -> dict[str, Any]:
    """Upload camera frame → image_to_textured_mesh → wait_for_job."""

    b64 = _base64_from_data_url(image_data_url)

    upload = await call_tool_json(
        client,
        "upload_image",
        {"source": "base64", "data": b64},
    )
    file_id = upload.get("file_id")
    if not file_id:
        raise RuntimeError(f"upload_image missing file_id: {upload!r}")

    mesh_args: dict[str, Any] = {"image_file_id": file_id}
    if model_preference:
        mesh_args["model_preference"] = model_preference

    submit = await call_tool_json(client, "image_to_textured_mesh", mesh_args)
    job_id = submit.get("job_id")
    if not job_id:
        raise RuntimeError(f"image_to_textured_mesh missing job_id: {submit!r}")

    return await call_tool_json(
        client,
        "wait_for_job",
        {"job_id": job_id, "timeout_sec": timeout_sec},
    )


def spoken_mesh_summary(result: dict[str, Any]) -> str:
    """Short TTS-friendly summary after mesh generation completes."""

    job = result.get("job") or {}
    status = str(job.get("status") or result.get("status") or "unknown")
    if status != "completed":
        err = job.get("error") or result.get("error") or "generation failed"
        return f"3D generation did not finish. Status {status}. {err}"

    summary = result.get("summary") or {}
    model = summary.get("model") or job.get("model") or "the model"
    urls = result.get("urls") or {}
    download = urls.get("download_url") or urls.get("primary_download_url")
    if download:
        return (
            f"Your 3D model is ready from {model}. "
            f"Download it from the API job result URL in the studio."
        )
    return f"Your 3D model finished successfully using {model}."
