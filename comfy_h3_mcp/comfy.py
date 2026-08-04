"""Thin async client for the local ComfyUI HTTP API.

Deliberately small: submit a graph, poll history, interrupt, upload assets.
No websocket, no session state - a prompt_id is the only handle we hand out.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import httpx

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
CLIENT_ID = os.environ.get("COMFYUI_CLIENT_ID") or f"comfy-h3-mcp-{uuid.uuid4().hex[:8]}"

# Video generation is slow, but every call here is either a submit or a poll -
# neither blocks on the run itself, so a short timeout is correct.
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


class ComfyError(RuntimeError):
    """A ComfyUI API call failed in a way worth showing the caller verbatim."""


async def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=COMFYUI_URL, timeout=_TIMEOUT)


async def object_info(node: str | None = None) -> dict[str, Any]:
    path = f"/object_info/{node}" if node else "/object_info"
    async with await _client() as c:
        r = await c.get(path)
        r.raise_for_status()
        return r.json()


async def submit(graph: dict[str, Any]) -> str:
    """Queue a graph. Returns the prompt_id."""
    async with await _client() as c:
        r = await c.post("/prompt", json={"prompt": graph, "client_id": CLIENT_ID})
        if r.status_code >= 400:
            # ComfyUI puts validation detail in the body; surface it rather than
            # a bare 400, since it names the offending node and input.
            try:
                raise ComfyError(_format_validation(r.json()))
            except ValueError:
                raise ComfyError(f"HTTP {r.status_code}: {r.text[:2000]}") from None
        return r.json()["prompt_id"]


def _format_validation(body: dict[str, Any]) -> str:
    err = body.get("error", {})
    lines = [f"{err.get('type', 'error')}: {err.get('message', body)}"]
    if err.get("details"):
        lines.append(f"  {err['details']}")
    for node_id, info in (body.get("node_errors") or {}).items():
        lines.append(f"  node {node_id} ({info.get('class_type', '?')}):")
        for e in info.get("errors", []):
            lines.append(f"    - {e.get('message')} {e.get('details', '')}".rstrip())
    return "\n".join(lines)


async def history(prompt_id: str) -> dict[str, Any] | None:
    async with await _client() as c:
        r = await c.get(f"/history/{prompt_id}")
        r.raise_for_status()
        return r.json().get(prompt_id)


async def queue_state() -> tuple[list, list]:
    """Returns (running, pending) raw queue entries."""
    async with await _client() as c:
        r = await c.get("/queue")
        r.raise_for_status()
        d = r.json()
        return d.get("queue_running", []), d.get("queue_pending", [])


async def progress_for(prompt_id: str) -> dict[str, Any] | None:
    """Best-effort live progress. /progress is not present on every build."""
    async with await _client() as c:
        try:
            r = await c.get("/progress")
            if r.status_code != 200:
                return None
            for entry in (r.json() or {}).values() if isinstance(r.json(), dict) else []:
                if isinstance(entry, dict) and entry.get("prompt_id") == prompt_id:
                    return entry
        except (httpx.HTTPError, ValueError):
            return None
    return None


async def interrupt() -> None:
    """Interrupt whatever is currently executing."""
    async with await _client() as c:
        r = await c.post("/interrupt")
        r.raise_for_status()


async def delete_queued(prompt_id: str) -> None:
    """Drop a still-pending prompt from the queue."""
    async with await _client() as c:
        r = await c.post("/queue", json={"delete": [prompt_id]})
        r.raise_for_status()


async def upload_asset(path: str, subfolder: str = "") -> str:
    """Upload a local file into ComfyUI's input dir. Returns the input-relative name."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ComfyError(f"No such file: {p}")
    data = {"overwrite": "true"}
    if subfolder:
        data["subfolder"] = subfolder
    async with await _client() as c:
        with p.open("rb") as fh:
            r = await c.post(
                "/upload/image",
                files={"image": (p.name, fh, "application/octet-stream")},
                data=data,
            )
        if r.status_code >= 400:
            raise ComfyError(f"Upload failed ({r.status_code}): {r.text[:500]}")
        body = r.json()
    name = body.get("name", p.name)
    sub = body.get("subfolder") or ""
    return f"{sub}/{name}" if sub else name


async def input_image_visible(name: str) -> bool:
    """Can LoadImage actually see this input file?

    ComfyUI builds the LoadImage combo by scanning its input dir when
    /object_info is requested, so a freshly uploaded file shows up right away.
    Verifying it anyway turns a future caching change - or a permissions
    problem - into a clear error instead of a mysterious workflow failure.
    """
    try:
        info = await object_info("LoadImage")
        options = info["LoadImage"]["input"]["required"]["image"][0]
        return isinstance(options, list) and name in options
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        # Can't tell - don't block the caller on a check that itself failed.
        return True


def _is_local() -> bool:
    return any(h in COMFYUI_URL for h in ("127.0.0.1", "localhost", "::1"))


def global_sage_attention() -> bool | None:
    """Is the ComfyUI server itself running with --use-sage-attention?

    When it is, patching sage per-graph is redundant. ComfyUI exposes no API for
    this, so read the process table - which only works when the server is local.
    Returns None when that cannot be determined.
    """
    if not _is_local():
        return None
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if any(part.endswith(b"main.py") for part in cmdline):
                return b"--use-sage-attention" in cmdline
    except OSError:
        return None
    return None


async def resolve_asset(ref: str, kind: str) -> str:
    """Accept either a local path (uploaded) or a name already in ComfyUI's input dir."""
    if os.path.sep in ref or ref.startswith("~"):
        candidate = Path(ref).expanduser()
        if candidate.is_file():
            return await upload_asset(str(candidate))
        raise ComfyError(f"{kind} looks like a path but does not exist: {ref}")
    return ref
