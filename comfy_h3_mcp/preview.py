"""Turn a finished H3 video into things an MCP client can actually perceive.

MCP has no video content block, so a clip comes back as a contact sheet
(ImageContent) plus its soundtrack (AudioContent). Both are built with ffmpeg,
which keeps this dependency-free on the Python side.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import httpx

from .comfy import COMFYUI_URL, ComfyError

_TIMEOUT = httpx.Timeout(120.0, connect=5.0)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def _run(*args: str) -> tuple[int, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out


async def fetch_output(filename: str, subfolder: str = "", type_: str = "output") -> bytes:
    """Pull a rendered file back out of ComfyUI via /view."""
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=_TIMEOUT) as c:
        r = await c.get(
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": type_},
        )
        if r.status_code >= 400:
            raise ComfyError(f"Could not fetch {filename} ({r.status_code})")
        return r.content


async def contact_sheet(
    video: bytes, columns: int = 4, rows: int = 3, tile_width: int = 320
) -> bytes:
    """Sample columns*rows frames evenly across the clip into one JPEG grid."""
    want = columns * rows
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.mp4"
        dst = Path(td) / "sheet.jpg"
        src.write_bytes(video)

        code, out = await _run("ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-count_packets", "-show_entries",
                               "stream=nb_read_packets", "-of", "csv=p=0", str(src))
        try:
            total = int(out.decode().strip().splitlines()[0])
        except (ValueError, IndexError):
            total = 0

        # Even sampling when we know the frame count; otherwise let ffmpeg take
        # the first `want` frames rather than failing outright.
        if total >= want:
            step = max(1, total // want)
            select = f"select='not(mod(n\\,{step}))'"
        else:
            select = "select='1'"

        vf = f"{select},scale={tile_width}:-1,tile={columns}x{rows}"
        code, out = await _run("ffmpeg", "-y", "-i", str(src), "-vf", vf,
                               "-frames:v", "1", "-q:v", "3", str(dst))
        if code != 0 or not dst.exists():
            raise ComfyError(f"ffmpeg contact sheet failed:\n{out.decode()[-1500:]}")
        return dst.read_bytes()


async def extract_audio(video: bytes) -> bytes | None:
    """Pull the soundtrack out as mp3. Returns None if the clip has no audio."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.mp4"
        dst = Path(td) / "out.mp3"
        src.write_bytes(video)

        code, out = await _run("ffprobe", "-v", "error", "-select_streams", "a:0",
                               "-show_entries", "stream=codec_type", "-of",
                               "csv=p=0", str(src))
        if code != 0 or "audio" not in out.decode():
            return None

        code, out = await _run("ffmpeg", "-y", "-i", str(src), "-vn",
                               "-codec:a", "libmp3lame", "-q:a", "4", str(dst))
        if code != 0 or not dst.exists():
            return None
        return dst.read_bytes()


async def probe(video: bytes) -> dict:
    """Duration / resolution / fps, for reporting alongside the sheet."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.mp4"
        src.write_bytes(video)
        code, out = await _run(
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_packets", "-count_packets",
            "-of", "csv=p=0", str(src))
        if code != 0:
            return {}
        parts = out.decode().strip().split(",")
        if len(parts) < 3:
            return {}
        try:
            num, den = parts[2].split("/")
            fps = round(int(num) / int(den), 2)
        except (ValueError, ZeroDivisionError):
            fps = None
        info = {"width": int(parts[0]), "height": int(parts[1]), "fps": fps}
        if len(parts) > 3 and parts[3].isdigit():
            info["frames"] = int(parts[3])
            if fps:
                info["duration_seconds"] = round(int(parts[3]) / fps, 2)
        return info
