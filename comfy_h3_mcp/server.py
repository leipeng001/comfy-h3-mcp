"""MCP server exposing MiniMax-H3 video+audio generation on a local ComfyUI.

Six tools, not a general ComfyUI control plane. Generation is slow, so the
generate tools submit and return a prompt_id immediately; poll with job_status.
"""

from __future__ import annotations

import random
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Audio, Image

from . import comfy, graphs, preview

mcp = MCPServer(
    "comfy-h3",
    version="0.1.0",
    instructions=(
        "MiniMax-H3 video+audio generation on a local ComfyUI.\n\n"
        "TIMING: generation takes MINUTES, not seconds. At the defaults "
        "(864x480, 20 steps) expect roughly 3-4 minutes on an RTX 4090; at "
        "1344x768 with 30 steps, 13-18 minutes. The generate tools return a "
        "prompt_id immediately with an estimated_seconds field - trust that "
        "estimate and poll job_status at a matching interval rather than "
        "assuming a slow run is stuck.\n\n"
        "RESOLUTION: leave width/height unset so they default to 864x480 "
        "(0.4 MP, 16:9) unless the user explicitly asks for something larger. "
        "Bigger is NOT better here - 1344x768 costs 2.5x the time AND drifts "
        "further from the prompt, because 0.4 MP is closer to the model's "
        "trained regime. Upscale in post if you need delivery resolution.\n\n"
        "ONE SHOT PER GENERATION: H3 has no shot-boundary mechanism, so a "
        "single clip cannot contain cuts. A prompt describing several shots, or "
        "carrying a timecoded shot list, yields one continuous take that smears "
        "through the beats - the timecodes are just noise to the text encoder. "
        "For anything with cuts, call the tool once PER SHOT, each with a single "
        "camera setup, then edit the clips together. For character or location "
        "consistency across shots, take a good frame from the first clip and "
        "pass it to h3_reference_to_video as a reference image, addressed in "
        "the prompt as <Picture 1>. Keep each shot to its own beat; 124 frames "
        "(~5s) is plenty for one setup.\n\n"
        "SAGE ATTENTION: call list_assets FIRST to check sage_attention.global. "
        "If ComfyUI was launched with --use-sage-attention, sage is already "
        "active everywhere and the per-call sage_attention parameter is "
        "redundant - do not set it, and do not tell the user to enable sage. "
        "Only pass sage_attention when global is false and the KJNodes pack is "
        "available."
    ),
)

# Defaults mirror the official template, video_minimax_h3_t2v.json.
DEFAULT_STEPS = 20
DEFAULT_SAMPLER = "res_multistep"
DEFAULT_SCHEDULER = "simple"
DEFAULT_MEGAPIXELS = 0.4


def _plan(width: int, height: int, megapixels: float, length: int) -> dict[str, Any]:
    """Resolve the caller's request to what the model will actually run.

    width/height left at 0 means "derive from megapixels at 16:9", which is how
    the official template drives it via ResolutionSelector.
    """
    if width and height:
        w, h = graphs.snap_axis(width), graphs.snap_axis(height)
        if w * h > graphs.MAX_PIXELS:
            w, h = graphs.adapt_canvas(width, height)
    else:
        w, h = graphs.resolution_for(megapixels)

    frames = graphs.align_frame_count(length)
    return {
        "width": w,
        "height": h,
        "megapixels": round(w * h / 1_048_576, 2),
        "length": frames,
        "duration_seconds": round(frames / graphs.FPS, 2),
    }


def _submitted(
    prompt_id: str, plan: dict[str, Any], seed: int, steps: int,
    sage_attention: str, **extra: Any
) -> dict:
    sage_global = comfy.global_sage_attention()
    sage_on = bool(sage_global) or sage_attention != "disabled"
    est = graphs.estimate_runtime(
        plan["width"], plan["height"], plan["length"], steps, sage_on
    )
    poll = max(15, est["estimated_seconds"] // 8)
    return {
        "prompt_id": prompt_id,
        "status": "submitted",
        "seed": seed,
        "steps": steps,
        **plan,
        **est,
        "sage_active": sage_on,
        **extra,
        "next": (
            f"Poll job_status(prompt_id) about every {poll}s. This is expected "
            f"to take {est['estimated']} - it is not stuck."
        ),
    }


@mcp.tool()
async def h3_image_to_video(
    prompt: str,
    first_frame: str | None = None,
    last_frame: str | None = None,
    width: int = 0,
    height: int = 0,
    megapixels: float = DEFAULT_MEGAPIXELS,
    length: int = 124,
    seed: int | None = None,
    steps: int = DEFAULT_STEPS,
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    shift_video: float | None = None,
    shift_audio: float | None = None,
    filename_prefix: str = "video/h3_i2v",
    sage_attention: str = "disabled",
) -> dict:
    """Generate video with synchronized audio from a text prompt, optionally
    anchored by a first and/or last keyframe (MiniMax-H3 fl2va model).

    With no keyframes this is pure text-to-video. Images may be a local file
    path (uploaded automatically) or a name already in ComfyUI's input folder.
    length is a frame count at 24 fps and snaps up to the model's 17k+5 grid;
    124 frames is about 5 seconds, and the trained range is roughly 124-362.
    Leave width/height unset: they default to 864x480, which is both faster
    and closer to the prompt than larger canvases. Only raise it if the user
    asks. This produces ONE continuous shot - it cannot contain cuts, so for a
    multi-shot sequence call this once per shot and edit the clips together
    rather than describing several shots in one prompt.
    Takes MINUTES: ~3-4 min at the defaults on an RTX 4090, longer at higher
    resolution or step count. The response carries estimated_seconds; poll
    job_status at that cadence instead of assuming a long run has hung.
    Do NOT set sage_attention without first checking list_assets - if ComfyUI
    runs with --use-sage-attention, sage is already on and this is redundant.
    Returns immediately with a prompt_id - poll job_status to get the output.
    """
    if sage_attention not in graphs.SAGE_MODES:
        raise ValueError(f"sage_attention must be one of {list(graphs.SAGE_MODES)}")
    seed = random.randint(0, 2**32 - 1) if seed is None else seed
    plan = _plan(width, height, megapixels, length)

    first = await comfy.resolve_asset(first_frame, "first_frame") if first_frame else None
    last = await comfy.resolve_asset(last_frame, "last_frame") if last_frame else None

    graph = graphs.build_image_to_video(
        prompt=prompt,
        width=plan["width"],
        height=plan["height"],
        length=plan["length"],
        first_frame=first,
        last_frame=last,
        seed=seed,
        steps=steps,
        sampler_name=sampler_name,
        scheduler=scheduler,
        shift_video=shift_video,
        shift_audio=shift_audio,
        filename_prefix=filename_prefix,
        sage_attention=sage_attention,
    )
    prompt_id = await comfy.submit(graph)
    mode = "text-to-video" if not (first or last) else "keyframe-guided"
    return _submitted(prompt_id, plan, seed, steps, sage_attention, mode=mode)


@mcp.tool()
async def h3_reference_to_video(
    prompt: str,
    ref_images: list[str] | None = None,
    ref_videos: list[str] | None = None,
    ref_audios: list[str] | None = None,
    ref_image_size: str = "match",
    width: int = 0,
    height: int = 0,
    megapixels: float = DEFAULT_MEGAPIXELS,
    length: int = 124,
    seed: int | None = None,
    steps: int = DEFAULT_STEPS,
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    shift_video: float | None = None,
    shift_audio: float | None = None,
    filename_prefix: str = "video/h3_ref2v",
    sage_attention: str = "disabled",
) -> dict:
    """Generate video with synchronized audio from a prompt plus reference
    images, videos, and/or audio (MiniMax-H3 ref2va model).

    References are addressed positionally in the prompt as <Picture i>,
    <Video k> and <Audio j>, all 1-based per type - e.g. "<Picture 1> walks
    through the door speaking in the voice of <Audio 1>". Max 9 images, 3
    videos, 3 audio. A reference video's own soundtrack is passed through
    automatically. ref_image_size "match" scales references to the output's
    pixel area; "max" uses a 2048px short edge for better identity fidelity but
    is several times slower, since reference tokens ride through every step.
    This is the tool for CONSISTENCY ACROSS SHOTS: generate shot 1, pull a
    frame of the character or location from it, and pass it here as a reference
    for shots 2..n so they match. Leave width/height unset (864x480 default).
    Takes MINUTES; see estimated_seconds in the response and poll at that
    cadence. Check list_assets before setting sage_attention - it is redundant
    when ComfyUI already runs with --use-sage-attention.
    Returns immediately with a prompt_id - poll job_status to get the output.
    """
    ref_images = ref_images or []
    ref_videos = ref_videos or []
    ref_audios = ref_audios or []

    if len(ref_images) > 9:
        raise ValueError("At most 9 reference images.")
    if len(ref_videos) > 3:
        raise ValueError("At most 3 reference videos.")
    if len(ref_audios) > 3:
        raise ValueError("At most 3 reference audio clips.")
    if ref_image_size not in ("match", "max"):
        raise ValueError("ref_image_size must be 'match' or 'max'.")
    if not (ref_images or ref_videos or ref_audios):
        raise ValueError(
            "No references given. Use h3_image_to_video for text-to-video."
        )

    if sage_attention not in graphs.SAGE_MODES:
        raise ValueError(f"sage_attention must be one of {list(graphs.SAGE_MODES)}")
    seed = random.randint(0, 2**32 - 1) if seed is None else seed
    plan = _plan(width, height, megapixels, length)

    images = [await comfy.resolve_asset(r, "ref_image") for r in ref_images]
    videos = [await comfy.resolve_asset(r, "ref_video") for r in ref_videos]
    audios = [await comfy.resolve_asset(r, "ref_audio") for r in ref_audios]

    graph = graphs.build_reference_to_video(
        prompt=prompt,
        width=plan["width"],
        height=plan["height"],
        length=plan["length"],
        ref_images=images,
        ref_videos=videos,
        ref_audios=audios,
        ref_image_size=ref_image_size,
        seed=seed,
        steps=steps,
        sampler_name=sampler_name,
        scheduler=scheduler,
        shift_video=shift_video,
        shift_audio=shift_audio,
        filename_prefix=filename_prefix,
        sage_attention=sage_attention,
    )
    prompt_id = await comfy.submit(graph)
    return _submitted(
        prompt_id,
        plan,
        seed,
        steps,
        sage_attention,
        references={
            "images": len(images),
            "videos": len(videos),
            "audios": len(audios),
        },
    )


@mcp.tool()
async def job_status(prompt_id: str) -> dict:
    """Check a submitted H3 job. Returns queued / running / completed / failed,
    plus the output file paths once it has finished."""
    hist = await comfy.history(prompt_id)

    if hist is None:
        running, pending = await comfy.queue_state()
        for entry in running:
            if len(entry) > 1 and entry[1] == prompt_id:
                out: dict[str, Any] = {"prompt_id": prompt_id, "status": "running"}
                prog = await comfy.progress_for(prompt_id)
                if prog:
                    out["progress"] = {
                        "value": prog.get("value"),
                        "max": prog.get("max"),
                        "node": prog.get("node"),
                    }
                return out
        for pos, entry in enumerate(pending):
            if len(entry) > 1 and entry[1] == prompt_id:
                return {
                    "prompt_id": prompt_id,
                    "status": "queued",
                    "position": pos + 1,
                }
        return {
            "prompt_id": prompt_id,
            "status": "unknown",
            "detail": "Not in queue or history. Wrong id, or ComfyUI restarted.",
        }

    status = hist.get("status", {})
    if status.get("status_str") == "error" or not status.get("completed", False):
        if status.get("status_str") == "error":
            errors = [
                m for m in status.get("messages", []) if m and m[0] == "execution_error"
            ]
            return {
                "prompt_id": prompt_id,
                "status": "failed",
                "errors": errors or status.get("messages"),
            }
        return {"prompt_id": prompt_id, "status": "running"}

    files: list[dict[str, Any]] = []
    for node_output in (hist.get("outputs") or {}).values():
        for key in ("images", "videos", "audio", "gifs"):
            for item in node_output.get(key, []) or []:
                files.append(
                    {
                        "filename": item.get("filename"),
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                        "url": f"{comfy.COMFYUI_URL}/view?filename={item.get('filename')}"
                        f"&subfolder={item.get('subfolder', '')}"
                        f"&type={item.get('type', 'output')}",
                    }
                )
    return {"prompt_id": prompt_id, "status": "completed", "outputs": files}


@mcp.tool()
async def job_cancel(prompt_id: str) -> dict:
    """Cancel an H3 job: removes it if still queued, interrupts it if running."""
    running, pending = await comfy.queue_state()

    if any(len(e) > 1 and e[1] == prompt_id for e in pending):
        await comfy.delete_queued(prompt_id)
        return {"prompt_id": prompt_id, "status": "cancelled", "was": "queued"}

    if any(len(e) > 1 and e[1] == prompt_id for e in running):
        await comfy.interrupt()
        return {"prompt_id": prompt_id, "status": "interrupted", "was": "running"}

    return {
        "prompt_id": prompt_id,
        "status": "not_cancelled",
        "detail": "Not in the queue - already finished, or never submitted.",
    }


@mcp.tool()
async def job_preview(
    prompt_id: str,
    columns: int = 4,
    rows: int = 3,
    tile_width: int = 320,
    include_audio: bool = True,
) -> list:
    """Look at and listen to a finished H3 job.

    MCP has no video content type, so this returns a contact sheet of frames
    sampled evenly across the clip (viewable as an image) plus the soundtrack
    as audio. Use it to actually judge a result before iterating on the prompt.
    """
    if not preview.ffmpeg_available():
        raise RuntimeError("ffmpeg is not on PATH; job_preview needs it.")

    status = await job_status(prompt_id)
    if status["status"] != "completed":
        raise RuntimeError(
            f"Job is {status['status']}, not completed. "
            f"{status.get('detail') or status.get('errors') or ''}".strip()
        )
    if not status["outputs"]:
        raise RuntimeError("Job completed but produced no output files.")

    out = status["outputs"][0]
    video = await preview.fetch_output(
        out["filename"], out.get("subfolder", ""), out.get("type", "output")
    )

    sheet = await preview.contact_sheet(video, columns, rows, tile_width)
    info = await preview.probe(video)

    blocks: list = [
        f"{out['filename']} - {info.get('width')}x{info.get('height')}, "
        f"{info.get('frames', '?')} frames @ {info.get('fps', '?')} fps "
        f"({info.get('duration_seconds', '?')}s). Contact sheet is "
        f"{columns}x{rows} frames sampled evenly across the clip, in reading order.\n"
        f"File: {out['url']}",
        Image(data=sheet, format="jpeg"),
    ]

    if include_audio:
        audio = await preview.extract_audio(video)
        if audio:
            blocks.append(Audio(data=audio, format="mp3"))
        else:
            blocks.append("No audio stream found in the output.")

    return blocks


@mcp.tool()
async def list_assets() -> dict:
    """List what this ComfyUI can actually load: H3 models, and the images,
    videos and audio already sitting in the input folder (usable by name)."""
    info = await comfy.object_info()

    def opts(node: str, field: str) -> list[str]:
        try:
            v = info[node]["input"]["required"][field][0]
            return v if isinstance(v, list) else []
        except (KeyError, IndexError, TypeError):
            return []

    unets = opts("UNETLoader", "unet_name")
    missing = [
        m for m in (graphs.MODEL_FL2VA, graphs.MODEL_REF2VA) if m not in unets
    ]

    sage_global = comfy.global_sage_attention()
    sage_node = graphs.SAGE_NODE in info
    if sage_global:
        sage_advice = (
            "Sage is ALREADY ACTIVE globally (--use-sage-attention). Do not pass "
            "the sage_attention parameter and do not suggest enabling sage."
        )
    elif sage_global is False and sage_node:
        sage_advice = (
            "Sage is not global, but the KJNodes patch node is available: pass "
            "sage_attention='auto' for roughly a 1.35x speedup."
        )
    elif sage_global is False:
        sage_advice = (
            "Sage is not global and the KJNodes patch node is not loaded. "
            "Install ComfyUI-KJNodes plus the sageattention library, or relaunch "
            "ComfyUI with --use-sage-attention."
        )
    else:
        sage_advice = "Could not determine sage state (ComfyUI may be remote)."

    return {
        "comfyui_url": comfy.COMFYUI_URL,
        "sage_attention": {
            "global": sage_global,
            "patch_node_available": sage_node,
            "advice": sage_advice,
        },
        "timing_reference": {
            "864x480, 20 steps, sage": "~3m 43s",
            "864x480, 24 steps, sage": "~4m 11s",
            "1344x768, 30 steps, sage": "~13m 13s",
            "1344x768, 30 steps, no sage": "~18m 02s",
            "note": "RTX 4090 at 124 frames; scales with pixels x steps x length.",
        },
        "h3_models": {
            "image_to_video": graphs.MODEL_FL2VA,
            "reference_to_video": graphs.MODEL_REF2VA,
            "missing": missing or None,
        },
        "input_images": opts("LoadImage", "image"),
        "input_videos": opts("LoadVideo", "file"),
        "input_audio": opts("LoadAudio", "audio"),
        "samplers": opts("KSampler", "sampler_name"),
        "schedulers": opts("KSampler", "scheduler"),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
