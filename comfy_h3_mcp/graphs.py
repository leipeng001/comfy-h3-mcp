"""API-format graph construction for MiniMax-H3.

The frame-count and canvas math mirrors comfy_extras/nodes_minimax_h3.py so the
caller sees the real duration/resolution up front instead of discovering that
ComfyUI silently snapped them.
"""

from __future__ import annotations

import math
from typing import Any

FPS = 24
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344

MODEL_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
MODEL_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
CLIP_TYPE = "minimax"


def align_frame_count(n: int) -> int:
    """Snap up to the model's 17k+5 grid."""
    n = max(5, n)
    while n % 17 != 5:
        n += 1
    return n


def seconds_to_frames(seconds: float) -> int:
    return align_frame_count(int(round(seconds * FPS)))


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """768 short edge, 768*1344 area cap, each axis rounded to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nom_w, nom_h = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


# Measured on an RTX 4090 at 124 frames: seconds per sampling step per pixel.
# Both figures come from real runs (864x480/20 and 1344x768/30) and predict the
# other configuration to within ~8%, so treat them as a rough guide, not a clock.
SECS_PER_PIXEL_STEP_SAGE = 2.33e-5
SECS_PER_PIXEL_STEP_PLAIN = 3.29e-5
MODEL_LOAD_COLD_SECS = 30.0

ASPECT_16_9 = (16, 9)


def estimate_runtime(
    width: int, height: int, length: int, steps: int, sage: bool
) -> dict[str, Any]:
    """Rough wall-clock estimate so callers know whether to expect 3 min or 20."""
    rate = SECS_PER_PIXEL_STEP_SAGE if sage else SECS_PER_PIXEL_STEP_PLAIN
    per_step = width * height * rate * (length / 124)
    total = per_step * steps + MODEL_LOAD_COLD_SECS
    return {
        "estimated_seconds": round(total),
        "estimated": f"~{int(total // 60)}m {int(total % 60):02d}s",
        "seconds_per_step": round(per_step, 1),
        "basis": "RTX 4090 measurements; scales with pixels x steps x length",
    }


def resolution_for(megapixels: float, w_ratio: int = 16, h_ratio: int = 9,
                   multiple: int = 32) -> tuple[int, int]:
    """Mirror of the core ResolutionSelector node the template drives width/height from."""
    total = megapixels * 1024 * 1024
    scale = math.sqrt(total / (w_ratio * h_ratio))
    return (round(w_ratio * scale / multiple) * multiple,
            round(h_ratio * scale / multiple) * multiple)


def snap_axis(v: int) -> int:
    return max(CANVAS_MULTIPLE, round(v / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


class GraphBuilder:
    """Incrementing-id API-format graph."""

    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}
        self._n = 0

    def add(self, class_type: str, **inputs: Any) -> str:
        self._n += 1
        nid = str(self._n)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid


def _tail(
    g: GraphBuilder,
    *,
    model: str,
    positive: str,
    latent: str,
    video_vae: str,
    audio_vae: str,
    seed: int,
    steps: int,
    sampler_name: str,
    scheduler: str,
    filename_prefix: str,
) -> None:
    """Sampler -> decode -> mux -> save, mirroring video_minimax_h3_t2v.json.

    Uses the SamplerCustomAdvanced path with a BasicGuider: H3 emits only
    positive conditioning, and the guider runs it with no CFG rather than
    faking an unconditional branch. Both VAEs read the joint AV latent
    directly - the nested video/audio pair needs no explicit split node.
    """
    noise = g.add("RandomNoise", noise_seed=seed)
    guider = g.add("BasicGuider", model=[model, 0], conditioning=[positive, 0])
    sampler = g.add("KSamplerSelect", sampler_name=sampler_name)
    sigmas = g.add(
        "BasicScheduler",
        model=[model, 0],
        scheduler=scheduler,
        steps=steps,
        denoise=1.0,
    )
    sampled = g.add(
        "SamplerCustomAdvanced",
        noise=[noise, 0],
        guider=[guider, 0],
        sampler=[sampler, 0],
        sigmas=[sigmas, 0],
        latent_image=[latent, 1],
    )
    frames = g.add("VAEDecode", samples=[sampled, 0], vae=[video_vae, 0])
    audio = g.add("VAEDecodeAudio", samples=[sampled, 0], vae=[audio_vae, 0])
    video = g.add(
        "CreateVideo", images=[frames, 0], fps=FPS, bit_depth=8, audio=[audio, 0]
    )
    g.add(
        "SaveVideo",
        video=[video, 0],
        filename_prefix=filename_prefix,
        format="auto",
        codec="auto",
    )


# KJNodes registers this with an upstream typo in the id ("Pathch").
SAGE_NODE = "PathchSageAttentionKJ"
SAGE_MODES = (
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
)


def _common_loaders(
    g: GraphBuilder,
    unet: str,
    shift_video: float | None,
    shift_audio: float | None,
    sage_attention: str = "disabled",
):
    model = g.add("UNETLoader", unet_name=unet, weight_dtype="default")

    if sage_attention != "disabled":
        model = g.add(SAGE_NODE, model=[model, 0], sage_attention=sage_attention)

    # The template omits MiniMaxH3SigmaShift entirely: supported_models.py
    # already applies shift=12.0 for this architecture. Only add the node when
    # the caller explicitly wants a different value.
    if shift_video is not None or shift_audio is not None:
        model = g.add(
            "MiniMaxH3SigmaShift",
            model=[model, 0],
            shift_video=12.0 if shift_video is None else shift_video,
            shift_audio=3.0 if shift_audio is None else shift_audio,
        )

    clip = g.add("CLIPLoader", clip_name=TEXT_ENCODER, type=CLIP_TYPE, device="default")
    video_vae = g.add("VAELoader", vae_name=VAE_VIDEO)
    audio_vae = g.add("VAELoader", vae_name=VAE_AUDIO)
    return model, clip, video_vae, audio_vae


def build_image_to_video(
    *,
    prompt: str,
    width: int,
    height: int,
    length: int,
    first_frame: str | None,
    last_frame: str | None,
    seed: int,
    steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float | None,
    shift_audio: float | None,
    filename_prefix: str,
    sage_attention: str = "disabled",
) -> dict[str, Any]:
    g = GraphBuilder()
    model, clip, video_vae, audio_vae = _common_loaders(
        g, MODEL_FL2VA, shift_video, shift_audio, sage_attention
    )

    cond_inputs: dict[str, Any] = {
        "clip": [clip, 0],
        "vae": [video_vae, 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
    }
    if first_frame:
        cond_inputs["first_frame"] = [g.add("LoadImage", image=first_frame), 0]
    if last_frame:
        cond_inputs["last_frame"] = [g.add("LoadImage", image=last_frame), 0]

    cond = g.add("MiniMaxH3ImageToVideo", **cond_inputs)
    _tail(
        g,
        model=model,
        positive=cond,
        latent=cond,
        video_vae=video_vae,
        audio_vae=audio_vae,
        seed=seed,
        steps=steps,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    )
    return g.nodes


def build_reference_to_video(
    *,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_images: list[str],
    ref_videos: list[str],
    ref_audios: list[str],
    ref_image_size: str,
    seed: int,
    steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float | None,
    shift_audio: float | None,
    filename_prefix: str,
    sage_attention: str = "disabled",
) -> dict[str, Any]:
    g = GraphBuilder()
    model, clip, video_vae, audio_vae = _common_loaders(
        g, MODEL_REF2VA, shift_video, shift_audio, sage_attention
    )

    cond_inputs: dict[str, Any] = {
        "clip": [clip, 0],
        "vae": [video_vae, 0],
        "audio_vae": [audio_vae, 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
        "ref_image_size": ref_image_size,
    }

    # Autogrow inputs expand to dotted, 0-indexed keys: finalize_prefix() joins
    # the autogrow input's own id with each generated name.
    for i, name in enumerate(ref_images):
        cond_inputs[f"ref_images.ref_image_{i}"] = [g.add("LoadImage", image=name), 0]

    for i, name in enumerate(ref_videos):
        loaded = g.add("LoadVideo", file=name)
        parts = g.add("GetVideoComponents", video=[loaded, 0])
        cond_inputs[f"ref_videos.ref_video_{i}"] = [parts, 0]
        # The soundtrack rides along on the same index, which is exactly the
        # pairing MiniMaxH3ReferenceToVideo expects.
        cond_inputs[f"ref_video_audios.ref_video_audio_{i}"] = [parts, 1]

    for i, name in enumerate(ref_audios):
        cond_inputs[f"ref_audios.ref_audio_{i}"] = [g.add("LoadAudio", audio=name), 0]

    cond = g.add("MiniMaxH3ReferenceToVideo", **cond_inputs)
    _tail(
        g,
        model=model,
        positive=cond,
        latent=cond,
        video_vae=video_vae,
        audio_vae=audio_vae,
        seed=seed,
        steps=steps,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    )
    return g.nodes
