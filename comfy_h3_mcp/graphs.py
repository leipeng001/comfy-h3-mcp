"""API-format graph construction for MiniMax-H3.

The frame-count and canvas math mirrors comfy_extras/nodes_minimax_h3.py so the
caller sees the real duration/resolution up front instead of discovering that
ComfyUI silently snapped them.

Model weights (8G playbook): DmitryDB NVFP4 compact via local `*_nf4.safetensors`
aliases. Requires Comfy ops.py all-NUL comfy_quant patch on 0.33.1.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

FPS = 24
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344

# Preferred non-pruned names (first existing wins). Override with env.
# DmitryDB NVFP4 compact (~10.86 GiB) is the 8G target; requires Comfy ops.py
# all-NUL comfy_quant fix (local patch on 0.33.1). Official full INT8 abandoned.
_REF2VA_CANDIDATES = (
    "minimax_h3_ref2va_nf4.safetensors",
    "minimax_h3_ref2va_nvfp4.safetensors",
    "minimax_h3_ref2va_nvfp4_compact.safetensors",
    "MiniMax-H3_Ref2VA-NVFP4.safetensors",
    "minimax_h3_ref2va_nvfp4_full.safetensors",
    "minimax_h3_ref2va_nvfp4_mixed.safetensors",
    "minimax_h3_ref2va.safetensors",
)
_FL2VA_CANDIDATES = (
    "minimax_h3_fl2va_nf4.safetensors",
    "minimax_h3_fl2va_nvfp4.safetensors",
    "minimax_h3_fl2va_nvfp4_compact.safetensors",
    "MiniMax-H3_FL2VA-NVFP4.safetensors",
    "minimax_h3_fl2va_nvfp4_full.safetensors",
    "minimax_h3_fl2va_nvfp4_mixed.safetensors",
    "minimax_h3_fl2va.safetensors",
)
_PRUNED_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
_PRUNED_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
# Skip partial HF downloads (official int8 is ~20–35 GiB; stubs are KB–GB mid-flight).
_MIN_WEIGHT_BYTES = 8 * 1024 ** 3

VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
CLIP_TYPE = "minimax"


def _video_vae_name() -> str:
    return os.environ.get("H3_VIDEO_VAE", "").strip() or VAE_VIDEO


def _audio_vae_name() -> str:
    return os.environ.get("H3_AUDIO_VAE", "").strip() or VAE_AUDIO


_REMOTE_REF2VA = "minimax_h3_ref2va_int8_convrot.safetensors"
_REMOTE_FL2VA = "minimax_h3_fl2va_int8_convrot.safetensors"
_REMOTE_TE = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"


def _text_encoder_name() -> str:
    return os.environ.get("H3_TEXT_ENCODER", "").strip() or TEXT_ENCODER


def _is_remote_stack() -> bool:
    return os.environ.get("H3_REMOTE", "").strip().lower() in {"1", "true", "yes"}


# Back-compat aliases — resolved at import / first use
MODEL_FL2VA = _PRUNED_FL2VA
MODEL_REF2VA = _PRUNED_REF2VA


def _diffusion_models_dir() -> Path:
    override = os.environ.get("H3_DIFFUSION_MODELS_DIR", "").strip()
    if override:
        return Path(override)
    return Path(r"D:\aigc\ComfyUI\ComfyUI\models\diffusion_models")


def _weight_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= _MIN_WEIGHT_BYTES
    except OSError:
        return False


def _pick_model(candidates: tuple[str, ...], *, pruned: str, kind: str) -> str:
    env_key = "H3_REF2VA_MODEL" if kind == "ref2va" else "H3_FL2VA_MODEL"
    env_name = os.environ.get(env_key, "").strip()
    models_dir = _diffusion_models_dir()
    if env_name:
        # Remote Comfy: trust the filename without checking the local D: disk.
        if _is_remote_stack() or not models_dir.is_dir():
            return env_name
        p = models_dir / env_name
        if _weight_ready(p):
            return env_name
        raise FileNotFoundError(
            f"{env_key}={env_name} missing or incomplete under {models_dir} "
            f"(need >= {_MIN_WEIGHT_BYTES // 1024**3} GiB)"
        )

    if _is_remote_stack():
        return _REMOTE_REF2VA if kind == "ref2va" else _REMOTE_FL2VA

    for name in candidates:
        if _weight_ready(models_dir / name):
            return name

    allow_pruned = os.environ.get("H3_ALLOW_PRUNED", "").strip() in {"1", "true", "yes"}
    if allow_pruned and _weight_ready(models_dir / pruned):
        print(
            f"h3_model_warn using PRUNED {pruned} "
            f"(set a non-pruned file or unset H3_ALLOW_PRUNED). "
            f"Tried: {', '.join(candidates)}"
        )
        return pruned

    tried = ", ".join(candidates)
    raise FileNotFoundError(
        f"No non-pruned MiniMax H3 {kind} weights under {models_dir}.\n"
        f"Tried: {tried}\n"
        f"Pruned fallback exists as {pruned} but is blocked "
        f"(pruned drops logic/faces per 8G playbook).\n"
        f"Download NF4 or v2 INT8 full weights, or set {env_key}=filename, "
        f"or temporarily H3_ALLOW_PRUNED=1 for smoke tests only."
    )


def resolve_h3_models() -> tuple[str, str]:
    """Return (ref2va_name, fl2va_name); updates module MODEL_* aliases."""
    global MODEL_REF2VA, MODEL_FL2VA, TEXT_ENCODER
    MODEL_REF2VA = _pick_model(_REF2VA_CANDIDATES, pruned=_PRUNED_REF2VA, kind="ref2va")
    MODEL_FL2VA = _pick_model(_FL2VA_CANDIDATES, pruned=_PRUNED_FL2VA, kind="fl2va")
    te = os.environ.get("H3_TEXT_ENCODER", "").strip()
    if te:
        TEXT_ENCODER = te
    elif _is_remote_stack():
        TEXT_ENCODER = _REMOTE_TE
    return MODEL_REF2VA, MODEL_FL2VA


def apply_model_stack_from_params(params: dict | None) -> None:
    """Push shot_params model names into env for resolve_h3_models (remote INT8)."""
    if not params:
        return
    if params.get("comfy_backend") == "remote" or params.get("h3_ref2va_model"):
        os.environ["H3_REMOTE"] = "1"
    else:
        os.environ.pop("H3_REMOTE", None)
        for k in ("H3_REF2VA_MODEL", "H3_FL2VA_MODEL", "H3_TEXT_ENCODER"):
            # Only clear if we previously set remote names; leave user env alone
            # when params carry no override.
            pass
    if params.get("h3_ref2va_model"):
        os.environ["H3_REF2VA_MODEL"] = str(params["h3_ref2va_model"])
    if params.get("h3_fl2va_model"):
        os.environ["H3_FL2VA_MODEL"] = str(params["h3_fl2va_model"])
    if params.get("h3_text_encoder"):
        os.environ["H3_TEXT_ENCODER"] = str(params["h3_text_encoder"])


# Resolve eagerly when diffusion dir exists; keep pruned names if resolve fails
# so import still works for math helpers — build_* will re-resolve.
try:
    if _diffusion_models_dir().is_dir():
        resolve_h3_models()
except FileNotFoundError:
    pass


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
    width: int,
    height: int,
    length: int,
    steps: int,
    sage: bool,
    acceleration: str = "off",
) -> dict[str, Any]:
    """Rough wall-clock estimate so callers know whether to expect 3 min or 20."""
    rate = SECS_PER_PIXEL_STEP_SAGE if sage else SECS_PER_PIXEL_STEP_PLAIN
    per_step = width * height * rate * (length / 124)
    speedup = ACCELERATION_ESTIMATED_SPEEDUP[acceleration]
    total = per_step * steps / speedup + MODEL_LOAD_COLD_SECS
    return {
        "estimated_seconds": round(total),
        "estimated": f"~{int(total // 60)}m {int(total % 60):02d}s",
        "seconds_per_step": round(per_step, 1),
        "basis": (
            "RTX 4090 measurements; scales with pixels x steps x length; "
            f"{acceleration} acceleration estimate"
        ),
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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _two_pass_enabled(*, profile: str | None = None) -> bool:
    """REFINE_TWO_PASS only applies when profile is refine (never draft)."""
    if not _env_flag("REFINE_TWO_PASS"):
        return False
    if profile is None:
        profile = os.environ.get("H3_RESOURCE_PROFILE", "").strip().lower()
    if profile and profile != "refine":
        return False
    # If profile unset but flag on, allow (API/MCP callers set profile via env).
    return True


def _split_sigma_start(steps: int) -> int:
    """Pass1/pass2 cut for SplitSigmas.

    Goldfish: BasicScheduler steps=12, SplitSigmas step=8 (pass1 ≈ 2/3).
    Spatial ×1.5 needs pass1 far enough along — early cuts (e.g. 8/20) leave a
    mid-noise latent that upscale destroys; ~≥50–60% of steps on pass1 works.

    Explicit ``H3_SPLIT_SIGMA_START`` is honored as-is (no auto-nudge).
    """
    raw = os.environ.get("H3_SPLIT_SIGMA_START", "").strip()
    if raw:
        split = int(raw)
        return max(1, min(steps - 1, split))
    # Default: first ~60% on pass1 (goldfish 8/12).
    return max(1, min(steps - 1, int(round(steps * 0.6))))


def _pass2_manual_sigmas() -> str | None:
    """Pass2 sigma schedule after spatial upscale.

    Mid-SplitSigmas + ×1.5 still yields soft mush / no subject even when not
    tan/grid. Official Upscaler example restarts pass2 at high sigma.
    Default: that ManualSigmas list. Set ``H3_PASS2_MANUAL_SIGMAS=`` empty
    string only when intentionally continuing SplitSigmas-low (identity upscale).
    """
    if "H3_PASS2_MANUAL_SIGMAS" in os.environ:
        return os.environ["H3_PASS2_MANUAL_SIGMAS"].strip() or None
    return "0.9035, 0.6316, 0.3158, 0.0000"


def _spatial_upscale(mode: str) -> bool:
    return (mode or "").strip().lower() not in {"none", "identity", "1.0", ""}


def _latent_upscale_node(
    g: GraphBuilder,
    *,
    latent: str,
    mode: str,
) -> str:
    """Spatial upscale between passes. Separates AV nested latent like 金鱼图.

    stub = real H/W×1.5 interpolate (VRAM-true); real = MinimaxH3LatentUpscaler3D.
    """
    mode = (mode or "stub").strip().lower()
    sep = g.add("LTXVSeparateAVLatent", av_latent=[latent, 0])
    # sep outputs: video_latent=0, audio_latent=1
    if mode == "real":
        model_name = os.environ.get(
            "H3_LATENT_UPSCALER",
            "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        )
        up_vid = g.add(
            "MinimaxH3LatentUpscaler3D",
            latent=[sep, 0],
            model_name=model_name,
            mode="scale by multiplier",
            **{"mode.scale": 1.5},
            align=32,
            # Official example / goldfish last False: temporal chunking off.
            enable_temporal_chunking=False,
            force_unload=False,
            device="cuda",
            precision="fp16",
        )
    elif mode in {"none", "identity", "1.0"}:
        # T5: keep AV split/concat + pass2, no spatial enlarge.
        up_vid = g.add(
            "H3LatentSpatialUpsampleStub",
            latent=[sep, 0],
            scale=1.0,
            mode="nearest",
        )
    else:
        up_vid = g.add(
            "H3LatentSpatialUpsampleStub",
            latent=[sep, 0],
            scale=1.5,
            mode="bilinear",
        )
    return g.add(
        "LTXVConcatAVLatent",
        video_latent=[up_vid, 0],
        audio_latent=[sep, 1],
    )


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
    denoise: float = 1.0,
    two_pass: bool | None = None,
    upscaler: str | None = None,
) -> None:
    """Sampler -> decode -> mux -> save, mirroring video_minimax_h3_t2v.json.

    Uses the SamplerCustomAdvanced path with a BasicGuider: H3 emits only
    positive conditioning, and the guider runs it with no CFG rather than
    faking an unconditional branch. Both VAEs read the joint AV latent
    directly - the nested video/audio pair needs no explicit split node.

    denoise<1.0 enables partial re-sample (v1.5 refine second pass) when the
    latent already holds an encoded init video/frame.

    When two_pass is enabled, delegates to _tail_two_pass (SplitSigmas +
    latent upscale + second sampler). CreateVideo only after pass2 decode.
    """
    if two_pass is None:
        two_pass = _two_pass_enabled()
    if two_pass:
        _tail_two_pass(
            g,
            model=model,
            positive=positive,
            latent=latent,
            video_vae=video_vae,
            audio_vae=audio_vae,
            seed=seed,
            steps=steps,
            sampler_name=sampler_name,
            scheduler=scheduler,
            filename_prefix=filename_prefix,
            denoise=denoise,
            upscaler=upscaler
            or os.environ.get("H3_TWO_PASS_UPSCALER", "stub").strip()
            or "stub",
        )
        return

    noise = g.add("RandomNoise", noise_seed=seed)
    guider = g.add("BasicGuider", model=[model, 0], conditioning=[positive, 0])
    sampler = g.add("KSamplerSelect", sampler_name=sampler_name)
    sigmas = g.add(
        "BasicScheduler",
        model=[model, 0],
        scheduler=scheduler,
        steps=steps,
        denoise=float(denoise),
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


def _tail_two_pass(
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
    denoise: float = 1.0,
    upscaler: str = "stub",
) -> None:
    """Two-pass: pass1 → (optional spatial ×1.5) → pass2 → decode → CreateVideo.

    Spatial ×1.5 path (quality): **full** pass1 denoise → Upscaler → pass2
    ManualSigmas high-σ refine. Mid-SplitSigmas + ×1.5 yields mush / no subject
    (or tan/grid if split too early). Identity upscale keeps SplitSigmas high/low.
    """
    noise = g.add("RandomNoise", noise_seed=seed)
    guider = g.add("BasicGuider", model=[model, 0], conditioning=[positive, 0])
    sampler = g.add("KSamplerSelect", sampler_name=sampler_name)
    sigmas = g.add(
        "BasicScheduler",
        model=[model, 0],
        scheduler=scheduler,
        steps=steps,
        denoise=float(denoise),
    )
    spatial = _spatial_upscale(upscaler)
    if spatial:
        # Full low-res denoise before learned/stub spatial enlarge.
        pass1_sigmas_ref = [sigmas, 0]
    else:
        split = _split_sigma_start(steps)
        split_node = g.add("SplitSigmas", sigmas=[sigmas, 0], step=split)
        pass1_sigmas_ref = [split_node, 0]
    pass1 = g.add(
        "SamplerCustomAdvanced",
        noise=[noise, 0],
        guider=[guider, 0],
        sampler=[sampler, 0],
        sigmas=pass1_sigmas_ref,
        latent_image=[latent, 1],
    )
    up = _latent_upscale_node(g, latent=pass1, mode=upscaler)
    if spatial:
        manual = _pass2_manual_sigmas()
        if not manual:
            raise ValueError(
                "spatial upscale requires pass2 ManualSigmas "
                "(unset H3_PASS2_MANUAL_SIGMAS or set a sigma list)"
            )
        pass2_sigmas = g.add("ManualSigmas", sigmas=manual)
        pass2_sigmas_ref = [pass2_sigmas, 0]
    else:
        pass2_sigmas_ref = [split_node, 1]
    pass2 = g.add(
        "SamplerCustomAdvanced",
        noise=[noise, 0],
        guider=[guider, 0],
        sampler=[sampler, 0],
        sigmas=pass2_sigmas_ref,
        latent_image=[up, 0],
    )
    frames = g.add("VAEDecode", samples=[pass2, 0], vae=[video_vae, 0])
    audio = g.add("VAEDecodeAudio", samples=[pass2, 0], vae=[audio_vae, 0])
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

SOL_NODE = "SolAttnPatch"
EASYCACHE_NODE = "EasyCache"
ACCELERATION_PRESETS: dict[str, dict[str, Any]] = {
    "off": {},
    # Sol-Attn only, with a denser threshold and sensitive edge blocks exact.
    "quality": {
        "sol_tau": 1.0,
        "dense_blocks": "0,-1",
    },
    # Sol-Attn plus a low EasyCache threshold: useful default for local drafts.
    "balanced": {
        "sol_tau": 1.3,
        "dense_blocks": "0,-1",
        "cache_threshold": 0.10,
    },
    # More sparsity and ComfyUI's standard cache threshold. Approximate output.
    "fast": {
        "sol_tau": 1.5,
        "dense_blocks": "0,-1",
        "cache_threshold": 0.20,
    },
}
ACCELERATION_ESTIMATED_SPEEDUP = {
    "off": 1.0,
    "quality": 1.18,
    # Sampling-path factors fitted from the local 4090 measurements below;
    # fixed model-load/decode time is added separately by estimate_runtime.
    "balanced": 1.57,
    "fast": 2.35,
}


def _apply_acceleration(g: GraphBuilder, model: str, preset: str) -> str:
    """Patch the model with the selected inference-only acceleration stack."""
    config = ACCELERATION_PRESETS[preset]
    if "sol_tau" in config:
        model = g.add(
            SOL_NODE,
            model=[model, 0],
            tau=config["sol_tau"],
            start_percent=0.20,
            end_percent=0.90,
            min_tokens=4096,
            int8_qk=True,
            int8_pv=True,
            sink_conditioning="exact_kv_and_rows",
            morton=False,
            morton_curve="2d_frame",
            dense_blocks=config["dense_blocks"],
            verbose=False,
            use_tma=False,
        )
    if "cache_threshold" in config:
        model = g.add(
            EASYCACHE_NODE,
            model=[model, 0],
            reuse_threshold=config["cache_threshold"],
            start_percent=0.15,
            end_percent=0.90,
            verbose=False,
        )
    return model


def _common_loaders(
    g: GraphBuilder,
    unet: str,
    shift_video: float | None,
    shift_audio: float | None,
    sage_attention: str = "disabled",
    acceleration: str = "balanced",
):
    use_hybrid = _env_flag("H3_USE_HYBRID_LOADER")
    if use_hybrid:
        resolve_h3_models()
        base = os.environ.get("H3_FL2VA_MODEL", "").strip() or MODEL_FL2VA
        overlay = os.environ.get("H3_REF2VA_MODEL", "").strip() or MODEL_REF2VA
        model = g.add(
            "MinimaxH3_HybridLoader",
            base_model=base,
            overlay_model=overlay,
        )
    else:
        model = g.add("UNETLoader", unet_name=unet, weight_dtype="default")

    if sage_attention != "disabled":
        model = g.add(SAGE_NODE, model=[model, 0], sage_attention=sage_attention)

    # Optional LoRA (e.g. H3 turbo). Applied after sage, before sigma-shift / Sol.
    lora_name = os.environ.get("H3_LORA", "").strip()
    if lora_name:
        strength = float(os.environ.get("H3_LORA_STRENGTH", "1.0") or "1.0")
        model = g.add(
            "LoraLoaderModelOnly",
            model=[model, 0],
            lora_name=lora_name,
            strength_model=strength,
        )

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

    model = _apply_acceleration(g, model, acceleration)

    clip = g.add("CLIPLoader", clip_name=_text_encoder_name(), type=CLIP_TYPE, device="default")
    video_vae = g.add("VAELoader", vae_name=_video_vae_name())
    audio_vae = g.add("VAELoader", vae_name=_audio_vae_name())
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
    acceleration: str = "balanced",
    denoise: float = 1.0,
    two_pass: bool | None = None,
    upscaler: str | None = None,
) -> dict[str, Any]:
    resolve_h3_models()
    g = GraphBuilder()
    model, clip, video_vae, audio_vae = _common_loaders(
        g, MODEL_FL2VA, shift_video, shift_audio, sage_attention, acceleration
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
        denoise=denoise,
        two_pass=two_pass,
        upscaler=upscaler,
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
    acceleration: str = "balanced",
    denoise: float = 1.0,
    two_pass: bool | None = None,
    upscaler: str | None = None,
) -> dict[str, Any]:
    resolve_h3_models()
    g = GraphBuilder()
    model, clip, video_vae, audio_vae = _common_loaders(
        g, MODEL_REF2VA, shift_video, shift_audio, sage_attention, acceleration
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
        denoise=denoise,
        two_pass=two_pass,
        upscaler=upscaler,
    )
    return g.nodes
