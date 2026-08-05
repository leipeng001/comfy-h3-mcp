# comfy-h3-mcp

A small MCP server for driving **MiniMax-H3** video+audio generation on a local
ComfyUI. Eight tools, not a general ComfyUI control plane — the point is to keep
the agent's context cost near zero for the one thing this rig actually does.

## Tools

| Tool | What it does |
|---|---|
| `h3_image_to_video` | Text-to-video, or keyframe-guided via `first_frame` / `last_frame` (fl2va model) |
| `h3_reference_to_video` | Prompt + reference images / videos / audio (ref2va model) |
| `job_status` | Poll a `prompt_id` → queued / running / completed / failed + output URLs |
| `job_cancel` | Drop from queue if pending, interrupt if running |
| `job_preview` | Return a contact sheet and soundtrack for a finished clip |
| `grab_reference` | Save a frame of a finished clip as a reusable reference still |
| `gpt_image_first_frame` | Generate and upload a GPT Image keyframe (metered API) |
| `list_assets` | H3 models present, and input files usable by name |

Generation takes minutes, so the two generate tools **submit and return a
`prompt_id` immediately**. There is no session state — the `prompt_id` is the
only handle, and ComfyUI already owns it.

## Acceleration presets

Both generation tools accept one `acceleration` value:

| Preset | Stack | Use |
|---|---|---|
| `off` | Native H3 path (global Sage still applies) | Native-trajectory A/B reference |
| `quality` | Sol-Attn, `tau=1.0`, edge blocks dense | Quality-sensitive accelerated renders |
| `balanced` (default) | Sol-Attn, `tau=1.3` + EasyCache `0.10` | Normal local generation |
| `fast` | Sol-Attn, `tau=1.5` + EasyCache `0.20` | Prompt and motion drafts |

These are inference-time approximations. They do not retrain or modify the
checkpoint, but they can change motion, fine detail, and audio for the same
seed. Keep `off` available as the reference and re-run important shots with
`quality` or `off` when fidelity matters.

The Sol-Attn custom node must be installed in ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-SolAttn_triton.git
```

`list_assets` reports whether both `SolAttnPatch` and native `EasyCache` are
loaded. Sol-Attn's Triton kernels compile for each new tensor shape, so the
first run at a resolution/duration can be slower than subsequent warm runs.

## ComfyUI GUI workflow

Load [`workflows/minimax_h3_sol_balanced.json`](workflows/minimax_h3_sol_balanced.json)
from ComfyUI's **Workflows → Open** menu. It is derived from ComfyUI's official
MiniMax-H3 text/image-to-video template and inserts the measured balanced path:

```text
UNETLoader → SolAttnPatch → EasyCache → scheduler + guider
```

The workflow opens at 0.4 MP, 5 seconds, and 20 RES multistep steps. Global
SageAttention remains controlled by the ComfyUI launch flag. The note inside
the workflow gives the `quality`, `balanced`, and `fast` settings.

## Setup

```bash
uv venv && uv pip install -e .
claude mcp add comfy-h3 -s user \
  -e COMFYUI_URL=http://127.0.0.1:8188 \
  -- /path/to/comfy-h3-mcp/.venv/bin/comfy-h3-mcp
```

`COMFYUI_URL` defaults to `http://127.0.0.1:8188`.

The optional `gpt_image_first_frame` tool also needs `OPENAI_API_KEY` in the
environment that launches the MCP server. It uses the metered GPT Image API;
the local H3 generation tools do not require that key.

## Timing

Generation takes **minutes**. Every submit returns `estimated_seconds` plus a
suggested poll interval, so a client knows the difference between "slow" and
"stuck". Measured on an RTX 4090 at 124 frames:

| Config | Time |
|---|---|
| 864×480, 20 steps, sage | 3m 43s |
| 864×480, 20 steps, sage + `balanced` | **2m 33s** |
| 864×480, 20 steps, sage + `fast` (warm) | **1m 52s** |
| 864×480, 24 steps, sage | 4m 11s |
| 1344×768, 30 steps, sage | 13m 13s |
| 1344×768, 30 steps, no sage | 18m 02s |

The estimator scales with pixels × steps × length. Its accelerated factors are
fitted to the two local measurements above. `balanced` skipped 5/20 denoiser
evaluations; `fast` skipped 7/20. Roughly 30 s is fixed model load/decode work,
which these DiT optimizations do not remove — the `nvfp4` text-encoder path is
emulated on this hardware.

**Draft small, finish large.** 864×480 is the template's draft setting; the
model's documented full-quality 16:9 target is ~1.0 MP (1344×768). Iterate
prompts at the default, then re-run keepers at 1344×768 with the same seed —
noting that a seed does not guarantee an identical image across a resolution
change, only a related composition.

## Sage attention

`list_assets` reports `sage_attention.global`, detected by inspecting the ComfyUI
process for `--use-sage-attention` (local servers only; `None` when it can't be
determined). **When sage is global, the per-call `sage_attention` parameter is
redundant** — the tool descriptions tell clients not to set it and not to
recommend enabling sage, which otherwise happens: an agent that can't see the
launch flag will report a normal run as "grinding" and advise a no-op fix.

Sage gives roughly **1.36×** here. It is a lossy approximation: same seed
produces a *different sample*, not the same one faster (SSIM 0.78 on luma,
audio differs too). Keep it in a fixed state once you lock a seed.

## Required models

Loaded by name, so they must be present:

- `models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- `models/vae/minimax_h3_video_vae_fp16.safetensors`
- `models/vae/minimax_h3_audio_vae_fp32.safetensors`
- `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`

`list_assets` reports anything missing.

## Notes on the model's constraints

These are enforced in `graphs.py`, mirroring `comfy_extras/nodes_minimax_h3.py`,
so callers see the real numbers up front rather than discovering that ComfyUI
snapped them:

- **Length** is a frame count at 24 fps, snapped **up** to the `17k+5` grid.
  124 ≈ 5.2 s. Trained range is roughly 124–362; longer is untested.
- **Canvas** is capped at 768×1344 pixels of area with each axis rounded to 32.
  Oversized requests are re-fitted via the model's own `adapt_canvas` rule, so
  1920×1080 becomes 1344×768.
- **References** are addressed positionally in the prompt as `<Picture i>`,
  `<Video k>`, `<Audio j>` — all **1-based per type**. Max 9 images, 3 videos,
  3 audio. A reference video's soundtrack is wired through automatically.
- **Prefer stills to reference videos.** `ref_videos` carry identity poorly and
  drag their own soundtrack into the output, fighting the audio the prompt asked
  for. Use `grab_reference` to lift a frame from an earlier clip instead: run
  `job_preview`, pick a tile off the contact sheet, and `grab_reference(
  prompt_id, tile=N)` saves that exact source frame at full resolution into
  ComfyUI's input folder, ready to pass as `ref_images`.
- `ref_image_size="max"` uses a 2048 px short edge for better identity fidelity
  but is several times slower, because reference tokens ride through every
  sampling step.

## Defaults

These mirror the official `video_minimax_h3_t2v.json` template, verified
node-for-node:

| | Value |
|---|---|
| Resolution | 864×480 — 16:9 at **0.4 MP**, rounded to 32 |
| Steps | **20** |
| Sampler | **`res_multistep`** |
| Scheduler | `simple`, `denoise=1.0` |
| Guidance | `BasicGuider` — no CFG, no negative conditioning |
| Sigma shift | **none** — `supported_models.py` already applies `shift=12.0` |

Pass `width`/`height` explicitly to override the megapixel calculation, or
raise `megapixels`. Note the cost: 1344×768 is 2.5× the pixels of the default,
and `res_multistep` is a higher-order sampler, so 20 steps here is not a
downgrade from 30 steps of `euler` — it is faster *and* comparable in quality.

`shift_video`/`shift_audio` default to `None`, which omits `MiniMaxH3SigmaShift`
entirely. Set either one to insert the node and override the model default.

## Graph shape

```
UNETLoader ─► Sol-Attn ─► EasyCache ─┐
CLIPLoader(minimax) ─┐           ├─► BasicGuider ─┐
VAELoader(video) ────┼─► MiniMaxH3{ImageToVideo,  │
VAELoader(audio) ────┘      ReferenceToVideo}     │
                        │ positive ───────────────┘
                        └─ latent ────────────────┐
                                                  │
RandomNoise ─┐                                    │
KSamplerSelect(res_multistep) ─┼─► SamplerCustomAdvanced ◄┘
BasicScheduler(simple, 20) ────┘         │
                                         ├─► VAEDecode(video vae) ─────┐
                                         └─► VAEDecodeAudio(audio vae) ┤
                                                                       ▼
                                                  CreateVideo(24fps) ─► SaveVideo
```

Both VAEs read the **joint AV latent directly** — the nested video/audio pair
needs no explicit split node. (`LTXVSeparateAVLatent` does work here despite the
name, but the template doesn't use it and neither do we.)

Autogrow reference inputs serialize as **dotted** API keys
(`ref_images.ref_image_0`, `ref_video_audios.ref_video_audio_0`), per
`finalize_prefix()` in `comfy_api/latest/_io.py`.

## Requires

MCP SDK **2.0+** (`MCPServer`; `FastMCP` was removed).
