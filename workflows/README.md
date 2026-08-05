# GUI workflows

`minimax_h3_sol_balanced.json` is a loadable ComfyUI workflow derived from the
official `video_minimax_h3_t2v.json` template in
`Comfy-Org/workflow_templates`.

It preserves the official MiniMax-H3 subgraph, resolution selector, duration
snapping, RES multistep sampler, joint video/audio decoding, and SaveVideo
output. The model route is changed to:

```text
UNETLoader → SolAttnPatch → EasyCache → BasicScheduler / BasicGuider
```

Loaded settings match the MCP's `balanced` preset:

- Sol-Attn: `tau=1.3`, active from 20% through 90%, INT8 QK, exact conditioning
  and audio rows, first/last transformer blocks dense.
- EasyCache: threshold `0.10`, active from 15% through 90%.

Required custom node:

- <https://github.com/kijai/ComfyUI-SolAttn_triton>

EasyCache is included in current ComfyUI builds. SageAttention is enabled by
the ComfyUI process launch flag on `wopr`; it is intentionally not duplicated
inside the workflow.
