"""A-0 stub: real spatial ×1.5 upsample of H3 video latent (not a no-op)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _upsample_video_latent(samples: torch.Tensor, scale: float, mode: str) -> torch.Tensor:
    """Upsample H×W of (B,C,T,H,W) or nested AV dict values; keep T."""
    if scale == 1.0:
        return samples.contiguous()

    def _resize(x: torch.Tensor, nh: int, nw: int) -> torch.Tensor:
        x = x.float()
        if mode == "nearest":
            y = F.interpolate(x, size=(nh, nw), mode="nearest")
        else:
            y = F.interpolate(x, size=(nh, nw), mode=mode, align_corners=False)
        return y.to(dtype=samples.dtype)

    if samples.ndim == 5:
        b, c, t, h, w = samples.shape
        x = samples.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        y = _resize(x, nh, nw)
        return y.reshape(b, t, c, nh, nw).permute(0, 2, 1, 3, 4).contiguous()
    if samples.ndim == 4:
        nh = max(1, int(round(samples.shape[-2] * scale)))
        nw = max(1, int(round(samples.shape[-1] * scale)))
        return _resize(samples, nh, nw).contiguous()
    raise ValueError(f"H3LatentSpatialUpsampleStub: unsupported latent shape {tuple(samples.shape)}")


class H3LatentSpatialUpsampleStub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "mode": (["bilinear", "nearest", "bicubic"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "up"
    CATEGORY = "aigc/h3"

    def up(self, latent, scale, mode):
        out = {}
        for key, val in latent.items():
            if isinstance(val, torch.Tensor):
                out[key] = _upsample_video_latent(val, float(scale), mode)
            else:
                out[key] = val
        return (out,)


NODE_CLASS_MAPPINGS = {
    "H3LatentSpatialUpsampleStub": H3LatentSpatialUpsampleStub,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentSpatialUpsampleStub": "H3 Latent Spatial Upsample Stub (×1.5)",
}
