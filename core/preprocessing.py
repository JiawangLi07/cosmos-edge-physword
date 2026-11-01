"""Executable spatial and VAE normalization contract for single-device PW steps."""

import math
from dataclasses import asdict, dataclass, replace

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PreprocessingConfig:
    input_size: tuple[int, int]
    output_size: tuple[int, int] | None = None
    interpolation: str = "bilinear"
    align_corners: bool = False
    antialias: bool = False
    quantization: str = "round"
    pixel_scale: float = 127.5
    pixel_offset: float = -1.0
    pixel_device: str = "cpu"
    latent_mean: tuple[float, ...] = (0.0,) * 48
    latent_std: tuple[float, ...] = (1.0,) * 48

    def __post_init__(self):
        for name in ("input_size", "output_size"):
            value = getattr(self, name)
            if value is not None:
                value = tuple(value)
                if len(value) != 2 or any(type(x) is not int or x <= 0 for x in value):
                    raise ValueError("Spatial sizes must be positive integer (height, width)")
                object.__setattr__(self, name, value)
        if self.input_size is None:
            raise ValueError("Declare the original RGB input size, before external transforms")
        if self.interpolation not in {"bilinear", "bicubic"} or self.quantization not in {"round", "floor"}:
            raise ValueError("Unsupported RGB interpolation or uint8 quantization")
        if type(self.align_corners) is not bool or type(self.antialias) is not bool:
            raise ValueError("Interpolation switches must be booleans")
        if not math.isfinite(self.pixel_scale) or self.pixel_scale <= 0 or not math.isfinite(self.pixel_offset):
            raise ValueError("VAE pixel normalization must be finite with positive scale")
        if self.pixel_device not in {"cpu", "cuda"}:
            raise ValueError("Pin the pixel normalization device to cpu or current cuda device")
        for name in ("latent_mean", "latent_std"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.latent_mean) != 48 or len(self.latent_std) != 48:
            raise ValueError("Edge VAE normalization requires 48 channel values")
        if any(not math.isfinite(x) for x in (*self.latent_mean, *self.latent_std)) or any(
            x <= 0 for x in self.latent_std
        ):
            raise ValueError("VAE latent normalization must be finite with positive std")

    def contract(self):
        return {
            **asdict(self),
            "rgb": "TCHW uint8 RGB [0,255]; no crop or augmentation",
            "pixel_equation": "float32(rgb) / pixel_scale + pixel_offset",
            "latent_equation": "(raw_mode_latent - latent_mean) / latent_std",
            "vae_statistic": "posterior mode; no latent sampling",
        }

    def rgb(self, frames):
        if frames.ndim != 4 or frames.dtype != torch.uint8 or frames.shape[1] != 3:
            raise ValueError("Preprocessing requires original TCHW uint8 RGB")
        if tuple(frames.shape[-2:]) != self.input_size:
            raise ValueError("RGB input size differs from the frozen preprocessing contract")
        if self.output_size is None or self.output_size == self.input_size:
            return frames
        pixels = F.interpolate(
            frames.float(),
            size=self.output_size,
            mode=self.interpolation,
            align_corners=self.align_corners,
            antialias=self.antialias,
        ).clamp(0, 255)
        return (pixels.round() if self.quantization == "round" else pixels.floor()).to(torch.uint8)

    def batch(self, batch):
        return replace(
            batch,
            history=replace(batch.history, frames=self.rgb(batch.history.frames)),
            future_rgb=self.rgb(batch.future_rgb),
        )

    def video_pixels(self, frames):
        pixels = frames.to(self.pixel_device).permute(1, 0, 2, 3).unsqueeze(0).float()
        return pixels / self.pixel_scale + self.pixel_offset

    def normalize_latents(self, raw):
        if raw.ndim != 5 or raw.shape[1] != 48:
            raise ValueError("Raw Edge VAE latents must be [B,48,T,H,W]")
        mean = raw.new_tensor(self.latent_mean, dtype=torch.float32).view(1, 48, 1, 1, 1)
        std = raw.new_tensor(self.latent_std, dtype=torch.float32).view(1, 48, 1, 1, 1)
        return (raw.float() - mean) / std
