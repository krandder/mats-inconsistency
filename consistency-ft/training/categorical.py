"""Full-vocabulary categorical sampling without a float32 exponential race.

Identity inspection imports no Torch. Sampling uses at least FP32 softmax/log-softmax,
uses a normalized FP64 cumulative sum and one FP64 uniform variate, and never
filters, clips, retries or repairs an output. Finite-precision CDF/uniform
resolution still applies; this is not exact real-number probability arithmetic.
"""

import hashlib
import math
from pathlib import Path


def categorical_sampler_identity():
    return {"algorithm": "softmax-at-least-fp32-fp64-cdf-uniform-right-v1",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def normalized_cdf(weights):
    """Accumulate every FP32/FP64 weight, including zeros, and normalize in FP64."""
    import torch

    if (weights.ndim != 1 or weights.numel() < 1 or weights.dtype not in (torch.float32, torch.float64)
            or weights.device.type not in ("cpu", "cuda")
            or not torch.isfinite(weights).all() or torch.any(weights < 0)):
        raise ValueError("categorical weights require a finite nonnegative FP32/FP64 vector")
    cumulative = weights.detach().to(dtype=torch.float64).cumsum(dim=0)
    total = cumulative[-1]
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("categorical weights require positive finite total mass")
    return cumulative / total


def cdf_index(cdf, uniform):
    """Choose the first CDF value strictly above a supplied uniform in [0, 1).

    Keeping this numerical primitive separate permits exact boundary checks
    without replacing the production RNG or manufacturing a generated response.
    """
    import torch

    if (cdf.ndim != 1 or cdf.numel() < 1 or cdf.dtype != torch.float64
            or not torch.isfinite(cdf).all() or torch.any(cdf < 0)
            or torch.any(cdf[1:] < cdf[:-1]) or cdf[-1] != 1):
        raise ValueError("categorical CDF must be finite, monotone and end at one")
    if (uniform.ndim != 0 or uniform.dtype != torch.float64 or uniform.device != cdf.device
            or not torch.isfinite(uniform) or not 0 <= uniform < 1):
        raise ValueError("categorical uniform must be a same-device FP64 scalar in [0, 1)")
    index = int(torch.searchsorted(cdf, uniform, right=True).item())
    if index >= cdf.numel():
        raise ValueError("categorical CDF search escaped its support")
    return index


def sample_categorical(logits, *, generator):
    """Return a token and selected log-softmax, promoting low precision to FP32.

    Existing FP64 logits retain their normalization precision. Qwen's runtime
    supplies FP32 logits; BF16/FP16 generic forwards are promoted to FP32.
    """
    import torch

    if (logits.ndim != 1 or logits.numel() < 1 or not torch.is_floating_point(logits)
            or logits.device.type not in ("cpu", "cuda") or generator.device != logits.device
            or not torch.isfinite(logits).all()):
        raise ValueError("categorical logits require a finite vector and same-device generator")
    with torch.no_grad():
        normalized_logits = logits if logits.dtype == torch.float64 else logits.float()
        cdf = normalized_cdf(normalized_logits.softmax(dim=-1))
        uniform = torch.rand((), dtype=torch.float64, device=logits.device, generator=generator)
        token = cdf_index(cdf, uniform)
        logprob = float(normalized_logits.log_softmax(dim=-1)[token].item())
        if not math.isfinite(logprob):
            raise ValueError("nonfinite sampled log probability")
        return token, logprob
