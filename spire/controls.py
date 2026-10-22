"""Control units for the neuron-isolation experiment.

`BinaryAct` is a stateless binary threshold with a straight-through estimator: it keeps the
binary, sparse output of a spiking unit but has no membrane, no leak and no reset.
`RateMatchedBinaryAct` additionally calibrates its threshold on data so that its activation
rate matches the firing rate of the LIF unit it replaces (`run_spire.py --binary_lif --rate_match`).
"""

import torch
import torch.nn as nn

# class-name markers of the spiking units built by spire.srp_spf.build_lif (SpikingJelly nodes
# and the multi-step wrapper); "IFNode" also matches "LIFNode" and "MultiStepLIFNode".
LIF_LIKE_CLASS_MARKERS = ("IFNode", "MultiStepLIF", "MultiStepIF", "_MultiStepWrapper")


class _BinaryStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, threshold)
        return (x > threshold).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        surrogate_window = (x - threshold).abs() < 1.0
        return grad_output * surrogate_window.to(dtype=grad_output.dtype), None


class BinaryAct(nn.Module):
    """Stateless binary threshold activation with a straight-through estimator."""

    def __init__(self, th: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor(float(th), dtype=torch.float32))

    def reset(self) -> None:
        """Compatibility no-op: the binary control has no membrane state."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _BinaryStraightThrough.apply(x, self.threshold.to(device=x.device, dtype=x.dtype))


class RateMatchedBinaryAct(nn.Module):
    """Stateless binary threshold whose threshold is calibrated on data so that the fraction
    of activations above it matches a target activation rate.

    Same straight-through estimator as `BinaryAct`; the threshold is set to the
    `(1 - target_rate)` quantile of the observed inputs instead of being fixed, which removes
    the sparsity confound when comparing against LIF at a matched firing rate.
    """

    def __init__(self, target_rate: float = 0.1, th: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor(float(th), dtype=torch.float32))
        self.target_rate = float(target_rate)

    def reset(self) -> None:
        """Compatibility no-op: the binary control has no membrane state."""

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> None:
        """Set the threshold to the (1 - target_rate) quantile of `x`."""
        flat = x.detach().reshape(-1).float()
        q = min(max(1.0 - self.target_rate, 0.0), 1.0)
        # torch.quantile caps the input size; subsample very large tensors.
        if flat.numel() > 4_000_000:
            idx = torch.randint(0, flat.numel(), (4_000_000,), device=flat.device)
            flat = flat[idx]
        thr = torch.quantile(flat, q)
        self.threshold = thr.to(self.threshold.device, self.threshold.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _BinaryStraightThrough.apply(x, self.threshold.to(device=x.device, dtype=x.dtype))


def is_lif_like(module: nn.Module) -> bool:
    class_name = type(module).__name__
    return any(marker in class_name for marker in LIF_LIKE_CLASS_MARKERS)


def replace_lif_with_binary(module: nn.Module, v_threshold: float = 1.0,
                            rate_matched: bool = False, target_rate: float = 0.1) -> nn.Module:
    """Recursively replace the LIF-like child modules with a binary control.

    With `rate_matched=False` the fixed-threshold `BinaryAct` is used. With `rate_matched=True`
    a `RateMatchedBinaryAct(target_rate)` is used, whose threshold is calibrated on data later
    (see `run_spire.py`).
    """
    for name, child in list(module.named_children()):
        if is_lif_like(child):
            if rate_matched:
                setattr(module, name, RateMatchedBinaryAct(target_rate=target_rate, th=v_threshold))
            else:
                setattr(module, name, BinaryAct(th=v_threshold))
        else:
            replace_lif_with_binary(child, v_threshold=v_threshold,
                                    rate_matched=rate_matched, target_rate=target_rate)
    return module


def count_lif_like_modules(module: nn.Module) -> int:
    """Count the LIF-like modules under `module` by class-name markers."""
    return sum(1 for child in module.modules() if child is not module and is_lif_like(child))


def count_binary_modules(module: nn.Module) -> int:
    """Count the binary-control modules (fixed or rate-matched) under `module`."""
    return sum(1 for child in module.modules()
               if isinstance(child, (BinaryAct, RateMatchedBinaryAct)))
