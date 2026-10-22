from typing import List, Tuple, Optional
import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== SpikingJelly backend (preferred) =====
try:
    from spikingjelly.activation_based import neuron, surrogate, functional
except Exception:  # pragma: no cover
    neuron = None
    surrogate = None
    functional = None


# ----------------------------
# Utilities
# ----------------------------
def _build_surrogate(name: str = "atan", alpha: float = 2.0):
    """Create a spikingjelly surrogate function instance (best-effort across versions)."""
    if surrogate is None:
        return None
    n = (name or "atan").lower()
    # common names in spikingjelly.activation_based.surrogate
    if hasattr(surrogate, "ATan") and n in {"atan", "arctan"}:
        try:
            return surrogate.ATan(alpha=alpha)
        except TypeError:
            return surrogate.ATan()
    if hasattr(surrogate, "Sigmoid") and n in {"sigmoid"}:
        try:
            return surrogate.Sigmoid(alpha=alpha)
        except TypeError:
            return surrogate.Sigmoid()
    if hasattr(surrogate, "PiecewiseQuadratic") and n in {"pwq", "piecewise", "piecewise_quadratic"}:
        try:
            return surrogate.PiecewiseQuadratic(alpha=alpha)
        except TypeError:
            return surrogate.PiecewiseQuadratic()
    if hasattr(surrogate, "Triangle") and n in {"tri", "triangle"}:
        try:
            return surrogate.Triangle(alpha=alpha)
        except TypeError:
            return surrogate.Triangle()
    # fallback
    if hasattr(surrogate, "ATan"):
        try:
            return surrogate.ATan(alpha=alpha)
        except TypeError:
            return surrogate.ATan()
    return None


class _MultiStepWrapper(nn.Module):
    """Fallback wrapper to run a single-step spikingjelly node in multi-step mode."""
    def __init__(self, node: nn.Module):
        super().__init__()
        self.node = node

    def reset(self):
        if hasattr(self.node, "reset"):
            self.node.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Ts, ...]
        outs = []
        Ts = x.shape[0]
        for t in range(Ts):
            outs.append(self.node(x[t]))
        return torch.stack(outs, dim=0)


def reset_net(m: nn.Module):
    """Best-effort reset for spikingjelly nodes/modules."""
    if functional is not None:
        try:
            functional.reset_net(m)
            return
        except Exception:
            pass
    if hasattr(m, "reset"):
        try:
            m.reset()
        except Exception:
            pass


def build_lif(
    tau: float = 2.0,
    v_threshold: float = 1.0,
    detach_reset: bool = True,
    surrogate_name: str = "atan",
    surrogate_alpha: float = 2.0,
) -> nn.Module:
    """
    Build a multi-step LIF node using spikingjelly (preferred).
    Returns a module that accepts input with shape [Ts, ...] and outputs [Ts, ...].
    """
    if neuron is None:
        raise ImportError(
            "spikingjelly is not installed. Please install spikingjelly "
            "(e.g., pip install spikingjelly) or switch to snntorch backend."
        )

    surr = _build_surrogate(surrogate_name, surrogate_alpha)

    # Prefer unified LIFNode with step_mode='m' if available
    if hasattr(neuron, "LIFNode"):
        cls = neuron.LIFNode
        sig = inspect.signature(cls)
        kw = {}
        if "tau" in sig.parameters:
            kw["tau"] = float(tau)
        if "v_threshold" in sig.parameters:
            kw["v_threshold"] = float(v_threshold)
        if "v_reset" in sig.parameters:
            kw["v_reset"] = 0.0
        if "detach_reset" in sig.parameters:
            kw["detach_reset"] = bool(detach_reset)
        # surrogate arg name differs across versions
        if surr is not None:
            for name in ("surrogate_function", "surrogate_func", "surrogate"):
                if name in sig.parameters:
                    kw[name] = surr
                    break
        if "step_mode" in sig.parameters:
            kw["step_mode"] = "m"
            return cls(**kw)
        # no step_mode -> single-step, wrap it
        node = cls(**kw)
        return _MultiStepWrapper(node)

    # Older versions may have MultiStepLIFNode
    if hasattr(neuron, "MultiStepLIFNode"):
        cls = neuron.MultiStepLIFNode
        sig = inspect.signature(cls)
        kw = {}
        if "tau" in sig.parameters:
            kw["tau"] = float(tau)
        if "v_threshold" in sig.parameters:
            kw["v_threshold"] = float(v_threshold)
        if "v_reset" in sig.parameters:
            kw["v_reset"] = 0.0
        if "detach_reset" in sig.parameters:
            kw["detach_reset"] = bool(detach_reset)
        if surr is not None:
            for name in ("surrogate_function", "surrogate_func", "surrogate"):
                if name in sig.parameters:
                    kw[name] = surr
                    break
        return cls(**kw)

    raise RuntimeError("Unsupported spikingjelly version: cannot find LIFNode / MultiStepLIFNode.")


# ----------------------------
# C' Router: PeriodDetector
# ----------------------------
class PeriodDetector(nn.Module):
    """
    FFT only for:
      - dominant periods (integer in samples)
      - weights (softmax over amplitudes)
      - reliability gate r -> enabled

    Input x: [B, T, C]
    Output:
      periods: List[int] (unique)
      weights: Tensor [P] softmaxed
      r: Tensor scalar (reliability)
      enabled: bool  (r >= r_min and P>0)
    """
    def __init__(
        self,
        top_k: int = 3,
        window: Optional[int] = 256,
        max_period: Optional[int] = None,
        temperature: float = 1.0,
        r_min: float = 0.08,
        remove_mean: bool = True,
    ):
        super().__init__()
        self.top_k = int(top_k)
        self.window = None if window is None else int(window)
        self.max_period = None if max_period is None else int(max_period)
        self.temperature = float(temperature)
        self.r_min = float(r_min)
        self.remove_mean = bool(remove_mean)

    @torch.no_grad()
    def forward(self, x_btc: torch.Tensor) -> Tuple[List[int], torch.Tensor, torch.Tensor, bool]:
        assert x_btc.dim() == 3, f"expect [B,T,C], got {x_btc.shape}"
        B, T, C = x_btc.shape

        if self.top_k <= 0 or T <= 2:
            return [], torch.empty(0, device=x_btc.device), torch.tensor(0.0, device=x_btc.device), False

        # optional tail window
        if self.window is not None and self.window < T:
            x = x_btc[:, -self.window:, :]
            W = self.window
        else:
            x = x_btc
            W = T

        x = x.float()
        if self.remove_mean:
            x = x - x.mean(dim=1, keepdim=True)

        fft = torch.fft.rfft(x, dim=1)          # [B, Wf, C]
        amp = fft.abs().mean(dim=(0, 2))        # [Wf]
        if amp.numel() <= 2:
            return [], torch.empty(0, device=x_btc.device), torch.tensor(0.0, device=x_btc.device), False

        amp[0] = 0.0  # drop DC
        k = min(self.top_k, amp.numel() - 1)
        vals, idx = torch.topk(amp, k=k, largest=True)

        total = amp.sum().clamp_min(1e-8)
        r = vals.sum() / total  # reliability

        # freq index -> period in samples
        periods = (W / idx.float()).round().long().clamp(min=1, max=W - 1)
        if self.max_period is not None:
            periods = periods.clamp(max=self.max_period)

        # merge duplicates
        uniq_p, inv = torch.unique(periods, sorted=False, return_inverse=True)
        w_uniq = torch.zeros_like(uniq_p, dtype=vals.dtype)
        w_uniq.scatter_add_(0, inv, vals)

        temp = max(self.temperature, 1e-6)
        weights = torch.softmax(w_uniq / temp, dim=0)

        enabled = (r.item() >= self.r_min) and (uniq_p.numel() > 0)
        return uniq_p.tolist(), weights.to(x_btc.device), r.to(x_btc.device), bool(enabled)


# ----------------------------
# SRP Encoder
# ----------------------------
def shift_right_btd(x: torch.Tensor, p: int) -> torch.Tensor:
    """Shift right by p along time dim, zero pad on the left. x: [B,T,D]."""
    if p <= 0:
        return x
    B, T, D = x.shape
    p = min(int(p), T - 1)
    x_pad = F.pad(x, (0, 0, p, 0))  # pad time-left
    return x_pad[:, :T, :]


class SRPConvSpikeEncoder(nn.Module):
    """
    x[B,T,C] -> I(t)=Conv+BN -> PeriodDetector on I or X -> SRP residual on I -> LIF -> spk[B,Ts,T,D]
    """
    def __init__(
        self,
        in_ch: int,
        d_model: int,
        Ts: int,
        conv_kernel: int = 3,
        # LIF
        lif_tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        # SRP
        use_srp: bool = True,
        alpha: float = 1.0,
        # Router
        top_k: int = 3,
        window: Optional[int] = 256,
        max_period: Optional[int] = None,
        temperature: float = 1.0,
        r_min: float = 0.08,
        router_source: str = "I",  # "I" or "X"
        remove_mean: bool = True,
    ):
        super().__init__()
        self.Ts = int(Ts)
        self.d_model = int(d_model)
        self.use_srp = bool(use_srp)
        self.alpha = float(alpha)
        self.router_source = (router_source or "I").upper()

        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2)
        self.bn = nn.BatchNorm1d(d_model)

        self.detector = PeriodDetector(
            top_k=top_k,
            window=window,
            max_period=max_period,
            temperature=temperature,
            r_min=r_min,
            remove_mean=remove_mean,
        )

        self.lif = build_lif(
            tau=lif_tau,
            v_threshold=v_threshold,
            detach_reset=detach_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

    def forward(self, x_btc: torch.Tensor, return_router: bool = True):
        """
        x_btc: [B,T,C]
        return spk: [B,Ts,T,D]
        optionally router: (periods, weights, r, enabled)
        """
        assert x_btc.dim() == 3, f"expect [B,T,C], got {x_btc.shape}"
        B, T, C = x_btc.shape

        I = self.bn(self.conv(x_btc.transpose(1, 2))).transpose(1, 2)  # [B,T,D]

        # Router on I(t) (default) or raw X (ablation)
        if self.router_source == "X":
            periods, weights, r, enabled = self.detector(x_btc)
        else:
            periods, weights, r, enabled = self.detector(I)

        # SRP residual on I(t)
        if self.use_srp and enabled and len(periods) > 0:
            R = torch.zeros_like(I)
            for wi, p in zip(weights, periods):
                delta = I - shift_right_btd(I, p)
                R = R + wi.view(1, 1, 1) * delta
            I = I + self.alpha * R

        # LIF eventification (multi-step)
        reset_net(self.lif)
        cur_ms = I.unsqueeze(0).repeat(self.Ts, 1, 1, 1)  # [Ts,B,T,D]
        spk_ms = self.lif(cur_ms)                          # [Ts,B,T,D]
        spk = spk_ms.permute(1, 0, 2, 3).contiguous()      # [B,Ts,T,D]

        if return_router:
            return spk, (periods, weights, r, enabled)
        return spk


# ----------------------------
# SPF Folding Block
# ----------------------------
def fold_time(z_bsttd: torch.Tensor, p: int) -> Tuple[torch.Tensor, int]:
    """
    z: [B,Ts,T,D] -> z2d: [Ts,B,D,N,p]
    return (z2d, pad_len)
    """
    B, Ts, T, D = z_bsttd.shape
    p = max(1, min(int(p), T))
    N = math.ceil(T / p)
    total = N * p
    pad_len = total - T

    x = z_bsttd.permute(1, 0, 3, 2).contiguous()  # [Ts,B,D,T]
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))
    x = x.view(Ts, B, D, N, p)
    return x, pad_len


def unfold_time(z2d_tbdnp: torch.Tensor, T: int, pad_len: int) -> torch.Tensor:
    """
    z2d: [Ts,B,D,N,p] -> [B,Ts,T,D]
    """
    Ts, B, D, N, p = z2d_tbdnp.shape
    x = z2d_tbdnp.reshape(Ts, B, D, N * p)
    if pad_len > 0:
        x = x[..., :-pad_len]
    x = x[..., :T]
    return x.permute(1, 0, 3, 2).contiguous()


def sew_residual(x: torch.Tensor, fx: torch.Tensor, mode: str = "ADD") -> torch.Tensor:
    m = (mode or "ADD").upper()
    if m == "ADD":
        return x + fx
    if m == "AND":
        return x * fx
    if m == "OR":
        return x + fx - x * fx
    if m == "XOR":
        return x + fx - 2 * x * fx
    return x + fx


class SpikingConv2dBlock(nn.Module):
    """
    x: [Ts,B,D,H,W] -> [Ts,B,D,H,W]
    IMPORTANT: H/W vary with period p, so we reset LIF state every forward call.
    """
    def __init__(
        self,
        ch: int,
        lif_tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.lif = build_lif(
            tau=lif_tau,
            v_threshold=v_threshold,
            detach_reset=detach_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

    def forward(self, x_tbdhw: torch.Tensor) -> torch.Tensor:
        reset_net(self.lif)

        Ts, B, D, H, W = x_tbdhw.shape
        y = x_tbdhw.reshape(Ts * B, D, H, W)
        y = self.bn(self.conv(y))
        y = y.view(Ts, B, D, H, W)
        y = self.lif(y)
        return y


class ANNRateConv2dBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU()

    def forward(self, x_bdhw: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x_bdhw)))


class SPFBlock(nn.Module):
    """
    z: [B,Ts,T,D]
    periods: List[int], weights: [P], enabled: bool
    """
    def __init__(
        self,
        ch: int,
        Ts: int,
        mode: str = "spiking",     # "spiking" or "ann_rate" (E4)
        sew_mode: str = "ADD",
        lif_tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.ch = int(ch)
        self.Ts = int(Ts)
        self.mode = (mode or "spiking").lower()
        self.sew_mode = sew_mode

        self.spiking2d = SpikingConv2dBlock(
            ch,
            lif_tau=lif_tau,
            v_threshold=v_threshold,
            detach_reset=detach_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )
        self.ann2d = ANNRateConv2dBlock(ch)
        self.reencode_lif = build_lif(
            tau=lif_tau,
            v_threshold=v_threshold,
            detach_reset=detach_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

    def forward(self, z: torch.Tensor, periods: List[int], weights: torch.Tensor, enabled: bool) -> torch.Tensor:
        if (not enabled) or (len(periods) == 0):
            return z

        B, Ts, T, D = z.shape
        assert Ts == self.Ts, f"Ts mismatch: {Ts} vs {self.Ts}"
        assert D == self.ch, f"channel mismatch: {D} vs {self.ch}"

        outs = []
        for p in periods:
            p = max(1, min(int(p), T))

            if self.mode == "ann_rate":
                # IMPORTANT: re-encode LIF state should not leak across periods
                reset_net(self.reencode_lif)

                rate = z.mean(dim=1)                      # [B,T,D]
                N = math.ceil(T / p)
                total = N * p
                pad_len = total - T

                x = rate.permute(0, 2, 1).contiguous()    # [B,D,T]
                if pad_len > 0:
                    x = F.pad(x, (0, pad_len))
                x = x.view(B, D, N, p)                    # [B,D,N,p]
                y = self.ann2d(x)                         # [B,D,N,p]
                y = y.view(B, D, N * p)
                if pad_len > 0:
                    y = y[:, :, :-pad_len]
                y = y[:, :, :T]
                y_btd = y.permute(0, 2, 1).contiguous()   # [B,T,D]

                cur_ms = y_btd.unsqueeze(0).repeat(Ts, 1, 1, 1)  # [Ts,B,T,D]
                spk_ms = self.reencode_lif(cur_ms)              # [Ts,B,T,D]
                out = spk_ms.permute(1, 0, 2, 3).contiguous()   # [B,Ts,T,D]
            else:
                z2d, pad_len = fold_time(z, p)                  # [Ts,B,D,N,p]
                out2d = self.spiking2d(z2d)                     # [Ts,B,D,N,p]
                out = unfold_time(out2d, T=T, pad_len=pad_len)  # [B,Ts,T,D]

            outs.append(out)

        stack = torch.stack(outs, dim=0)                         # [P,B,Ts,T,D]
        w = weights.view(-1, 1, 1, 1, 1).to(stack.device)        # [P,1,1,1,1]
        mix = (w * stack).sum(dim=0)                             # [B,Ts,T,D]
        return sew_residual(z, mix, mode=self.sew_mode)
