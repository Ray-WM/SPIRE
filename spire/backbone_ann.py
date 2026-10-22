"""Matched ReLU control of SPIRE: the same SRP / SPF / TCN topology with continuous (ReLU) units.
The forward pass exposes the activations from which run_spire.py reads the activation-template statistics.
"""
from typing import Optional, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



# ---- shared helpers (ANN version) ----
def shift_right_btd(x: torch.Tensor, p: int) -> torch.Tensor:
    if p <= 0:
        return x
    B, T, D = x.shape
    p = min(int(p), T - 1)
    x_pad = F.pad(x, (0, 0, p, 0))
    return x_pad[:, :T, :]


class PeriodDetector(nn.Module):
    """Same PeriodDetector as SNN version, but no spiking dependency."""
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
    def forward(self, x_btc: torch.Tensor):
        assert x_btc.dim() == 3, f"expect [B,T,C], got {x_btc.shape}"
        B, T, C = x_btc.shape
        if self.top_k <= 0 or T <= 2:
            return [], torch.empty(0, device=x_btc.device), torch.tensor(0.0, device=x_btc.device), False

        if self.window is not None and self.window < T:
            x = x_btc[:, -self.window:, :]
            W = self.window
        else:
            x = x_btc
            W = T

        x = x.float()
        if self.remove_mean:
            x = x - x.mean(dim=1, keepdim=True)
        fft = torch.fft.rfft(x, dim=1)
        amp = fft.abs().mean(dim=(0, 2))
        if amp.numel() <= 2:
            return [], torch.empty(0, device=x_btc.device), torch.tensor(0.0, device=x_btc.device), False
        amp[0] = 0.0
        k = min(self.top_k, amp.numel() - 1)
        vals, idx = torch.topk(amp, k=k, largest=True)
        total = amp.sum().clamp_min(1e-8)
        r = vals.sum() / total
        periods = (W / idx.float()).round().long().clamp(min=1, max=W - 1)
        if self.max_period is not None:
            periods = periods.clamp(max=self.max_period)

        uniq_p, inv = torch.unique(periods, sorted=False, return_inverse=True)
        w_uniq = torch.zeros_like(uniq_p, dtype=vals.dtype)
        w_uniq.scatter_add_(0, inv, vals)
        weights = torch.softmax(w_uniq / max(self.temperature, 1e-6), dim=0)
        enabled = (r.item() >= self.r_min) and (uniq_p.numel() > 0)
        return uniq_p.tolist(), weights.to(x_btc.device), r.to(x_btc.device), bool(enabled)


def fold_time_btd(z_btd: torch.Tensor, p: int) -> Tuple[torch.Tensor, int]:
    """z: [B,T,D] -> [B,D,N,p]"""
    B, T, D = z_btd.shape
    p = max(1, min(int(p), T))
    N = math.ceil(T / p)
    total = N * p
    pad_len = total - T
    x = z_btd.permute(0, 2, 1).contiguous()  # [B,D,T]
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))
    x = x.view(B, D, N, p)
    return x, pad_len


def unfold_time_btd(z_bdnP: torch.Tensor, T: int, pad_len: int) -> torch.Tensor:
    """[B,D,N,p] -> [B,T,D]"""
    B, D, N, p = z_bdnP.shape
    x = z_bdnP.view(B, D, N * p)
    if pad_len > 0:
        x = x[:, :, :-pad_len]
    x = x[:, :, :T]
    return x.permute(0, 2, 1).contiguous()


class SPFBlockANN(nn.Module):
    """ANN SPF: z[B,T,D] -> fold -> 2D conv -> unfold -> weighted sum -> residual add."""
    def __init__(self, ch: int, sew_mode: str = "ADD"):
        super().__init__()
        self.ch = int(ch)
        self.sew_mode = sew_mode
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU()

    def sew(self, x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        m = (self.sew_mode or "ADD").upper()
        if m == "ADD":
            return x + fx
        if m == "AND":
            return x * fx
        if m == "OR":
            return x + fx - x * fx
        if m == "XOR":
            return x + fx - 2 * x * fx
        return x + fx

    def forward(self, z_btd: torch.Tensor, periods: List[int], weights: torch.Tensor, enabled: bool) -> torch.Tensor:
        if (not enabled) or (len(periods) == 0):
            return z_btd
        B, T, D = z_btd.shape
        assert D == self.ch, f"channel mismatch {D} vs {self.ch}"
        outs = []
        for p in periods:
            x2d, pad_len = fold_time_btd(z_btd, p)      # [B,D,N,p]
            y2d = self.act(self.bn(self.conv(x2d)))     # [B,D,N,p]
            y = unfold_time_btd(y2d, T=T, pad_len=pad_len)  # [B,T,D]
            outs.append(y)
        stack = torch.stack(outs, dim=0)                # [P,B,T,D]
        w = weights.view(-1, 1, 1, 1).to(stack.device)
        mix = (w * stack).sum(dim=0)                    # [B,T,D]
        return self.sew(z_btd, mix)


class TemporalBlock1D(nn.Module):
    """Standard TCN block (ANN). Input/Output: [B,T,C]"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float, sew_mode: str = "ADD"):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = pad
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = pad
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.sew_mode = sew_mode

    def _chomp(self, x: torch.Tensor, chomp: int) -> torch.Tensor:
        if chomp <= 0:
            return x
        return x[:, :, :-chomp].contiguous()

    def sew(self, x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        m = (self.sew_mode or "ADD").upper()
        if m == "ADD":
            return x + fx
        if m == "AND":
            return x * fx
        if m == "OR":
            return x + fx - x * fx
        if m == "XOR":
            return x + fx - 2 * x * fx
        return x + fx

    def forward(self, z_btc: torch.Tensor) -> torch.Tensor:
        B, T, C = z_btc.shape
        x = z_btc.permute(0, 2, 1).contiguous()  # [B,C,T]
        y = self.drop1(self.act1(self.bn1(self._chomp(self.conv1(x), self.chomp1))))
        y = self.drop2(self.act2(self.bn2(self._chomp(self.conv2(y), self.chomp2))))
        res = x if self.down is None else self.down(x)
        out = self.sew(res, y)
        return out.permute(0, 2, 1).contiguous()  # [B,T,C]


class ANNTCN_SRPSPF(nn.Module):
    """
    ANN counterpart of SRP+SPF:
      - router on I(t) (conv+bn)
      - SRP residual on I(t)
      - standard TCN backbone
      - SPF folding on continuous features (no spikes)

    Input:  [B,T,C]
    Output: [B,C,T], [B,C]
    """
    def __init__(
        self,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        # backbone
        kernel_size: int = 16,
        num_levels: int = 3,
        channel: int = 16,
        dilation: int = 2,
        dropout: float = 0.0,
        # encoder dim
        hidden_size: int = 64,
        # SRP
        srp_conv_kernel: int = 3,
        use_srp: bool = True,
        alpha: float = 1.0,
        # router
        top_k: int = 3,
        window: Optional[int] = 256,
        max_period: Optional[int] = None,
        temperature: float = 1.0,
        r_min: float = 0.08,
        router_source: str = "I",
        remove_mean: bool = True,
        # SPF
        use_spf: bool = True,
        spf_at: int = 0,
        spf_sew_mode: str = "ADD",
        # ignored compat
        encoder_type: str = "conv",
        pe_type: str = "none",
        pe_mode: str = "concat",
        num_pe_neuron: int = 0,
        neuron_pe_scale: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.__input_size = int(input_size)
        self.__output_size = int(input_size)
        self.__max_length = int(max_length) if max_length is not None else None

        self.use_srp = bool(use_srp)
        self.alpha = float(alpha)
        self.router_source = (router_source or "I").upper()

        self.conv = nn.Conv1d(self.__input_size, int(hidden_size), kernel_size=srp_conv_kernel, padding=srp_conv_kernel // 2)
        self.bn = nn.BatchNorm1d(int(hidden_size))
        self.detector = PeriodDetector(
            top_k=top_k,
            window=window,
            max_period=max_period,
            temperature=temperature,
            r_min=r_min,
            remove_mean=remove_mean,
        )

        blocks = []
        for i in range(int(num_levels)):
            d = int(dilation) ** i
            in_ch = int(hidden_size) if i == 0 else int(channel)
            out_ch = int(channel)
            blocks.append(TemporalBlock1D(in_ch, out_ch, kernel_size=kernel_size, dilation=d, dropout=dropout))
        self.tcn = nn.ModuleList(blocks)

        self.use_spf = bool(use_spf)
        self.spf_at = int(spf_at)
        self.spf = SPFBlockANN(int(channel), sew_mode=spf_sew_mode)

        self.head = nn.Conv1d(int(channel), self.__input_size, kernel_size=1)

        self.last_router = None

    @property
    def input_size(self):
        return self.__input_size

    @property
    def output_size(self):
        return self.__output_size

    @property
    def max_length(self):
        return self.__max_length

    def forward(self, x_btc: torch.Tensor, return_internal: bool = False):
        B, T, C = x_btc.shape
        I = self.bn(self.conv(x_btc.transpose(1, 2))).transpose(1, 2)  # [B,T,H]

        if self.router_source == "X":
            periods, weights, r, enabled = self.detector(x_btc)
        else:
            periods, weights, r, enabled = self.detector(I)

        self.last_router = (periods, weights.detach(), float(r.detach().cpu().item()), bool(enabled))

        R = torch.zeros_like(I)
        if self.use_srp and enabled and len(periods) > 0:
            for wi, p in zip(weights, periods):
                delta = I - shift_right_btd(I, p)
                R = R + wi.view(1, 1, 1) * delta
            I = I + self.alpha * R

        z = I
        for i, blk in enumerate(self.tcn):
            z = blk(z)  # [B,T,channel]
            if self.use_spf and (i == self.spf_at):
                z = self.spf(z, periods, weights, enabled)

        y = self.head(z.permute(0, 2, 1))  # [B,C,T]
        if return_internal:
            # Matched internal states for the ReLU activation-template control:
            # enc_act = ReLU of the (residual-injected) encoder current,
            # back_act = continuous backbone features.
            internals = {
                "srp_residual": R,                      # [B,T,H]
                "enc_act": torch.relu(I),               # [B,T,H]
                "back_act": z,                          # [B,T,channel]
                "router": self.last_router,
            }
            return y, y[:, :, -1], internals
        return y, y[:, :, -1]
