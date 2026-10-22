"""
Spiking TCN + SRP + SPF backbone with a causal / non-causal switch, used by run_protocols.py
for the operational-protocol comparison (A1 causal current-step, A2 one-step prediction,
A3 non-causal reconstruction) under output-error scoring.

  causal=True   left-padded (causal) convolutions: y[t] depends on x[0:t] only (A1, A2)
  causal=False  symmetric padding: y[t] depends on x[t-r:t+r] (A3)

The SRP encoder, the SPF block and the decode head are the same as in backbone.py; this
module does not expose the internal spike states.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .srp_spf import SRPConvSpikeEncoder, SPFBlock, sew_residual, build_lif, reset_net


class Chomp1d(nn.Module):
    """Remove the last `chomp_size` elements (for causal mode only)."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class SpikingTemporalBlock1D(nn.Module):
    """
    Multi-step spiking temporal block with causal/non-causal switch.

    causal=True:  left padding + chomp, y[t] depends on x[0:t] only (prediction / causal reconstruction)
    causal=False: symmetric padding, y[t] depends on x[t-r:t+r] (non-causal reconstruction)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Ts: int,
        kernel_size: int = 16,
        dilation: int = 1,
        stride: int = 1,
        dropout: float = 0.0,
        sew_mode: str = "ADD",
        causal: bool = True,
        # LIF params
        lif_tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.Ts = int(Ts)
        self.sew_mode = sew_mode
        self.causal = causal

        if causal:
            # causal: full left padding + chomp
            pad = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=pad, dilation=dilation)
            self.chomp1 = Chomp1d(pad)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   stride=1, padding=pad, dilation=dilation)
            self.chomp2 = Chomp1d(pad)
        else:
            # non-causal: symmetric padding, no chomp
            pad = (kernel_size - 1) * dilation // 2
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=pad, dilation=dilation)
            self.chomp1 = None
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   stride=1, padding=pad, dilation=dilation)
            self.chomp2 = None

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.lif1 = build_lif(tau=lif_tau, v_threshold=v_threshold,
                              detach_reset=detach_reset,
                              surrogate_name=surrogate_name,
                              surrogate_alpha=surrogate_alpha)
        self.drop1 = nn.Dropout(dropout)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.lif2 = build_lif(tau=lif_tau, v_threshold=v_threshold,
                              detach_reset=detach_reset,
                              surrogate_name=surrogate_name,
                              surrogate_alpha=surrogate_alpha)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def _conv_bn(self, x_tbdt: torch.Tensor, conv: nn.Conv1d,
                 bn: nn.BatchNorm1d, chomp: Optional[Chomp1d]) -> torch.Tensor:
        Ts, B, C, T = x_tbdt.shape
        y = x_tbdt.reshape(Ts * B, C, T)
        y = conv(y)
        if chomp is not None:
            y = chomp(y)
        # keep the output length equal to the input length
        if y.shape[-1] > T:
            y = y[..., :T]
        elif y.shape[-1] < T:
            y = torch.nn.functional.pad(y, (0, T - y.shape[-1]))
        y = bn(y)
        y = y.view(Ts, B, y.shape[1], T)
        return y

    def _downsample(self, x_tbdt: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return x_tbdt
        Ts, B, C, T = x_tbdt.shape
        y = x_tbdt.reshape(Ts * B, C, T)
        y = self.downsample(y)
        y = y.view(Ts, B, y.shape[1], T)
        return y

    def forward(self, z_bsttc: torch.Tensor) -> torch.Tensor:
        assert z_bsttc.dim() == 4, f"expect [B,Ts,T,C], got {z_bsttc.shape}"
        B, Ts, T, C = z_bsttc.shape
        assert Ts == self.Ts, f"Ts mismatch: {Ts} vs {self.Ts}"

        x = z_bsttc.permute(1, 0, 3, 2).contiguous()  # [Ts,B,C,T]

        # conv1 -> lif1
        y = self._conv_bn(x, self.conv1, self.bn1, self.chomp1)
        reset_net(self.lif1)
        y = self.lif1(y)
        y = self.drop1(y)

        # conv2 -> lif2
        y = self._conv_bn(y, self.conv2, self.bn2, self.chomp2)
        reset_net(self.lif2)
        y = self.lif2(y)
        y = self.drop2(y)

        res = self._downsample(x)
        out = sew_residual(res, y, mode=self.sew_mode)
        return out.permute(1, 0, 3, 2).contiguous()  # [B,Ts,T,Cout]


class SpikeTCN_SRPSPF_AD(nn.Module):
    """
    Spiking TCN + SRP + SPF for Anomaly Detection.

    Same structure as the forecasting backbone, with one addition:
      causal: bool  (True=prediction-based, False=reconstruction-based)

    Input:  x  [B, T, C]
    Output: y_seq [B, C, T], y_last [B, C]
    """
    def __init__(
        self,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,

        # AD mode
        causal: bool = True,

        # backbone (TCN)
        kernel_size: int = 16,
        num_levels: int = 3,
        channel: int = 16,
        dilation: int = 2,
        dropout: float = 0.0,
        stride: int = 1,

        # simulation steps
        num_steps: int = 4,

        # encoder dim
        hidden_size: int = 64,

        # LIF params
        lif_tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,

        # SRP encoder params
        srp_conv_kernel: int = 3,
        use_srp: bool = True,
        alpha: float = 1.0,

        # Router params
        top_k: int = 3,
        window: Optional[int] = 256,
        max_period: Optional[int] = None,
        temperature: float = 1.0,
        r_min: float = 0.08,
        router_source: str = "I",
        remove_mean: bool = True,

        # SPF params
        use_spf: bool = True,
        spf_at: int = 0,
        spf_mode: str = "spiking",
        spf_sew_mode: str = "ADD",

        # compatibility
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

        self.num_steps = int(num_steps)
        self.hidden_size = int(hidden_size)
        self.use_spf = bool(use_spf)
        self.spf_at = int(spf_at)
        self.causal = bool(causal)

        # --- SRP Encoder (unchanged)
        self.encoder = SRPConvSpikeEncoder(
            in_ch=self.__input_size,
            d_model=self.hidden_size,
            Ts=self.num_steps,
            conv_kernel=srp_conv_kernel,
            lif_tau=lif_tau,
            v_threshold=v_threshold,
            detach_reset=detach_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
            use_srp=use_srp,
            alpha=alpha,
            top_k=top_k,
            window=window,
            max_period=max_period,
            temperature=temperature,
            r_min=r_min,
            router_source=router_source,
            remove_mean=remove_mean,
        )

        # --- Spiking TCN backbone (with causal switch)
        blocks = []
        for i in range(int(num_levels)):
            d = int(dilation) ** i
            in_ch = self.hidden_size if i == 0 else int(channel)
            out_ch = int(channel)
            blocks.append(
                SpikingTemporalBlock1D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    Ts=self.num_steps,
                    kernel_size=kernel_size,
                    dilation=d,
                    stride=stride,
                    dropout=dropout,
                    sew_mode="ADD",
                    causal=self.causal,
                    lif_tau=lif_tau,
                    v_threshold=v_threshold,
                    detach_reset=detach_reset,
                    surrogate_name=surrogate_name,
                    surrogate_alpha=surrogate_alpha,
                )
            )
        self.tcn = nn.ModuleList(blocks)

        # --- SPF folding block (unchanged)
        self.spf = SPFBlock(
            ch=int(channel),
            Ts=self.num_steps,
            mode=spf_mode,
            sew_mode=spf_sew_mode,
            lif_tau=lif_tau,
            v_threshold=v_threshold,
            detach_reset=detach_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # --- Decode head (unchanged)
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

    def forward(self, x_btc: torch.Tensor):
        """
        x_btc: [B,T,C]
        returns: y_seq [B,C,T], y_last [B,C]
        """
        spk, router = self.encoder(x_btc, return_router=True)
        periods, weights, r, enabled = router
        self.last_router = (periods, weights.detach(), float(r.detach().cpu().item()), bool(enabled))

        z = spk
        for i, blk in enumerate(self.tcn):
            z = blk(z)
            if self.use_spf and (i == self.spf_at):
                z = self.spf(z, periods, weights, enabled)

        rate = z.mean(dim=1)
        y = self.head(rate.permute(0, 2, 1))
        return y, y[:, :, -1]
