"""
Spiking TCN + SRP + SPF backbone of SPIRE, with access to the internal spike states.

Differences from protocol_backbone.py:
  - the SRP encoder exposes the per-step cross-period residual R(t)
  - forward(..., return_internal=True) returns the internal spike states
  - used by run_spire.py for the internal-state (template) readout

internals dictionary:
  'srp_residual':  [B, T, D]         SRP cross-period residual (the phase-aligned stimulus)
  'enc_spikes':    [B, Ts, T, D]     encoder LIF spikes
  'enc_rate':      [B, T, D]         encoder firing rate per step
  'back_spikes':   [B, Ts, T, D_ch]  spikes of the last backbone layer (after SPF)
  'back_rate':     [B, T, D_ch]      backbone firing rate per step (input of the decode head)
  'router':        (periods, weights, r, enabled)
"""

from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .srp_spf import (
    PeriodDetector, SPFBlock, sew_residual, build_lif, reset_net,
    shift_right_btd,
)


# ============================================================
# Modified SRP Encoder: saves R(t)
# ============================================================

class SRPConvSpikeEncoder_AStar(nn.Module):
    """
    Same as SRPConvSpikeEncoder, but stores:
      - self.last_srp_residual: [B, T, D]  (detached)
      - self.last_I_enhanced:   [B, T, D]  (detached, I after SRP injection)
    """
    def __init__(
        self,
        in_ch: int, d_model: int, Ts: int,
        conv_kernel: int = 3,
        lif_tau: float = 2.0, v_threshold: float = 1.0,
        detach_reset: bool = True,
        surrogate_name: str = "atan", surrogate_alpha: float = 2.0,
        use_srp: bool = True, alpha: float = 1.0,
        top_k: int = 3, window: Optional[int] = 256,
        max_period: Optional[int] = None, temperature: float = 1.0,
        r_min: float = 0.08, router_source: str = "I",
        remove_mean: bool = True,
    ):
        super().__init__()
        self.Ts = int(Ts)
        self.d_model = int(d_model)
        self.use_srp = bool(use_srp)
        self.alpha = float(alpha)
        self.router_source = (router_source or "I").upper()

        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=conv_kernel,
                              padding=conv_kernel // 2)
        self.bn = nn.BatchNorm1d(d_model)
        self.detector = PeriodDetector(
            top_k=top_k, window=window, max_period=max_period,
            temperature=temperature, r_min=r_min, remove_mean=remove_mean,
        )
        self.lif = build_lif(
            tau=lif_tau, v_threshold=v_threshold,
            detach_reset=detach_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # Storage of the internal signals
        self.last_srp_residual = None   # [B, T, D]
        self.last_I_enhanced = None     # [B, T, D]

    def forward(self, x_btc: torch.Tensor, return_router: bool = True):
        B, T, C = x_btc.shape

        I = self.bn(self.conv(x_btc.transpose(1, 2))).transpose(1, 2)  # [B,T,D]

        # Router
        if self.router_source == "X":
            periods, weights, r, enabled = self.detector(x_btc)
        else:
            periods, weights, r, enabled = self.detector(I)

        # SRP residual (kept for the internal-state readout)
        if self.use_srp and enabled and len(periods) > 0:
            R = torch.zeros_like(I)
            for wi, p in zip(weights, periods):
                delta = I - shift_right_btd(I, p)
                R = R + wi.view(1, 1, 1) * delta
            self.last_srp_residual = R.detach()
            I = I + self.alpha * R
        else:
            self.last_srp_residual = torch.zeros_like(I).detach()

        self.last_I_enhanced = I.detach()

        # LIF
        reset_net(self.lif)
        cur_ms = I.unsqueeze(0).repeat(self.Ts, 1, 1, 1)  # [Ts,B,T,D]
        spk_ms = self.lif(cur_ms)                          # [Ts,B,T,D]
        spk = spk_ms.permute(1, 0, 2, 3).contiguous()      # [B,Ts,T,D]

        if return_router:
            return spk, (periods, weights, r, enabled)
        return spk


# ============================================================
# Chomp1d (same as ad.py)
# ============================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


# ============================================================
# SpikingTemporalBlock1D (same as ad.py, causal switch)
# ============================================================

class SpikingTemporalBlock1D(nn.Module):
    def __init__(
        self, in_channels, out_channels, Ts,
        kernel_size=16, dilation=1, stride=1, dropout=0.0,
        sew_mode="ADD", causal=True,
        lif_tau=2.0, v_threshold=1.0, detach_reset=True,
        surrogate_name="atan", surrogate_alpha=2.0,
    ):
        super().__init__()
        self.Ts = int(Ts)
        self.sew_mode = sew_mode
        self.causal = causal

        if causal:
            pad = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=pad, dilation=dilation)
            self.chomp1 = Chomp1d(pad)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   stride=1, padding=pad, dilation=dilation)
            self.chomp2 = Chomp1d(pad)
        else:
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

    def _conv_bn(self, x_tbdt, conv, bn, chomp):
        Ts, B, C, T = x_tbdt.shape
        y = x_tbdt.reshape(Ts * B, C, T)
        y = conv(y)
        if chomp is not None:
            y = chomp(y)
        if y.shape[-1] != T:
            y = y[..., :T]
        y = bn(y)
        return y.view(Ts, B, y.shape[1], T)

    def _downsample(self, x_tbdt):
        if self.downsample is None:
            return x_tbdt
        Ts, B, C, T = x_tbdt.shape
        y = x_tbdt.reshape(Ts * B, C, T)
        return self.downsample(y).view(Ts, B, -1, T)

    def forward(self, z_bsttc):
        B, Ts, T, C = z_bsttc.shape
        x = z_bsttc.permute(1, 0, 3, 2).contiguous()  # [Ts,B,C,T]

        y = self._conv_bn(x, self.conv1, self.bn1, self.chomp1)
        reset_net(self.lif1)
        y = self.lif1(y)
        y = self.drop1(y)

        y = self._conv_bn(y, self.conv2, self.bn2, self.chomp2)
        reset_net(self.lif2)
        y = self.lif2(y)
        y = self.drop2(y)

        res = self._downsample(x)
        out = sew_residual(res, y, mode=self.sew_mode)
        return out.permute(1, 0, 3, 2).contiguous()  # [B,Ts,T,Cout]


# ============================================================
# SPIRE network
# ============================================================

class SpikeTCN_SRPSPF_AD_Star(nn.Module):
    """
    SPIRE backbone: causal spiking TCN with SRP/SPF and access to the internal states.

    forward(x, return_internal=False):
      return_internal=False: (y_seq, y_last)
      return_internal=True:  (y_seq, y_last, internals_dict)
    """
    def __init__(
        self,
        input_size=None, max_length=None,
        causal=True,
        kernel_size=16, num_levels=3, channel=16, dilation=2,
        dropout=0.0, stride=1,
        num_steps=4, hidden_size=64,
        lif_tau=2.0, v_threshold=1.0, detach_reset=True,
        surrogate_name="atan", surrogate_alpha=2.0,
        srp_conv_kernel=3, use_srp=True, alpha=1.0,
        top_k=3, window=256, max_period=None, temperature=1.0,
        r_min=0.08, router_source="I", remove_mean=True,
        use_spf=True, spf_at=0, spf_mode="spiking", spf_sew_mode="ADD",
        # compat
        encoder_type="conv", pe_type="none", pe_mode="concat",
        num_pe_neuron=0, neuron_pe_scale=1.0, **kwargs,
    ):
        super().__init__()
        self.__input_size = int(input_size)
        self.__output_size = int(input_size)
        self.__max_length = int(max_length) if max_length else None
        self.num_steps = int(num_steps)
        self.hidden_size = int(hidden_size)
        self.use_spf = bool(use_spf)
        self.spf_at = int(spf_at)
        self.causal = bool(causal)

        # encoder that exposes the SRP residual
        self.encoder = SRPConvSpikeEncoder_AStar(
            in_ch=self.__input_size, d_model=self.hidden_size, Ts=self.num_steps,
            conv_kernel=srp_conv_kernel,
            lif_tau=lif_tau, v_threshold=v_threshold, detach_reset=detach_reset,
            surrogate_name=surrogate_name, surrogate_alpha=surrogate_alpha,
            use_srp=use_srp, alpha=alpha,
            top_k=top_k, window=window, max_period=max_period,
            temperature=temperature, r_min=r_min,
            router_source=router_source, remove_mean=remove_mean,
        )

        # TCN backbone (same as ad.py)
        blocks = []
        for i in range(int(num_levels)):
            d = int(dilation) ** i
            in_ch = self.hidden_size if i == 0 else int(channel)
            out_ch = int(channel)
            blocks.append(SpikingTemporalBlock1D(
                in_ch, out_ch, self.num_steps,
                kernel_size=kernel_size, dilation=d, stride=stride,
                dropout=dropout, sew_mode="ADD", causal=self.causal,
                lif_tau=lif_tau, v_threshold=v_threshold,
                detach_reset=detach_reset,
                surrogate_name=surrogate_name,
                surrogate_alpha=surrogate_alpha,
            ))
        self.tcn = nn.ModuleList(blocks)

        # SPF
        self.spf = SPFBlock(
            ch=int(channel), Ts=self.num_steps,
            mode=spf_mode, sew_mode=spf_sew_mode,
            lif_tau=lif_tau, v_threshold=v_threshold,
            detach_reset=detach_reset,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # Decode head
        self.head = nn.Conv1d(int(channel), self.__input_size, kernel_size=1)

        self.last_router = None

    @property
    def input_size(self): return self.__input_size
    @property
    def output_size(self): return self.__output_size
    @property
    def max_length(self): return self.__max_length

    def forward(self, x_btc: torch.Tensor, return_internal: bool = False):
        """
        x_btc: [B, T, C]
        Returns:
          y_seq [B, C, T], y_last [B, C]
          if return_internal: also returns internals dict
        """
        # Encoder
        spk, router = self.encoder(x_btc, return_router=True)
        periods, weights, r, enabled = router
        self.last_router = (
            periods, weights.detach(),
            float(r.detach().cpu().item()), bool(enabled),
        )

        # Encoder-level signals
        enc_spikes = spk.detach()                   # [B, Ts, T, D]
        enc_rate = enc_spikes.mean(dim=1)           # [B, T, D]

        # TCN backbone
        z = spk
        for i, blk in enumerate(self.tcn):
            z = blk(z)
            if self.use_spf and (i == self.spf_at):
                z = self.spf(z, periods, weights, enabled)

        # Backbone-level signals
        back_spikes = z.detach()                     # [B, Ts, T, D_ch]
        back_rate = z.mean(dim=1)                    # [B, T, D_ch]

        # Decode
        rate = z.mean(dim=1)                         # [B, T, D_ch]
        y = self.head(rate.permute(0, 2, 1))         # [B, C, T]

        if not return_internal:
            return y, y[:, :, -1]

        internals = {
            'srp_residual': self.encoder.last_srp_residual,  # [B, T, D]
            'enc_spikes': enc_spikes,                         # [B, Ts, T, D]
            'enc_rate': enc_rate,                             # [B, T, D]
            'back_spikes': back_spikes,                       # [B, Ts, T, D_ch]
            'back_rate': back_rate.detach(),                   # [B, T, D_ch]
            'router': self.last_router,
        }
        return y, y[:, :, -1], internals
