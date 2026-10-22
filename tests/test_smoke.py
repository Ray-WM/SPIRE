"""Smoke tests on synthetic tensors (CPU, no data): the backbones build and run, the internal
states have the documented shapes, the template and the fixed scalers score, and the binary
control replaces every LIF unit."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from run_spire import (HybridScalers, SpikeTemplate, build_network_astar,  # noqa: E402
                       get_point_scores_astar, train_model)
from spire.controls import count_binary_modules, count_lif_like_modules, replace_lif_with_binary  # noqa: E402
from spire.protocol_backbone import SpikeTCN_SRPSPF_AD  # noqa: E402

CFG = dict(kernel_size=4, num_levels=2, channel=8, hidden_size=8, num_steps=3, lif_tau=2.0, dropout=0.0,
           use_srp=True, use_spf=True, alpha=0.5, top_k=2, r_min=0.0, spf_at=1)
WS = 24


def periodic_series(T=240, C=2, period=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    x = np.stack([np.sin(2 * np.pi * t / period + k) for k in range(C)], axis=1)
    return (x + 0.05 * rng.standard_normal((T, C))).astype(np.float32)


@pytest.mark.parametrize("C", [1, 3])
def test_backbone_internals(C):
    net = build_network_astar(C, WS, CFG)
    x = torch.randn(4, WS, C)
    y_seq, y_last, internals = net(x, return_internal=True)
    assert y_seq.shape == (4, C, WS) and y_last.shape == (4, C)
    assert internals["srp_residual"].shape[:2] == (4, WS)
    assert internals["enc_spikes"].shape[1] == CFG["num_steps"] and internals["enc_spikes"].shape[2] == WS
    assert internals["back_rate"].shape[:2] == (4, WS)
    feats = SpikeTemplate.extract_features(internals)
    assert feats.shape == (4, WS, SpikeTemplate.N_FEATURES)
    assert torch.isfinite(feats).all()


def test_ann_control_internals():
    net = build_network_astar(2, WS, CFG, model="ann")
    _, _, internals = net(torch.randn(3, WS, 2), return_internal=True)
    feats = SpikeTemplate.extract_features(internals)
    assert feats.shape == (3, WS, SpikeTemplate.N_FEATURES)
    assert count_lif_like_modules(net) == 0


def test_binary_control_replaces_every_lif():
    net = build_network_astar(2, WS, CFG)
    n_lif = count_lif_like_modules(net)
    assert n_lif > 0
    net = replace_lif_with_binary(net, rate_matched=True)
    assert count_lif_like_modules(net) == 0 and count_binary_modules(net) == n_lif
    y_seq, _, internals = net(torch.randn(2, WS, 2), return_internal=True)
    assert y_seq.shape == (2, 2, WS)
    assert set(internals["enc_spikes"].unique().tolist()) <= {0.0, 1.0}


def test_template_and_scores():
    data = periodic_series()
    train = data[:120]
    net = build_network_astar(2, WS, CFG)
    net = train_model(net, train, window_size=WS, batch_size=16, max_epochs=1, device="cpu")
    template = SpikeTemplate()
    template.fit(net, train, window_size=WS, batch_size=32, device="cpu")
    assert template.enabled and len(template.periods) >= 1
    for mode in ("recon", "spike", "hybrid"):
        scores = get_point_scores_astar(net, data, template, window_size=WS, batch_size=32,
                                        score_mode=mode, train_end=120, device="cpu")
        assert scores.shape == (len(data),) and np.isfinite(scores).all()
    scores, streams = get_point_scores_astar(net, data, template, window_size=WS, batch_size=32,
                                             score_mode="hybrid", train_end=120, return_streams=True, device="cpu")
    assert streams["feature_dev_stream"].shape == (len(data), SpikeTemplate.N_FEATURES)
    fused = HybridScalers().fit(streams["spike_stream"][:120], streams["recon_stream"][:120]).fuse(
        streams["spike_stream"], streams["recon_stream"])
    assert np.allclose(fused, scores, atol=1e-4)


@pytest.mark.parametrize("causal", [True, False])
def test_protocol_backbone(causal):
    net = SpikeTCN_SRPSPF_AD(input_size=2, max_length=WS, causal=causal, **CFG)
    y_seq, y_last = net(torch.randn(3, WS, 2))
    assert y_seq.shape == (3, 2, WS) and y_last.shape == (3, 2)
