"""SPIRE: phase-conditioned spiking dynamics for time-series anomaly detection.

Modules:
  srp_spf             period router, SRP residual encoder, SPF block, LIF builders (SpikingJelly)
  backbone            the SPIRE backbone with access to the internal spike states (run_spire.py)
  protocol_backbone   the same backbone with a causal / non-causal switch (run_protocols.py)
  backbone_ann        the matched ReLU control
  controls            the stateless binary units of the neuron-isolation experiment
"""

__version__ = "1.0.0"
