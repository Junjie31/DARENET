"""DualExpertDareNet: the explicit two-expert MoE view of DARE-Net.

This module is the code-level counterpart of the *dual experts + routing*
description in the paper (Sec.3 "DARE" and Sec.4 "Reliability-Aware Policy
Layer").  It makes the two experts a first-class object and exposes the exact
routing operators used in the paper.

How it relates to the single-branch ``DareNet``:
    * Training (Sec.3, Eq. routing) — ``DareNet(dataset_name, ffn_branch=...)``
      specialises ONE expert at a time, corresponding to the two extreme cases
      of the routing variable beta (beta=1 -> Complex/spectral, beta=0 ->
      Gated/gated_tf).  The two experts are therefore trained independently and
      each owns its own stem / Mamba backbone weights.
    * Inference (Sec.4) — this class loads BOTH specialised experts and runs
      them on the same input, producing a prediction per expert (y_c, y_g).
      A policy then routes / blends / abstains on top of those predictions.

So the "MoE" in DARE-Net is realised as *specialised training* plus
*prediction-level routing*, rather than a single network that jointly
supervises both experts in every step.  Keeping the two experts as two loaded
``DareNet`` instances is the honest and reproducible representation of that
design.
"""
import torch
import torch.nn as nn

from neural_methods.model.DareNet import DareNet


def _strip_module(state_dict):
    """Remove the ``module.`` prefix produced by torch.nn.DataParallel."""
    return {k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


class DualExpertDareNet(nn.Module):
    """Two specialised DARE-Net experts plus routing operators.

    Args:
        complex_ckpt: path to the Complex (spectral) expert checkpoint.
        gated_ckpt:   path to the Gated (gated_tf) expert checkpoint.
        dataset_name: name of the training dataset (affects nothing but the
            ``dataset_name`` bookkeeping of the underlying ``DareNet``).
        device:       torch device.
    """

    def __init__(self, complex_ckpt, gated_ckpt, dataset_name="UBFC", device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        # Complex (precision) expert  ->  spectral pathway
        self.expert_c = self._load(DareNet(dataset_name=dataset_name, ffn_branch="spectral"),
                                   complex_ckpt)
        # Gated (robust) expert       ->  gated_tf pathway
        self.expert_g = self._load(DareNet(dataset_name=dataset_name, ffn_branch="gated_tf"),
                                   gated_ckpt)
        self.eval()

    def _load(self, model, ckpt):
        sd = _strip_module(torch.load(ckpt, map_location="cpu"))
        model.load_state_dict(sd, strict=True)
        return model.to(self.device)

    @staticmethod
    def _zscore(y):
        return (y - y.mean(-1, keepdim=True)) / (y.std(-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def predict(self, x):
        """Return ``(y_c, y_g, fp)``: z-scored waveforms of both experts and
        the domain fingerprint (temporal-difference energy, spectral entropy)
        of the input window.
        """
        x = x.to(self.device)
        y_c, fp = self.expert_c(x, return_fp=True)
        y_g, _ = self.expert_g(x, return_fp=True)
        return self._zscore(y_c), self._zscore(y_g), fp

    # ------------------------------------------------------------------
    # Routing operators (paper Sec.4 equations)
    # ------------------------------------------------------------------
    @staticmethod
    def fixed_avg(y_c, y_g):
        """FixedAvg: ``(y_c + y_g) / 2``."""
        return (y_c + y_g) / 2.0

    @staticmethod
    def soft_blend(y_c, y_g, alpha):
        """SoftBlend: ``alpha * y_c + (1 - alpha) * y_g``, ``alpha`` in [0, 1]
        (a scalar or a per-sample tensor of shape ``[B]``).
        """
        if torch.is_tensor(alpha):
            alpha = alpha.to(device=y_c.device, dtype=y_c.dtype).reshape(-1, 1)
        return alpha * y_c + (1 - alpha) * y_g

    @staticmethod
    def hard_select(y_c, y_g, pi):
        """HardSelector: ``1{pi=c} * y_c + 1{pi=g} * y_g``, where ``pi`` is a
        per-sample binary tensor of shape ``[B]`` (1 -> Complex, 0 -> Gated).
        """
        pi = pi.to(device=y_c.device, dtype=y_c.dtype).reshape(-1, 1)
        return pi * y_c + (1 - pi) * y_g
