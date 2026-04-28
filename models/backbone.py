"""DINOv3 ViT backbone wrapper.

Loads a DINOv3 ViT (default ViT-S/16) and exposes:
    * a frozen forward producing a 2D feature map from patch tokens
    * an intermediate feature map (for the auxiliary head) extracted from
      a chosen Transformer block.

We bypass `torch.hub.load` because DINOv3's hubconf.py imports the
segmentor / detector / depther entrypoints at module top-level, which
transitively pulls torchmetrics, omegaconf, MultiScaleDeformableAttention
and other heavy deps that we don't need for a backbone-only use case.
Instead we import `dinov3.hub.backbones.<entrypoint>` directly from the
cached hub repo, cloning it on first use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


_HUB_REPO = "facebookresearch/dinov3"


def _ensure_dinov3_repo(hub_repo: str = _HUB_REPO) -> Path:
    """Make sure the DINOv3 repo is cached locally; return the cached path.

    Uses torch.hub's internal cache helper so we get the same on-disk layout
    as a normal `torch.hub.load`, but without executing hubconf.py.
    """
    cache_dir = Path(torch.hub.get_dir())
    repo_dir_name = hub_repo.replace("/", "_") + "_main"
    repo_dir = cache_dir / repo_dir_name
    if repo_dir.is_dir() and (repo_dir / "dinov3" / "hub" / "backbones.py").is_file():
        return repo_dir
    # _get_cache_or_reload is private but stable across recent torch versions.
    repo_dir_str = torch.hub._get_cache_or_reload(  # type: ignore[attr-defined]
        hub_repo, force_reload=False, trust_repo=True, calling_fn=None,
        verbose=True, skip_validation=True,
    )
    return Path(repo_dir_str)


def _load_dinov3_model(model_name: str, weights_path: Optional[str]) -> nn.Module:
    """Direct-import the requested DINOv3 backbone entrypoint and load weights."""
    repo_dir = _ensure_dinov3_repo()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from dinov3.hub import backbones as _bb  # noqa: WPS433
    factory = getattr(_bb, model_name, None)
    if factory is None:
        raise ValueError(
            f"Unknown DINOv3 backbone entrypoint: {model_name!r}. "
            f"Available: {[n for n in dir(_bb) if n.startswith('dinov3_')]}"
        )
    if weights_path is not None and os.path.isfile(weights_path):
        return factory(weights=weights_path, check_hash=False)
    return factory(pretrained=False)


class DINOv3Backbone(nn.Module):
    """Wrap a DINOv3 ViT and return spatial feature maps.

    Outputs:
        feat:     (B, C, h, w)  features from the final block (post-norm patch tokens)
        feat_aux: (B, C, h, w)  features from `aux_layer_idx`-th block (pre-final)
                  Returned only when `return_aux=True` in forward.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        weights_path: Optional[str] = None,
        aux_layer_idx: int = 6,
        freeze: bool = True,
        hub_repo: str = _HUB_REPO,
        hub_source: str = "github",  # kept for backward compat; unused
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.aux_layer_idx = aux_layer_idx

        self.model = _load_dinov3_model(model_name, weights_path)

        # DINOv3 ViT-S/16 specifics
        self.embed_dim: int = getattr(self.model, "embed_dim", 384)
        self.patch_size: int = getattr(self.model, "patch_size", 16)
        # Number of register tokens (DINOv3 uses 4 by default)
        self.num_register_tokens: int = getattr(self.model, "num_register_tokens", 4)

        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self._frozen = True
        else:
            self._frozen = False

    def train(self, mode: bool = True):  # noqa: D401 - keep frozen
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """tokens: (B, N, C) where N == h*w  ->  (B, C, h, w)."""
        b, n, c = tokens.shape
        assert n == h * w, f"token count {n} does not match grid {h}x{w}"
        return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def _frozen_forward(self, x: torch.Tensor, return_aux: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._raw_forward(x, return_aux)

    def _raw_forward(self, x: torch.Tensor, return_aux: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        H, W = x.shape[-2:]
        h = H // self.patch_size
        w = W // self.patch_size

        feat_main: torch.Tensor
        feat_aux: Optional[torch.Tensor] = None

        if return_aux and hasattr(self.model, "get_intermediate_layers"):
            # Total Transformer depth
            depth = len(self.model.blocks) if hasattr(self.model, "blocks") else 12
            aux_idx = max(0, min(self.aux_layer_idx, depth - 1))
            # Take aux block + last block; norm only the last one (matches forward_features)
            try:
                outs = self.model.get_intermediate_layers(
                    x, n=[aux_idx, depth - 1], reshape=False, norm=True
                )
                aux_tokens, main_tokens = outs[0], outs[1]
            except TypeError:
                # Fallback: older signature without `norm` kwarg
                outs = self.model.get_intermediate_layers(x, n=[aux_idx, depth - 1])
                aux_tokens, main_tokens = outs[0], outs[1]

            feat_main = self._tokens_to_map(main_tokens, h, w)
            feat_aux = self._tokens_to_map(aux_tokens, h, w)
            return feat_main, feat_aux

        # Default path: use forward_features and select patch tokens
        out = self.model.forward_features(x)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            patch_tokens = out["x_norm_patchtokens"]
        else:
            # Fallback: assume layout [CLS, register_tokens..., patch_tokens]
            tokens = out if isinstance(out, torch.Tensor) else out["x"]
            patch_tokens = tokens[:, 1 + self.num_register_tokens :, :]

        feat_main = self._tokens_to_map(patch_tokens, h, w)
        return feat_main, None

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if self._frozen:
            feat_main, feat_aux = self._frozen_forward(x, return_aux)
        else:
            feat_main, feat_aux = self._raw_forward(x, return_aux)
        if return_aux:
            return feat_main, feat_aux
        return feat_main
