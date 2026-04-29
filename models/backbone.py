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
from typing import List, Optional, Sequence, Tuple, Union

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
        freeze_until_block: Optional[int] = None,
        hub_repo: str = _HUB_REPO,
        hub_source: str = "github",  # kept for backward compat; unused
    ) -> None:
        """
        Args:
            freeze: if True, fully freeze (no_grad forward, eval mode).
            freeze_until_block: if `freeze=False` and this is set to k>0,
                blocks 0..k-1 are frozen and blocks k..depth-1 are trainable
                (Strategy A: unfreeze last few transformer blocks for fine-tuning).
                Patch embed, pos_embed, register tokens, cls token are always frozen
                in this mode. The final layer norm is left trainable.
        """
        super().__init__()
        self.model_name = model_name
        self.aux_layer_idx = aux_layer_idx

        self.model = _load_dinov3_model(model_name, weights_path)

        # DINOv3 ViT-S/16 specifics
        self.embed_dim: int = getattr(self.model, "embed_dim", 384)
        self.patch_size: int = getattr(self.model, "patch_size", 16)
        # Number of register tokens (DINOv3 uses 4 by default)
        self.num_register_tokens: int = getattr(self.model, "num_register_tokens", 4)

        self._frozen: bool = False
        self._freeze_until_block: int = 0
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self._frozen = True
        elif freeze_until_block is not None and int(freeze_until_block) > 0:
            self._apply_partial_freeze(int(freeze_until_block))

    def _apply_partial_freeze(self, freeze_until_block: int) -> None:
        depth = self.depth
        k = max(0, min(freeze_until_block, depth))
        for name, p in self.model.named_parameters():
            if name.startswith("blocks."):
                block_idx = int(name.split(".")[1])
                p.requires_grad = block_idx >= k
            elif name.startswith("norm"):
                p.requires_grad = True  # final layer norm — keep trainable
            else:
                # patch_embed / pos_embed / cls_token / register_tokens / mask_token / rope
                p.requires_grad = False
        self._freeze_until_block = k

    def train(self, mode: bool = True):  # noqa: D401 - keep frozen parts in eval
        super().train(mode)
        if self._frozen:
            self.model.eval()
        else:
            self.model.train(mode)
        return self

    @property
    def is_fully_frozen(self) -> bool:
        return self._frozen

    @property
    def depth(self) -> int:
        if hasattr(self.model, "blocks"):
            return len(self.model.blocks)
        return 12  # ViT-S/16 default

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """tokens: (B, N, C) where N == h*w  ->  (B, C, h, w)."""
        b, n, c = tokens.shape
        assert n == h * w, f"token count {n} does not match grid {h}x{w}"
        return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _intermediate_maps(self, x: torch.Tensor, indices: Sequence[int]) -> List[torch.Tensor]:
        """Return 2D feature maps from the requested transformer block indices.

        All requested blocks get the final layernorm applied (norm=True), which
        matches `forward_features`' final layer behavior. Indices are clamped
        to [0, depth-1].
        """
        H, W = x.shape[-2:]
        h = H // self.patch_size
        w = W // self.patch_size
        depth = self.depth
        clamped = [max(0, min(int(i), depth - 1)) for i in indices]
        try:
            outs = self.model.get_intermediate_layers(
                x, n=list(clamped), reshape=False, norm=True
            )
        except TypeError:
            # older DINOv3 signature without `norm` kwarg
            outs = self.model.get_intermediate_layers(x, n=list(clamped))
        return [self._tokens_to_map(t, h, w) for t in outs]

    def _final_map(self, x: torch.Tensor) -> torch.Tensor:
        """Final post-norm patch tokens via forward_features."""
        H, W = x.shape[-2:]
        h = H // self.patch_size
        w = W // self.patch_size
        out = self.model.forward_features(x)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            patch_tokens = out["x_norm_patchtokens"]
        else:
            # Fallback: assume layout [CLS, register_tokens..., patch_tokens]
            tokens = out if isinstance(out, torch.Tensor) else out["x"]
            patch_tokens = tokens[:, 1 + self.num_register_tokens :, :]
        return self._tokens_to_map(patch_tokens, h, w)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_layers: Optional[Sequence[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        """
        Three call modes:

        - ``backbone(x)`` -> ``(B, C, h, w)`` final post-norm features.
        - ``backbone(x, return_aux=True)`` -> ``(final, aux)`` for backward compatibility,
          where ``aux`` is the ``aux_layer_idx``-th block's features.
        - ``backbone(x, return_layers=[i, j, ...])`` -> list of 2D maps in the requested order.
          Used by Strategy B (multi-layer feature alignment).

        For full-freeze (``self._frozen=True``) the forward runs under ``torch.no_grad``
        to save memory; for partial-freeze (Strategy A) the forward keeps its autograd
        graph so gradients can flow back through the unfrozen tail.
        """
        # We only suppress grads when the entire backbone is frozen — otherwise
        # autograd needs to retain activations for backprop through unfrozen blocks.
        ctx = torch.no_grad() if self._frozen else _NullCtx()
        with ctx:
            if return_layers is not None:
                return self._intermediate_maps(x, return_layers)

            if return_aux:
                depth = self.depth
                indices = [self.aux_layer_idx, depth - 1]
                feats = self._intermediate_maps(x, indices)
                feat_aux, feat_main = feats[0], feats[1]
                return feat_main, feat_aux

            return self._final_map(x)


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
