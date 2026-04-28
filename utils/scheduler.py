"""Learning rate schedulers."""

from __future__ import annotations

from torch.optim import Optimizer


class PolyLRWithWarmup:
    """Poly LR schedule with linear warmup, stepped per-iteration.

    lr(t) = base_lr * warmup_factor                              for t < warmup_iters
    lr(t) = base_lr * (1 - (t - warmup_iters) / (T - warmup))^p  otherwise
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int,
        power: float = 0.9,
        warmup_iters: int = 0,
        warmup_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.total_iters = max(total_iters, 1)
        self.power = power
        self.warmup_iters = max(warmup_iters, 0)
        self.warmup_ratio = warmup_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.last_iter = 0

    def step(self) -> None:
        t = self.last_iter
        if self.warmup_iters > 0 and t < self.warmup_iters:
            alpha = t / self.warmup_iters
            factor = self.warmup_ratio + (1.0 - self.warmup_ratio) * alpha
        else:
            progress = (t - self.warmup_iters) / max(self.total_iters - self.warmup_iters, 1)
            progress = min(max(progress, 0.0), 1.0)
            factor = (1.0 - progress) ** self.power
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor
        self.last_iter += 1

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]
