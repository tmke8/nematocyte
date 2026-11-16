"""Optimisation and training code."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from torch.nn import Parameter
from torch.optim import Optimizer

__all__ = ["OptimisationCfg"]


@dataclass
class OptimisationCfg:
    """Config class for the optimisation."""

    lr: float = field(default=5.0e-5, doc="Learning rate.")
    weight_decay: float = field(default=0.0, doc="Weight decay.")
    optimizer_cls: str = field(
        default="torch.optim.AdamW", doc="Import path of the optimizer class."
    )
    optimizer_kwargs: dict[str, Any] | None = field(
        default=None, doc="Further keyword arguments for the optimizer."
    )

    def build(self, params: Iterator[tuple[str, Parameter]]) -> Optimizer:
        # Instantiate the optimizer for the given parameters.
        raise NotImplementedError()
