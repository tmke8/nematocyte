"""Neural network models definitions."""

from dataclasses import dataclass, field
from typing import Annotated, Literal, NotRequired, TypedDict

from nematocyte import option
from torch import nn

type Activation = Literal["relu", "gelu", "selu", "silu"]


def _get_activation(activation: Activation) -> nn.Module:
    match activation:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "selu":
            return nn.SELU()
        case "silu":
            return nn.SiLU()


type NormType = (
    Annotated[BatchNorm, option("BN")]
    | Annotated[LayerNorm, option("LN")]
    | Annotated[NoNorm, option("none")]
)


class BatchNormKwargs(TypedDict):
    eps: NotRequired[float]
    momentum: NotRequired[float]


@dataclass(frozen=True)
class BatchNorm:
    """Batch normalization."""

    # A config can also contain TypedDicts. A key that is `NotRequired` and not set in
    # the config is left out of the dict, so PyTorch's own default applies.
    kwargs: BatchNormKwargs = field(default_factory=BatchNormKwargs)


class LayerNormKwargs(TypedDict):
    eps: NotRequired[float]
    elementwise_affine: NotRequired[bool]


@dataclass(frozen=True)
class LayerNorm:
    """Layer normalization."""

    kwargs: LayerNormKwargs = field(default_factory=LayerNormKwargs)


@dataclass(frozen=True)
class NoNorm:
    """No normalization layer."""


def _get_norm(norm: NormType, num_features: int) -> nn.Module | None:
    match norm:
        case BatchNorm(kwargs):
            return nn.BatchNorm1d(num_features, **kwargs)
        case LayerNorm(kwargs):
            return nn.LayerNorm(num_features, **kwargs)
        case _:
            return None


@dataclass(kw_only=True, frozen=True)
class FcnFactory:
    """Factory for fully-connected neural networks."""

    num_hidden: int = field(default=1, doc="Number of hidden layers.")
    hidden_dim: int | None = field(
        default=None,
        doc="Width of the hidden layers. If not set, the input dimension is used.",
    )
    activation: Activation = field(default="gelu", doc="Activation function.")
    norm: NormType = field(default_factory=LayerNorm, doc="Normalization layer to use.")
    dropout_prob: float = field(
        default=0.0, doc="Dropout probability. No dropout is applied if this is 0."
    )
    final_bias: bool = field(
        default=True, doc="Whether the final linear layer has a bias."
    )
    input_norm: bool = field(
        default=False, doc="Whether to also apply the normalization layer to the input."
    )

    def _make_block(self, in_features: int, *, out_features: int) -> nn.Sequential:
        block = nn.Sequential()
        block.append(nn.Linear(in_features, out_features))
        if (norm := _get_norm(self.norm, out_features)) is not None:
            block.append(norm)
        block.append(_get_activation(self.activation))
        if self.dropout_prob > 0:
            block.append(nn.Dropout(p=self.dropout_prob))
        return block

    def build(
        self, in_dim: int, *, out_dim: int, with_flatten: bool = False
    ) -> nn.Sequential:
        predictor = nn.Sequential()
        if with_flatten:
            predictor.append(nn.Flatten())
        if self.input_norm and (norm := _get_norm(self.norm, in_dim)) is not None:
            predictor.append(norm)
        curr_dim = in_dim
        if self.num_hidden > 0:
            hidden_dim = in_dim if self.hidden_dim is None else self.hidden_dim
            for _ in range(self.num_hidden):
                predictor.append(
                    self._make_block(in_features=curr_dim, out_features=hidden_dim)
                )
                curr_dim = hidden_dim
        predictor.append(
            nn.Linear(in_features=curr_dim, out_features=out_dim, bias=self.final_bias)
        )
        return predictor


@dataclass(kw_only=True, frozen=True)
class SimpleCNNFactory:
    """Factory for a very simple CNN."""

    kernel_size: int = field(default=5, doc="Kernel size of the convolutions.")
    pool_stride: int = field(default=2, doc="Stride of the max-pooling layers.")
    activation: Activation = field(default="relu", doc="Activation function.")

    def build(self, in_dim: int, *, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=6, kernel_size=self.kernel_size),
            _get_activation(self.activation),
            nn.MaxPool2d(kernel_size=2, stride=self.pool_stride),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=self.kernel_size),
            _get_activation(self.activation),
            nn.MaxPool2d(kernel_size=2, stride=self.pool_stride),
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            _get_activation(self.activation),
            nn.Linear(in_features=120, out_features=84),
            _get_activation(self.activation),
            nn.Linear(in_features=84, out_features=out_dim),
        )
