"""Samples of the public API, with the types that a type checker should infer.

This file is not run. It is checked with ty, pyright and mypy, to make sure
that the API is understood by all of them without a plugin.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, NotRequired, TypedDict, assert_type, cast
import typing_extensions

import nematocyte
from nematocyte import option


@dataclass(kw_only=True)
class Fcn:
    num_hidden: int = 1


@dataclass(kw_only=True)
class Cnn:
    kernel_size: int = 5


@dataclass
class Opt:
    lr: float = field(default=5e-5, doc="Learning rate.")


# A group through a reusable alias ...
type Model = Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")]


@dataclass(kw_only=True)
class Config:
    model: Model = field(default_factory=Fcn, doc="The model architecture.")
    # ... and a group that is declared inline.
    encoder: Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")] = field(
        default_factory=Cnn
    )
    opt: Opt = field(default_factory=Opt)
    seed: int = 42


# -- annotated unions ----------------------------------------------------------

assert_type(option("fcn"), nematocyte.Option)


def use_groups(config: Config) -> None:
    # The annotations do not change what a checker sees: a plain union.
    assert_type(config.model, Fcn | Cnn)
    assert_type(config.encoder, Fcn | Cnn)
    if isinstance(config.model, Fcn):
        assert_type(config.model.num_hidden, int)
    else:
        assert_type(config.model.kernel_size, int)
    assert_type(config.opt.lr, float)


# -- the entry point -----------------------------------------------------------


@nematocyte.main(Config, config_path="configs")
def main(config: Config) -> float:
    assert_type(config, Config)
    return config.opt.lr


assert_type(main, nematocyte.EntryPoint)
assert_type(main(), None)
assert_type(main(["seed=1"]), None)


@nematocyte.main(Config, config_path=Path("configs"), config_name="primary")
def main_without_result(config: Config) -> None:
    del config


# The decorator returns something that takes a function of `Config`.
assert_type(
    nematocyte.main(Config, config_path="configs"),
    Callable[[Callable[[Config], object]], nematocyte.EntryPoint],
)


def takes_another_class(config: Opt) -> None:
    del config


# A function that does not take `Config` is an error.
nematocyte.main(Config, config_path="configs")(takes_another_class)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

# -- compose ----------------------------------------------------------------------

assert_type(nematocyte.compose(Config), Config)
assert_type(
    nematocyte.compose(Config, "configs", overrides=["model=cnn", "seed=1"]), Config
)
assert_type(nematocyte.compose(Opt, Path("configs"), config_name="primary"), Opt)

# -- the config of a run -----------------------------------------------------------

saved = json.loads(Path("config.json").read_text())
assert_type(nematocyte.from_run_config(Config, saved), Config)
assert_type(
    nematocyte.from_run_config(
        Config, {"model": {"_option_": "cnn", "kernel_size": 3}, "seed": 1}
    ),
    Config,
)
assert_type(nematocyte.from_run_config(Opt, MappingProxyType({"lr": 0.1})), Opt)
tags: list[nematocyte.JSONValue] = ["a", 1, None, [True, 1.5], {"b": "c"}]
assert_type(nematocyte.from_run_config(Config, {"wandb": {"tags": tags}}), Config)
assert_type(nematocyte.run_config(), dict[str, nematocyte.JSONValue])
# A nested value has to be narrowed first, or the result cast.
model = nematocyte.run_config()["model"]
assert_type(model, nematocyte.JSONValue)
if isinstance(model, Mapping):
    option_name = model["_option_"]
    if isinstance(option_name, str):
        assert_type(option_name, str)
loose = cast("dict[str, Any]", nematocyte.run_config())
assert_type(loose["model"]["_option_"], Any)
assert_type(nematocyte.from_run_config(Config, nematocyte.run_config()), Config)
# The values have to be JSON values.
nematocyte.from_run_config(Config, {"opt": Opt()})  # type: ignore[dict-item]  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

# -- the other functions -------------------------------------------------------------

assert_type(nematocyte.to_dict(nematocyte.compose(Config)), dict[str, Any])
assert_type(nematocyte.output_dir(), Path)
assert_type(nematocyte.log_level(), int)
assert_type(nematocyte.job_num(), int)
assert_type(nematocyte.configure_logging(), None)
assert_type(nematocyte.configure_logging(file=False), None)
assert_type(
    nematocyte.validate_configs(Config, "configs"), list[nematocyte.ValidationResult]
)
result = nematocyte.validate_config(Config, Path("configs"), Path("primary.yaml"))
assert_type(result, nematocyte.ValidationResult)
assert_type(result.path, Path)
assert_type(result.error, nematocyte.ConfigError | None)
assert_type(result.ok, bool)
assert_type(
    nematocyte.validate_configs(Config, "configs", config_name="x", construct=False),
    list[nematocyte.ValidationResult],
)

# -- TypedDicts --------------------------------------------------------------------


# A TypedDict that is an option of a group names its option in `_option_`.
class Sgd(TypedDict):
    _option_: Literal["sgd"]
    momentum: float


class Adam(TypedDict):
    _option_: Literal["adam"]
    beta1: float
    eps: NotRequired[float]


class Plain(TypedDict):
    lr: float


class TypedConfig(TypedDict):
    """A root config that is a TypedDict."""

    seed: int
    optimizer: Sgd | Adam
    model: Model
    plain: Plain


assert_type(nematocyte.compose(TypedConfig), TypedConfig)
assert_type(nematocyte.from_run_config(TypedConfig, saved), TypedConfig)
assert_type(nematocyte.compose(Sgd, "configs", overrides=["momentum=0.9"]), Sgd)


@nematocyte.main(TypedConfig, config_path="configs")
def typed_main(config: TypedConfig) -> None:
    assert_type(config["seed"], int)
    assert_type(config["optimizer"], Sgd | Adam)
    assert_type(config["model"], Fcn | Cnn)
    # The option narrows the union, in `match` and in `if`.
    match config["optimizer"]:
        case {"_option_": "sgd"}:
            pass
        case _:
            pass
    optimizer = config["optimizer"]
    if optimizer["_option_"] == "adam":
        assert_type(optimizer, Adam)
    else:
        assert_type(optimizer, Sgd)


@dataclass(kw_only=True)
class WithTypedDicts:
    plain: Plain = field(default_factory=lambda: Plain(lr=0.1))
    optimizer: Sgd | Adam = field(
        default_factory=lambda: Sgd(_option_="sgd", momentum=0.9)
    )


assert_type(nematocyte.compose(WithTypedDicts).optimizer, Sgd | Adam)
assert_type(nematocyte.to_dict(nematocyte.compose(WithTypedDicts)), dict[str, Any])


assert_type(typed_main, nematocyte.EntryPoint)


def takes_a_typed_config(config: TypedConfig) -> None:
    del config


# A function that takes another class than the TypedDict is an error, and the
# other way round.
nematocyte.main(Config, config_path="configs")(takes_a_typed_config)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
nematocyte.main(TypedConfig, config_path="configs")(takes_another_class)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

assert_type(
    nematocyte.validate_configs(TypedConfig, "configs"),
    list[nematocyte.ValidationResult],
)


# A TypedDict from typing_extensions, which can be closed.
class Closed(typing_extensions.TypedDict, closed=True):
    _option_: Literal["closed"]
    lr: float


class WithClosed(typing_extensions.TypedDict):
    plain: Plain
    closed: Closed | Sgd


assert_type(nematocyte.compose(WithClosed), WithClosed)
assert_type(nematocyte.compose(WithClosed)["closed"], Closed | Sgd)
