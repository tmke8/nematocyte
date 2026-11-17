"""Samples of the public API, with the types that a type checker should infer.

This file is not run. It is checked with ty, pyright and mypy, to make sure
that the API is understood by all of them without a plugin.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, assert_type

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
