"""Defaults of groups and subconfigs (`design/unions.md`)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import nematocyte
from nematocyte import option

from .helpers import write_files


@dataclass(kw_only=True)
class Fcn:
    num_hidden: int = 1


@dataclass(kw_only=True)
class Cnn:
    kernel_size: int = 5
    stride: int = 2


type Model = Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")]


@dataclass
class Opt:
    lr: float = 5e-05
    wd: float = 0.0


@dataclass(kw_only=True)
class Config:
    # The default is one specific config, not a change to what `cnn` means.
    model: Model = field(default_factory=lambda: Cnn(kernel_size=3))
    encoder: Model = field(default_factory=Fcn)
    opt: Opt = field(default_factory=lambda: Opt(lr=0.1))


def configs(tmp_path: Path) -> Path:
    return write_files(
        tmp_path / "configs",
        {
            "select_cnn.yaml": "defaults:\n  model: cnn\n",
            "set_wd.yaml": "opt:\n  wd: 0.5\n",
            "model/small_cnn.yaml": "defaults:\n  _self_: cnn\n\nstride: 1\n",
            "opt/preset.yaml": "wd: 0.1\n",
        },
    )


def test_default_option_follows_from_the_default_value() -> None:
    config = nematocyte.compose(Config)
    assert type(config.model) is Cnn
    assert type(config.encoder) is Fcn


# The table "A default instance is the config used when nothing is selected".


def test_default_instance_without_selection() -> None:
    assert nematocyte.compose(Config).model == Cnn(kernel_size=3)


def test_values_apply_on_top_of_the_default_instance() -> None:
    config = nematocyte.compose(Config, overrides=["model.stride=1"])
    assert config.model == Cnn(kernel_size=3, stride=1)


def test_selecting_the_option_starts_from_the_class_defaults(tmp_path: Path) -> None:
    config = nematocyte.compose(Config, configs(tmp_path), config_name="select_cnn")
    assert config.model == Cnn(kernel_size=5)
    assert nematocyte.compose(Config, overrides=["model=cnn"]).model == Cnn()


def test_selecting_another_option_gives_its_class_defaults() -> None:
    # In Hydra this is a validation error: a default instance fixes the class.
    config = nematocyte.compose(Config, overrides=["model=fcn"])
    assert config.model == Fcn()


def test_named_config_ends_at_an_option_and_its_class_defaults(tmp_path: Path) -> None:
    config = nematocyte.compose(
        Config, configs(tmp_path), overrides=["model=small_cnn"]
    )
    assert config.model == Cnn(kernel_size=5, stride=1)


# The table "Class defaults are requested by naming an option", for a plain
# subconfig: a preset never requests the class defaults.


def test_subconfig_default_instance() -> None:
    assert nematocyte.compose(Config).opt == Opt(lr=0.1)


def test_subconfig_override_keeps_the_default_instance() -> None:
    config = nematocyte.compose(Config, overrides=["opt.wd=0.5"])
    assert config.opt == Opt(lr=0.1, wd=0.5)


def test_subconfig_value_in_primary_config_keeps_the_default_instance(
    tmp_path: Path,
) -> None:
    config = nematocyte.compose(Config, configs(tmp_path), config_name="set_wd")
    assert config.opt == Opt(lr=0.1, wd=0.5)


def test_preset_applies_on_top_of_the_default_instance(tmp_path: Path) -> None:
    directory = configs(tmp_path)
    config = nematocyte.compose(Config, directory, overrides=["opt=preset"])
    assert config.opt == Opt(lr=0.1, wd=0.1)
    # The same result as setting the value directly.
    assert config == nematocyte.compose(Config, directory, overrides=["opt.wd=0.1"])


def test_same_group_at_two_paths_with_different_choices() -> None:
    config = nematocyte.compose(Config, overrides=["model=fcn", "encoder=cnn"])
    assert type(config.model) is Fcn
    assert type(config.encoder) is Cnn


def test_nested_defaults_come_from_the_default_instance() -> None:
    @dataclass
    class Inner:
        x: int = 1
        y: int = 2

    @dataclass
    class Outer:
        inner: Inner = field(default_factory=Inner)
        z: int = 3

    @dataclass
    class Root:
        outer: Outer = field(default_factory=lambda: Outer(inner=Inner(x=10), z=30))

    config = nematocyte.compose(Root, overrides=["outer.inner.y=20"])
    assert config.outer == Outer(inner=Inner(x=10, y=20), z=30)
