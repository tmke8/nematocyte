"""The text of `--help` and `--cfg` (`design/helptext.md`)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from ruamel.yaml import YAML

from nematocyte import option
from nematocyte._compose import Composer
from nematocyte._help import format_cfg, format_help
from nematocyte._overrides import parse_override
from nematocyte._settings import Settings
from nematocyte._types import type_text

from .helpers import Activation, Config, write_files


def help_text(
    config_cls: type,
    config_dir: Path | None = None,
    overrides: tuple[str, ...] = (),
    width: int = 88,
) -> str:
    composer = Composer(config_cls, config_dir, Settings)
    root = composer.compose(None, [parse_override(text) for text in overrides])
    return format_help(composer, root, width)


def cfg_text(
    config_cls: type,
    config_dir: Path | None = None,
    overrides: tuple[str, ...] = (),
    *,
    own_settings: bool = False,
) -> str:
    composer = Composer(config_cls, config_dir, Settings)
    root = composer.compose(None, [parse_override(text) for text in overrides])
    return format_cfg(composer, root, own_settings=own_settings, width=88)


@dataclass
class Opt:
    """Settings for the optimiser.

    More text that does not belong in an overview.
    """

    lr: float = field(default=5e-05, doc="Learning rate.")


@dataclass(kw_only=True)
class Fcn:
    """Fully-connected network."""

    num_hidden: int = field(default=1, doc="Number of hidden layers.")
    activation: Activation = field(default="gelu", doc="Activation function.")


@dataclass(kw_only=True)
class Cnn:
    kernel_size: int = 5


@dataclass(kw_only=True)
class Small:
    """Train a model on a dataset."""

    model: Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")] = field(
        default_factory=Fcn, doc="The model architecture."
    )
    opt: Opt = field(default_factory=Opt)
    seed: int = field(default=42, doc="Random seed.")
    gpu: int = field(
        default=-1, doc="Index of the GPU to use. Set to -1 to use the CPU."
    )


def user_part(text: str) -> str:
    """The help without nematocyte's own settings and the flags."""
    return text.split("\nnematocyte:", maxsplit=1)[0]


def test_one_override_per_line() -> None:
    assert (
        user_part(help_text(Small, width=76))
        == """\
Train a model on a dataset.

model=fcn              The model architecture.
                       * fcn      Fully-connected network.
                         cnn
model.num_hidden=1     [int] Number of hidden layers.
model.activation=gelu  [relu|gelu|selu|silu] Activation function.

opt:                   Settings for the optimiser.
opt.lr=5e-05           [float] Learning rate.

seed=42                [int] Random seed.
gpu=-1                 [int] Index of the GPU to use. Set to -1 to use the
                       CPU.
"""
    )


def test_help_reflects_the_other_arguments() -> None:
    text = help_text(Small, overrides=("model=cnn", "seed=7"))
    assert "model=cnn " in text
    assert "  fcn      Fully-connected network." in text
    assert "* cnn" in text
    assert "model.kernel_size=5" in text
    assert "model.num_hidden" not in text
    assert "seed=7 " in text


def test_configs_from_files_are_listed_by_name(config_dir: Path) -> None:
    text = help_text(Config, config_dir)
    assert "From files: my_cnn, single_linear_layer" in text
    assert "Presets: strong_weight_decay, weight_decay_adamw" in text
    assert "From files: slurm_test" in text


def test_nested_group_is_shown_with_its_path(config_dir: Path) -> None:
    text = help_text(Config, config_dir, ("model=cnn",))
    assert "model.norm=batchnorm" in text
    assert "* batchnorm   Batch normalisation." in text
    assert "model.norm.momentum=0.1" in text
    assert "From files: my_batchnorm" in text


def test_subconfig_falls_back_to_the_class_docstring() -> None:
    @dataclass
    class Root:
        opt: Opt = field(default_factory=Opt)
        other: Opt = field(default_factory=Opt, doc="From the field.")

    text = help_text(Root)
    assert "opt:            Settings for the optimiser." in text
    assert "other:          From the field." in text


def test_field_without_a_doc_is_shown_without_one() -> None:
    text = help_text(Small, overrides=("model=cnn",))
    assert "model.kernel_size=5  [int]\n" in text


def test_long_key_gets_a_line_of_its_own() -> None:
    @dataclass
    class Root:
        root: str = field(
            default="/a/very/long/path/that/goes/on/and/on/and/on",
            doc="Directory of the data.",
        )
        seed: int = field(default=1, doc="Random seed.")

    text = help_text(Root)
    lines = text.splitlines()
    index = lines.index("root=/a/very/long/path/that/goes/on/and/on/and/on")
    assert lines[index + 1].strip() == "[str] Directory of the data."
    # One long default does not widen the column for every line.
    seed_line = next(line for line in lines if line.startswith("seed=1"))
    assert seed_line.index("[int]") == lines[index + 1].index("[str]") <= 40


def test_required_value_is_marked() -> None:
    @dataclass
    class Root:
        name: str = field(doc="Name of the run.")

    assert "name=???" in help_text(Root)
    assert "(required)" in help_text(Root)


def test_own_settings_and_flags_are_shown(config_dir: Path) -> None:
    text = help_text(Config, config_dir)
    assert "nematocyte.launcher=basic" in text
    assert "* basic " in text
    assert "submitit_slurm " in text
    assert "nematocyte.sweeper=basic" in text
    assert "nematocyte.output_dir=null" in text
    for flag in ("--help, -h", "--cfg, -c", "--config-name, -cn", "--multirun, -m"):
        assert flag in text


def test_launcher_settings_get_help_the_same_way(config_dir: Path) -> None:
    text = help_text(Config, config_dir, ("nematocyte.launcher=slurm_test",))
    assert "nematocyte.launcher.partition=test" in text
    assert "SLURM partition to submit to." in text


@pytest.mark.parametrize(
    ("tp", "expected"),
    [
        (int, "int"),
        (Path, "path"),
        (int | None, "int | null"),
        (list[int] | None, "list[int] | null"),
        (dict[int, int] | None, "dict[int, int] | null"),
        (dict[str, Any] | None, "dict[str, any] | null"),
        (Literal["relu", "gelu"], "relu|gelu"),
        (Activation, "relu|gelu|selu|silu"),
        (Literal["BN", "LN"] | None, "BN|LN | null"),
        (Annotated[float, "meta"], "float"),
        (tuple[int, ...], "tuple[int, ...]"),
    ],
)
def test_type_text(tp: Any, expected: str) -> None:
    assert type_text(tp) == expected


# -- --cfg ------------------------------------------------------------------


def test_cfg_is_yaml_with_comments() -> None:
    assert (
        cfg_text(Small)
        == """\
# Train a model on a dataset.

# The model architecture. Options: fcn, cnn. Selected: fcn.
model:
  # Number of hidden layers. (int)
  num_hidden: 1
  # Activation function. (relu|gelu|selu|silu)
  activation: gelu
# Settings for the optimiser.
opt:
  # Learning rate. (float)
  lr: 5e-05
# Random seed. (int)
seed: 42
# Index of the GPU to use. Set to -1 to use the CPU. (int)
gpu: -1
"""
    )


def test_cfg_is_valid_yaml(config_dir: Path) -> None:
    write_files(config_dir, {})
    overrides = (
        "wandb.name='123'",
        "wandb.tags=[a,b]",
        "opt.optimizer_kwargs={eps:1e-8}",
        "dm.label_map={1:2}",
    )
    for own_settings in (False, True):
        text = cfg_text(Config, config_dir, overrides, own_settings=own_settings)
        loaded = YAML(typ="safe", pure=True).load(text)
        assert loaded["seed"] == 42
        assert loaded["wandb"] == {"name": "123", "group": None, "tags": ["a", "b"]}
        assert loaded["opt"]["optimizer_kwargs"] == {"eps": 1e-8}
        assert loaded["dm"]["label_map"] == {1: 2}
        assert loaded["dm"]["root"] == "/data"
        assert ("nematocyte" in loaded) is own_settings
    assert loaded["nematocyte"]["launcher"] == {}
    assert loaded["nematocyte"]["output_dir"] is None
