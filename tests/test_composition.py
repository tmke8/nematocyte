"""Finding and merging config files (`design/composition.md`)."""

from dataclasses import dataclass, field
import datetime
from pathlib import Path
from typing import Annotated, Any

import pytest

import nematocyte
from nematocyte import ConfigError, option
from nematocyte._files import load_file

from .helpers import (
    BatchNorm,
    Celeba,
    Cmnist,
    Cnn,
    Config,
    Fcn,
    LayerNorm,
    Opt,
    write_files,
)


def test_no_primary_config() -> None:
    # Without one, the tree comes from the root dataclass alone.
    assert nematocyte.compose(Config) == Config()


def test_empty_config_name_means_no_primary_config(config_dir: Path) -> None:
    assert nematocyte.compose(Config, config_dir, config_name="") == Config()


def test_primary_config(config_dir: Path) -> None:
    config = nematocyte.compose(Config, config_dir, config_name="good_run")
    assert config.seed == 1
    assert config.model == Fcn(num_hidden=0, activation="selu")
    assert config.dm == Celeba(default_res=64, superclass="Smiling", download=True)
    assert config.opt == Opt(lr=0.001, weight_decay=0.001)


def test_primary_config_in_a_subdirectory(config_dir: Path) -> None:
    # No `# @package _global_` and no leading slash in `defaults`.
    config = nematocyte.compose(
        Config, config_dir, config_name="experiments/cmnist_cnn"
    )
    assert config.dm == Cmnist(val_prop=0.3)
    assert config.model == Cnn(kernel_size=3)
    assert config.opt == Opt(lr=0.001, weight_decay=0.01)


def test_config_name_may_have_the_suffix(config_dir: Path) -> None:
    config = nematocyte.compose(Config, config_dir, config_name="good_run.yaml")
    assert config.seed == 1


def test_named_config_chain_for_a_group(config_dir: Path) -> None:
    # celeba_small -> celeba_smiling -> celeba -> Celeba
    config = nematocyte.compose(Config, config_dir, overrides=["dm=celeba_small"])
    assert config.dm == Celeba(default_res=64, superclass="Smiling")


def test_preset_that_extends_another_preset(config_dir: Path) -> None:
    config = nematocyte.compose(
        Config, config_dir, overrides=["opt=strong_weight_decay"]
    )
    assert config.opt == Opt(lr=0.001, weight_decay=0.1)


def test_primary_config_that_extends_another(config_dir: Path) -> None:
    config = nematocyte.compose(
        Config, config_dir, config_name="experiments/good_run_blond"
    )
    assert config.seed == 2
    # What `good_run` sets for `model` and `opt` is kept.
    assert config.model == Fcn(num_hidden=0, activation="selu")
    assert config.opt == Opt(lr=0.001, weight_decay=0.001)
    # The selection replaces the node: `download: true` of `good_run` is gone.
    assert config.dm == Celeba(superclass="Blond_Hair", download=False)


# -- `_self_` -------------------------------------------------------------


def test_group_config_needs_self(config_dir: Path) -> None:
    write_files(config_dir, {"dm/broken.yaml": "superclass: Male\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["dm=broken"])
    message = str(excinfo.value)
    assert "broken.yaml" in message
    assert "_self_" in message
    assert "celeba" in message


def test_self_chain_that_comes_back_to_itself(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "opt/a.yaml": "defaults:\n  _self_: b\n",
            "opt/b.yaml": "defaults:\n  _self_: a\n",
            "loop.yaml": "defaults:\n  _self_: loop\n",
        },
    )
    with pytest.raises(ConfigError, match="extends itself"):
        nematocyte.compose(Config, config_dir, overrides=["opt=a"])
    with pytest.raises(ConfigError, match="extends itself"):
        nematocyte.compose(Config, config_dir, config_name="loop")


def test_self_of_a_preset_cannot_name_something_missing(config_dir: Path) -> None:
    write_files(config_dir, {"opt/bad.yaml": "defaults:\n  _self_: nope\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["opt=bad"])
    message = str(excinfo.value)
    assert "bad.yaml" in message
    assert "'nope'" in message
    assert "weight_decay_adamw" in message


# -- `defaults` -------------------------------------------------------------


def test_defaults_as_a_list_is_an_error(config_dir: Path) -> None:
    write_files(config_dir, {"hydra_style.yaml": "defaults:\n  - model: cnn\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="hydra_style")
    assert "hydra_style.yaml" in str(excinfo.value)
    assert "mapping" in str(excinfo.value)


def test_place_given_twice_is_reported_by_the_parser(config_dir: Path) -> None:
    write_files(config_dir, {"twice.yaml": "defaults:\n  model: cnn\n  model: fcn\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="twice")
    assert "twice.yaml" in str(excinfo.value)
    assert "duplicate" in str(excinfo.value).lower()


def test_order_of_defaults_has_no_meaning(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "one.yaml": "defaults:\n  model: my_cnn\n  model.norm: my_batchnorm\n",
            "two.yaml": "defaults:\n  model.norm: my_batchnorm\n  model: my_cnn\n",
        },
    )
    one = nematocyte.compose(Config, config_dir, config_name="one")
    two = nematocyte.compose(Config, config_dir, config_name="two")
    assert one == two


def test_defaults_key_can_be_a_path(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "nested.yaml": """\
defaults:
  model: my_cnn
  model.norm: my_batchnorm

model:
  norm:
    momentum: 0.2
""",
        },
    )
    config = nematocyte.compose(Config, config_dir, config_name="nested")
    # Shortest path first: `my_batchnorm` wins over what `my_cnn` selects for
    # `norm`, and the value is then set on it.
    assert config.model == Cnn(stride=1, norm=BatchNorm(momentum=0.2))


def test_named_config_selects_for_its_own_fields(config_dir: Path) -> None:
    config = nematocyte.compose(Config, config_dir, overrides=["model=my_cnn"])
    assert config.model == Cnn(stride=1, norm=LayerNorm())


def test_path_in_defaults_needs_the_field(config_dir: Path) -> None:
    write_files(
        config_dir,
        {"bad.yaml": "defaults:\n  model: fcn\n  model.norm: my_batchnorm\n"},
    )
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="bad")
    message = str(excinfo.value)
    assert "bad.yaml" in message
    assert "'norm'" in message
    assert "Fcn" in message


def test_defaults_cannot_select_for_a_value(config_dir: Path) -> None:
    write_files(config_dir, {"bad.yaml": "defaults:\n  seed: one\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="bad")
    assert "bad.yaml" in str(excinfo.value)
    assert "'seed'" in str(excinfo.value)


def test_hydra_null_entry_is_explained(config_dir: Path) -> None:
    write_files(config_dir, {"bad.yaml": "defaults:\n  opt: null\n"})
    with pytest.raises(ConfigError, match="not needed"):
        nematocyte.compose(Config, config_dir, config_name="bad")


def test_selection_among_the_values_is_explained(config_dir: Path) -> None:
    write_files(config_dir, {"bad.yaml": "model: cnn\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="bad")
    assert "bad.yaml" in str(excinfo.value)
    assert "defaults" in str(excinfo.value)


# -- layers -------------------------------------------------------------------
# `good_run` sets `model.activation: selu`, `dm.download: true` and
# `opt.weight_decay: 0.001`.


def test_preset_wins_over_the_extended_config(config_dir: Path) -> None:
    write_files(
        config_dir,
        {"child.yaml": "defaults:\n  _self_: good_run\n  opt: weight_decay_adamw\n"},
    )
    in_file = nematocyte.compose(Config, config_dir, config_name="child")
    on_cli = nematocyte.compose(
        Config, config_dir, config_name="good_run", overrides=["opt=weight_decay_adamw"]
    )
    assert in_file.opt.weight_decay == 0.01
    assert on_cli.opt.weight_decay == 0.01


def test_preset_keeps_the_other_values_of_lower_layers(config_dir: Path) -> None:
    write_files(config_dir, {"opt/only_wd.yaml": "weight_decay: 0.5\n"})
    config = nematocyte.compose(
        Config, config_dir, config_name="good_run", overrides=["opt=only_wd"]
    )
    assert config.opt == Opt(lr=0.001, weight_decay=0.5)


def test_option_replaces_the_node(config_dir: Path) -> None:
    write_files(
        config_dir, {"child.yaml": "defaults:\n  _self_: good_run\n  model: cnn\n"}
    )
    in_file = nematocyte.compose(Config, config_dir, config_name="child")
    on_cli = nematocyte.compose(
        Config, config_dir, config_name="good_run", overrides=["model=cnn"]
    )
    # A CNN with its class defaults; `activation: selu` is dropped.
    assert in_file.model == Cnn()
    assert on_cli.model == Cnn()


def test_named_config_replaces_the_node(config_dir: Path) -> None:
    config = nematocyte.compose(
        Config, config_dir, config_name="good_run", overrides=["dm=celeba_blond"]
    )
    assert config.dm == Celeba(superclass="Blond_Hair", download=False)


def test_values_of_a_file_win_over_its_defaults(config_dir: Path) -> None:
    write_files(
        config_dir,
        {"child.yaml": "defaults:\n  opt: weight_decay_adamw\nopt:\n  lr: 3.0\n"},
    )
    config = nematocyte.compose(Config, config_dir, config_name="child")
    assert config.opt == Opt(lr=3.0, weight_decay=0.01)


def test_command_line_wins_over_the_primary_config(config_dir: Path) -> None:
    config = nematocyte.compose(
        Config, config_dir, config_name="good_run", overrides=["seed=7"]
    )
    assert config.seed == 7


# -- where files can be ------------------------------------------------------


def test_directory_named_after_a_field_holds_no_primary_configs(
    config_dir: Path,
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="model/single_linear_layer")
    message = str(excinfo.value)
    assert "model/single_linear_layer" in message
    assert "'model'" in message


def test_unknown_primary_config_lists_the_ones_there_are(config_dir: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="god_run")
    message = str(excinfo.value)
    assert "'god_run'" in message
    assert "experiments/cmnist_cnn" in message
    assert "Did you mean 'good_run'?" in message
    # Files of a field's directory are not primary configs.
    assert "single_linear_layer" not in message


def test_named_configs_for_a_nested_field(config_dir: Path) -> None:
    config = nematocyte.compose(
        Config, config_dir, overrides=["model=cnn", "model.norm=my_batchnorm"]
    )
    assert config.model == Cnn(norm=BatchNorm(momentum=0.5))


def test_file_and_directory_with_the_same_stem_do_not_clash(tmp_path: Path) -> None:
    @dataclass
    class Encoder:
        depth: int = 1

    @dataclass
    class WithEncoder:
        encoder: Encoder = field(default_factory=Encoder)
        width: int = 1

    @dataclass
    class Plain:
        width: int = 2

    @dataclass
    class Root:
        model: (
            Annotated[WithEncoder, option("with_encoder")]
            | Annotated[Plain, option("plain")]
        ) = field(default_factory=WithEncoder)

    directory = write_files(
        tmp_path / "c",
        {
            # An option of `model` ...
            "model/encoder.yaml": "defaults:\n  _self_: plain\nwidth: 7\n",
            # ... and a named config for the field `encoder`.
            "model/encoder/deep.yaml": "depth: 9\n",
        },
    )
    assert nematocyte.compose(Root, directory, overrides=["model=encoder"]).model == (
        Plain(width=7)
    )
    config = nematocyte.compose(Root, directory, overrides=["model.encoder=deep"])
    assert config.model == WithEncoder(encoder=Encoder(depth=9))


def test_missing_config_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        nematocyte.compose(Config, tmp_path / "nope")


# -- the schema is always enforced -------------------------------------------


def test_primary_config_is_always_validated(config_dir: Path) -> None:
    write_files(config_dir, {"bad_value.yaml": "seed: not-a-number\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="bad_value")
    message = str(excinfo.value)
    assert "seed" in message
    assert "bad_value.yaml" in message
    assert "not-a-number" in message
    assert "Config.seed (int)" in message


def test_unknown_key_names_key_file_and_class(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "bad_key.yaml": "made_up_key: 1\n",
            "bad_nested.yaml": "model:\n  hiden_dim: 10\n",
        },
    )
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="bad_key")
    message = str(excinfo.value)
    assert "'made_up_key'" in message
    assert "bad_key.yaml" in message
    assert "Config" in message

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="bad_nested")
    message = str(excinfo.value)
    assert "'hiden_dim'" in message
    assert "bad_nested.yaml" in message
    assert "Fcn" in message
    assert "Did you mean 'hidden_dim'?" in message


def test_key_of_another_option_is_an_error(config_dir: Path) -> None:
    write_files(
        config_dir,
        {"bad.yaml": "defaults:\n  model: cnn\nmodel:\n  num_hidden: 2\n"},
    )
    with pytest.raises(ConfigError, match="num_hidden"):
        nematocyte.compose(Config, config_dir, config_name="bad")


# -- YAML 1.2 --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1e-3", 0.001),
        ("yes", "yes"),
        ("on", "on"),
        ("No", "No"),
        ("010", 10),
        ("1:30", "1:30"),
        ("1_000", 1000),
        ("3.10", 3.1),
        ("2026-11-16", datetime.date(2026, 11, 16)),
        ("007", 7),
        (".inf", float("inf")),
        ("~", None),
        ("???", "???"),
    ],
)
def test_yaml_values(tmp_path: Path, text: str, expected: Any) -> None:
    path = tmp_path / "file.yaml"
    path.write_text(f"key: {text}\n")
    value = load_file(path).values["key"]
    assert value == expected
    assert type(value) is type(expected)


def test_repeated_key_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "file.yaml"
    path.write_text("seed: 1\nseed: 2\n")
    with pytest.raises(ConfigError, match=r"file\.yaml"):
        load_file(path)


def test_file_has_to_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "file.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_file(path)


def test_empty_file_is_an_empty_config(tmp_path: Path) -> None:
    path = tmp_path / "file.yaml"
    path.write_text("# nothing here\n")
    assert load_file(path).values == {}
    assert load_file(path).defaults == {}


def test_yml_suffix(config_dir: Path) -> None:
    write_files(config_dir, {"short.yml": "seed: 5\n"})
    assert nematocyte.compose(Config, config_dir, config_name="short").seed == 5


def test_subconfig_key_with_nothing_under_it_is_ignored(config_dir: Path) -> None:
    write_files(config_dir, {"empty.yaml": "model:\n  # nothing yet\nseed: 3\n"})
    config = nematocyte.compose(Config, config_dir, config_name="empty")
    assert config == Config(seed=3)
