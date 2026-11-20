"""Finding and merging config files (`design/composition.md`)."""

from dataclasses import dataclass, field
import datetime
from pathlib import Path
import sys
from typing import Annotated, Any
import warnings

import pytest

import nematocyte
from nematocyte import ConfigError, DroppedValueWarning, option
from nematocyte._files import load_file

from .helpers import (
    CONFIG_FILES,
    BatchNorm,
    Celeba,
    Cmnist,
    Cnn,
    Config,
    Fcn,
    LayerNorm,
    Opt,
    compose_settings,
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
    # No `# @package _global_` and no leading slash in `_defaults_`.
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
    # The selection replaces the node: `default_res: 64` of `celeba_small` is
    # gone, and `download: true` is there because the file sets it again.
    assert config.dm == Celeba(superclass="Blond_Hair", download=True)


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
            "opt/a.yaml": "_defaults_:\n  _self_: b\n",
            "opt/b.yaml": "_defaults_:\n  _self_: a\n",
            "loop.yaml": "_defaults_:\n  _self_: loop\n",
        },
    )
    with pytest.raises(ConfigError, match="extends itself"):
        nematocyte.compose(Config, config_dir, overrides=["opt=a"])
    with pytest.raises(ConfigError, match="extends itself"):
        nematocyte.compose(Config, config_dir, config_name="loop")


def test_self_of_a_preset_cannot_name_something_missing(config_dir: Path) -> None:
    write_files(config_dir, {"opt/bad.yaml": "_defaults_:\n  _self_: nope\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["opt=bad"])
    message = str(excinfo.value)
    assert "bad.yaml" in message
    assert "'nope'" in message
    assert "weight_decay_adamw" in message


# -- `_defaults_` -----------------------------------------------------------


def test_defaults_as_a_list_is_an_error(config_dir: Path) -> None:
    write_files(config_dir, {"hydra_style.yaml": "_defaults_:\n  - model: cnn\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="hydra_style")
    assert "hydra_style.yaml" in str(excinfo.value)
    assert "mapping" in str(excinfo.value)


def test_place_given_twice_is_reported_by_the_parser(config_dir: Path) -> None:
    write_files(config_dir, {"twice.yaml": "_defaults_:\n  model: cnn\n  model: fcn\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="twice")
    assert "twice.yaml" in str(excinfo.value)
    assert "duplicate" in str(excinfo.value).lower()


def test_order_of_defaults_has_no_meaning(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "one.yaml": "_defaults_:\n  model: my_cnn\n  model.norm: my_batchnorm\n",
            "two.yaml": "_defaults_:\n  model.norm: my_batchnorm\n  model: my_cnn\n",
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
_defaults_:
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
        {"bad.yaml": "_defaults_:\n  model: fcn\n  model.norm: my_batchnorm\n"},
    )
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="bad")
    message = str(excinfo.value)
    assert "bad.yaml" in message
    assert "'norm'" in message
    assert "Fcn" in message


def test_defaults_cannot_select_for_a_value(config_dir: Path) -> None:
    write_files(config_dir, {"bad.yaml": "_defaults_:\n  seed: one\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="bad")
    assert "bad.yaml" in str(excinfo.value)
    assert "'seed'" in str(excinfo.value)


def test_hydra_null_entry_is_explained(config_dir: Path) -> None:
    write_files(config_dir, {"bad.yaml": "_defaults_:\n  opt: null\n"})
    with pytest.raises(ConfigError, match="not needed"):
        nematocyte.compose(Config, config_dir, config_name="bad")


def test_selection_among_the_values_is_explained(config_dir: Path) -> None:
    write_files(config_dir, {"bad.yaml": "model: cnn\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="bad")
    assert "bad.yaml" in str(excinfo.value)
    assert "_defaults_" in str(excinfo.value)


def test_hydras_name_for_the_key_is_explained(config_dir: Path) -> None:
    write_files(config_dir, {"hydra_style.yaml": "defaults:\n  model: cnn\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="hydra_style")
    message = str(excinfo.value)
    assert "hydra_style.yaml" in message
    assert "'defaults' is not a field of Config" in message
    assert "'_defaults_'" in message


def test_hydras_name_for_the_key_in_a_group_file(config_dir: Path) -> None:
    write_files(config_dir, {"model/mine.yaml": "defaults:\n  _self_: cnn\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["model=mine"])
    message = str(excinfo.value)
    assert "model/mine.yaml" in message
    assert "'_defaults_'" in message


@dataclass
class WithDefaultsField:
    defaults: list[str] = field(default_factory=list)


def test_a_field_called_defaults_is_set_from_a_file(tmp_path: Path) -> None:
    write_files(tmp_path, {"run.yaml": "defaults: [a, b]\n"})
    config = nematocyte.compose(WithDefaultsField, tmp_path, config_name="run")
    assert config.defaults == ["a", "b"]


# -- layers -------------------------------------------------------------------
# `good_run` sets `model.activation: selu`, `dm.download: true` and
# `opt.weight_decay: 0.001`.


def test_preset_wins_over_the_extended_config(config_dir: Path) -> None:
    write_files(
        config_dir,
        {"child.yaml": "_defaults_:\n  _self_: good_run\n  opt: weight_decay_adamw\n"},
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


@pytest.mark.parametrize(("option", "model"), [("fcn", Fcn()), ("cnn", Cnn())])
def test_option_replaces_the_node(
    config_dir: Path, option: str, model: Fcn | Cnn
) -> None:
    write_files(
        config_dir,
        {"child.yaml": f"_defaults_:\n  _self_: good_run\n  model: {option}\n"},
    )
    with pytest.warns(DroppedValueWarning, match="model.activation=selu"):
        in_file = nematocyte.compose(Config, config_dir, config_name="child")
    with pytest.warns(DroppedValueWarning, match="model.activation=selu"):
        on_cli = nematocyte.compose(
            Config, config_dir, config_name="good_run", overrides=[f"model={option}"]
        )
    # The class defaults; `activation: selu` from `good_run` is dropped, even
    # when the option is `fcn`, the class that `good_run` selected.
    assert in_file.model == model
    assert on_cli.model == model


def test_named_config_replaces_the_node(config_dir: Path) -> None:
    with pytest.warns(DroppedValueWarning, match="dm.download=true"):
        config = nematocyte.compose(
            Config, config_dir, config_name="good_run", overrides=["dm=celeba_blond"]
        )
    assert config.dm == Celeba(superclass="Blond_Hair", download=False)


def test_values_of_a_file_win_over_its_defaults(config_dir: Path) -> None:
    write_files(
        config_dir,
        {"child.yaml": "_defaults_:\n  opt: weight_decay_adamw\nopt:\n  lr: 3.0\n"},
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


def test_directory_named_after_a_value_holds_no_primary_configs(
    config_dir: Path,
) -> None:
    write_files(config_dir, {"seed/one.yaml": "seed: 1\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="seed/one")
    message = str(excinfo.value)
    assert "seed/one" in message
    assert "configs/seed/" in message
    assert "'seed'" in message
    assert "is a value" in message
    # A value has no named configs to select.
    assert "seed=one" not in message


def test_primary_config_for_a_dict_replaces_it_as_a_whole(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "kwargs.yaml": "opt:\n  optimizer_kwargs:\n    amsgrad: true\n",
            "kwargs_heavy.yaml": "opt:\n  optimizer_kwargs:\n    eps: 1.0e-6\n",
        },
    )
    config = nematocyte.compose(
        Config, config_dir, config_name="kwargs", overrides=["+kwargs_heavy"]
    )
    # Hydra would merge the two dicts.
    assert config.opt.optimizer_kwargs == {"eps": 1e-6}


@pytest.mark.parametrize(
    "name", ["model.yaml", "opt.yml", "nematocyte.yaml", "model.json5"]
)
def test_file_at_the_top_named_after_a_subconfig_or_group_is_an_error(
    config_dir: Path, name: str
) -> None:
    write_files(config_dir, {name: "{}\n"})
    # Whatever is composed: the file is a mistake wherever it is used.
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="good_run")
    message = str(excinfo.value)
    stem = name.split(".", maxsplit=1)[0]
    assert name in message
    assert f"configs/{stem}/" in message
    assert f"{stem}=NAME" in message


def test_file_at_the_top_named_after_a_value_is_a_primary_config(
    config_dir: Path,
) -> None:
    write_files(config_dir, {"seed.yaml": "seed: 4\n"})
    assert nematocyte.compose(Config, config_dir, config_name="seed").seed == 4


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
            "model/encoder.yaml": "_defaults_:\n  _self_: plain\nwidth: 7\n",
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
        {"bad.yaml": "_defaults_:\n  model: cnn\nmodel:\n  num_hidden: 2\n"},
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


@pytest.mark.parametrize(
    ("files", "args"),
    [
        (["short.yaml", "short.yml"], {"config_name": "short"}),
        (["short.yaml", "short.yml"], {"config_name": "short.yml"}),
        (["opt/fast.yaml", "opt/fast.yml"], {"overrides": ["opt=fast"]}),
        (["short.yaml", "short.json5"], {"config_name": "short"}),
        (["short.yml", "short.json5"], {"config_name": "short.json5"}),
        (["opt/fast.yaml", "opt/fast.json5"], {"overrides": ["opt=fast"]}),
    ],
)
def test_same_name_with_both_suffixes_is_an_error(
    config_dir: Path, files: list[str], args: dict[str, Any]
) -> None:
    write_files(config_dir, dict.fromkeys(files, "{}\n"))
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, **args)
    message = str(excinfo.value)
    assert all(name.split("/")[-1] in message for name in files)
    assert "ambiguous" in message


@pytest.mark.parametrize("other", ["fast.yml", "fast.json5"])
def test_same_name_with_both_suffixes_in_a_listing(
    config_dir: Path, other: str
) -> None:
    # Also when the clash is only met while listing the names in a message.
    write_files(config_dir, {"opt/fast.yaml": "{}\n", f"opt/{other}": "{}\n"})
    with pytest.raises(ConfigError, match=other):
        nematocyte.compose(Config, config_dir, overrides=["opt=slow"])


# -- JSON5 files ----------------------------------------------------------------

JSON5_GOOD_RUN = """\
// The same as good_run.yaml, in JSON5.
{
  _doc_: 'A good run, on small images.',
  _defaults_: {model: 'single_linear_layer', dm: 'celeba_small'},
  seed: 1,
  model: {activation: 'selu'},
  dm: {download: true},
  opt: {
    lr: 0.001,
    weight_decay: 1e-3,  // a trailing comma follows
  },
}
"""


@pytest.mark.parametrize("config_name", ["json_run", "json_run.json5"])
def test_json5_primary_config(config_dir: Path, config_name: str) -> None:
    write_files(config_dir, {"json_run.json5": JSON5_GOOD_RUN})
    config = nematocyte.compose(Config, config_dir, config_name=config_name)
    assert config == nematocyte.compose(Config, config_dir, config_name="good_run")


def test_json5_doc(config_dir: Path) -> None:
    write_files(config_dir, {"json_run.json5": JSON5_GOOD_RUN})
    assert load_file(config_dir / "json_run.json5").doc == (
        "A good run, on small images."
    )


def test_json5_and_yaml_extend_each_other(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            # A JSON5 config that extends a YAML one...
            "opt/fast.json5": "{_defaults_: {_self_: 'strong_weight_decay'}, lr: 0.5}",
            # ...and a YAML config that extends the JSON5 one.
            "opt/faster.yaml": "_defaults_:\n  _self_: fast\nlr: 0.9\n",
        },
    )
    config = nematocyte.compose(Config, config_dir, overrides=["opt=fast"])
    assert config.opt == Opt(lr=0.5, weight_decay=0.1)
    config = nematocyte.compose(Config, config_dir, overrides=["opt=faster"])
    assert config.opt == Opt(lr=0.9, weight_decay=0.1)


def test_json5_values(tmp_path: Path) -> None:
    path = tmp_path / "file.json5"
    path.write_text("{a: 0x10, b: Infinity, c: '${seed}', d: [1, 'x',]}")
    assert load_file(path).values == {
        "a": 16,
        "b": float("inf"),
        "c": "${seed}",
        "d": [1, "x"],
    }


@dataclass(kw_only=True)
class Dated:
    day: datetime.date = datetime.date(2000, 1, 1)
    anything: Any = None
    name: str = ""


def test_dates_in_json5_and_yaml(tmp_path: Path) -> None:
    # JSON5 has no dates: a `date` field gets the same value from both
    # formats, an `Any` field does not.
    write_files(
        tmp_path,
        {
            "y.yaml": "day: 2026-11-16\nanything: 2026-11-16\n",
            "j.json5": "{day: '2026-11-16', anything: '2026-11-16'}",
        },
    )
    from_yaml = nematocyte.compose(Dated, tmp_path, config_name="y")
    from_json5 = nematocyte.compose(Dated, tmp_path, config_name="j")
    assert from_yaml.day == from_json5.day == datetime.date(2026, 11, 16)
    assert from_yaml.anything == datetime.date(2026, 11, 16)
    assert from_json5.anything == "2026-11-16"


def test_date_for_a_str_field(tmp_path: Path) -> None:
    write_files(
        tmp_path,
        {"j.json5": "{name: '2026-11-16'}", "y.yaml": "name: 2026-11-16\n"},
    )
    assert nematocyte.compose(Dated, tmp_path, config_name="j").name == "2026-11-16"
    # In YAML, the unquoted date is a `date`, not a string.
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Dated, tmp_path, config_name="y")
    message = str(excinfo.value)
    assert "y.yaml" in message
    assert "Dated.name" in message


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("{seed: 1, seed: 2}", ["seed", "JSON5"]),
        ("{seed: 1,,}", ["line 1", "JSON5"]),
        ("", ["JSON5"]),
        ("[1, 2]", ["mapping", "list"]),
    ],
)
def test_broken_json5_file(tmp_path: Path, content: str, expected: list[str]) -> None:
    path = tmp_path / "broken.json5"
    path.write_text(content)
    with pytest.raises(ConfigError) as excinfo:
        load_file(path)
    message = str(excinfo.value)
    assert "broken.json5" in message
    for part in expected:
        assert part in message


def test_json5_without_the_extra(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `None` in `sys.modules` makes the import fail, as if it were missing.
    monkeypatch.setitem(sys.modules, "json5", None)
    write_files(config_dir, {"opt/fast.json5": "{lr: 0.5}"})
    # Only reading a JSON5 file needs the extra; listing it does not.
    assert nematocyte.compose(Config, config_dir, config_name="good_run").seed == 1
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["opt=fast"])
    message = str(excinfo.value)
    assert "fast.json5" in message
    assert "nematocyte[json5]" in message


def test_subconfig_key_with_nothing_under_it_is_ignored(config_dir: Path) -> None:
    write_files(config_dir, {"empty.yaml": "model:\n  # nothing yet\nseed: 3\n"})
    config = nematocyte.compose(Config, config_dir, config_name="empty")
    assert config == Config(seed=3)


# -- added configs: +NAME ----------------------------------------------------


def test_added_config_applies_on_top_of_the_primary_config(config_dir: Path) -> None:
    config = nematocyte.compose(
        Config, config_dir, config_name="good_run", overrides=["+presets/debug"]
    )
    assert config.seed == 0
    # Only what the added config names changes: `weight_decay` of `good_run`
    # stays, and so does the model.
    assert config.opt == Opt(lr=0.1, weight_decay=0.001)
    assert config.model == Fcn(num_hidden=0, activation="selu")


def test_added_config_without_a_primary_config(config_dir: Path) -> None:
    config = nematocyte.compose(Config, config_dir, overrides=["+presets/debug"])
    assert config == Config(seed=0, opt=Opt(lr=0.1))


@pytest.mark.parametrize(
    "overrides",
    [["+presets/debug", "opt.lr=3.0"], ["opt.lr=3.0", "+presets/debug"]],
)
def test_command_line_values_win_over_an_added_config(
    config_dir: Path, overrides: list[str]
) -> None:
    config = nematocyte.compose(Config, config_dir, overrides=overrides)
    assert config.opt.lr == 3.0
    assert config.seed == 0


def test_added_config_composes_as_if_it_extended_the_primary_config(
    config_dir: Path,
) -> None:
    # `presets/debug_cnn` extends `presets/debug`; the primary config goes at
    # the bottom of that chain: good_run, then presets/debug, then
    # presets/debug_cnn.
    with pytest.warns(DroppedValueWarning, match="model.activation=selu"):
        config = nematocyte.compose(
            Config, config_dir, config_name="good_run", overrides=["+presets/debug_cnn"]
        )
    assert config.seed == 5
    assert config.opt == Opt(lr=0.1, weight_decay=0.001)
    # The selection replaces the node: `activation: selu` of `good_run` is gone.
    assert config.model == Cnn(stride=1)
    # The same chain, written out with `_self_`.
    write_files(
        config_dir,
        {
            "explicit/debug.yaml": "_defaults_:\n  _self_: good_run\n"
            + CONFIG_FILES["presets/debug.yaml"],
            "explicit/debug_cnn.yaml": CONFIG_FILES["presets/debug_cnn.yaml"].replace(
                "_self_: presets/debug", "_self_: explicit/debug"
            ),
        },
    )
    with pytest.warns(DroppedValueWarning, match="model.activation=selu"):
        explicit = nematocyte.compose(
            Config, config_dir, config_name="explicit/debug_cnn"
        )
    assert config == explicit


def test_command_line_selection_replaces_what_an_added_config_set(
    config_dir: Path,
) -> None:
    config = nematocyte.compose(
        Config, config_dir, overrides=["model=fcn", "+presets/debug_cnn"]
    )
    assert config.model == Fcn()


@pytest.mark.parametrize(
    ("overrides", "seed"),
    [
        (["+presets/debug", "+presets/seven"], 7),
        (["+presets/seven", "+presets/debug"], 0),
        # Given twice, only the last one counts.
        (["+presets/seven", "+presets/debug", "+presets/seven"], 7),
    ],
)
def test_several_added_configs_apply_in_order(
    config_dir: Path, overrides: list[str], seed: int
) -> None:
    write_files(config_dir, {"presets/seven.yaml": "seed: 7\n"})
    config = nematocyte.compose(Config, config_dir, overrides=overrides)
    assert config.seed == seed
    assert config.opt.lr == 0.1


def test_added_config_that_extends_the_primary_config_itself(
    config_dir: Path,
) -> None:
    write_files(
        config_dir,
        {"presets/good.yaml": "_defaults_:\n  _self_: good_run\nseed: 3\n"},
    )
    config = nematocyte.compose(
        Config, config_dir, config_name="good_run", overrides=["+presets/good"]
    )
    assert config == nematocyte.compose(
        Config, config_dir, config_name="good_run", overrides=["seed=3"]
    )


def test_added_config_can_set_nematocytes_settings(config_dir: Path) -> None:
    assert compose_settings(config_dir, ["+presets/debug"]).log == "debug"
    assert compose_settings(
        config_dir, ["+presets/debug", "nematocyte.log=warn"]
    ).log == ("warn")


def test_added_config_from_a_field_directory_is_an_error(config_dir: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["+model/single_linear_layer"])
    message = str(excinfo.value)
    assert "'+model/single_linear_layer'" in message
    assert "'model'" in message
    assert "model=single_linear_layer" in message


def test_unknown_added_config_lists_the_primary_configs(config_dir: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["+presets/debgu"])
    message = str(excinfo.value)
    assert "'+presets/debgu'" in message
    assert "presets/debug_cnn" in message
    assert "Did you mean 'presets/debug'?" in message


def test_dotted_name_of_an_added_config_suggests_slashes(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="'presets/debug'"):
        nematocyte.compose(Config, config_dir, overrides=["+presets.debug"])


def test_error_in_an_added_config_names_key_file_and_class(config_dir: Path) -> None:
    write_files(config_dir, {"presets/bad.yaml": "model:\n  hiden_dim: 10\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["+presets/bad"])
    message = str(excinfo.value)
    assert "presets/bad.yaml" in message
    assert "'hiden_dim'" in message
    assert "Fcn" in message


# -- the warning for a dropped value --------------------------------------


def _dropped(
    config_dir: Path, config_name: str | None, overrides: list[str]
) -> list[str]:
    """The messages of the drop warnings that composing gives."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nematocyte.compose(
            Config, config_dir, config_name=config_name, overrides=overrides
        )
    return [
        str(w.message) for w in caught if issubclass(w.category, DroppedValueWarning)
    ]


def test_drop_warning_names_key_value_file_and_selection(config_dir: Path) -> None:
    [message] = _dropped(config_dir, "good_run", ["model=fcn"])
    assert "'model.activation=selu', set in `configs/good_run.yaml`" in message
    assert "'model=fcn'" in message
    assert "'model'" in message
    # What `single_linear_layer` brought goes along with it.
    assert "num_hidden" not in message


def test_drop_warning_for_a_selection_in_a_file(config_dir: Path) -> None:
    write_files(
        config_dir, {"child.yaml": "_defaults_:\n  _self_: good_run\n  model: cnn\n"}
    )
    [message] = _dropped(config_dir, "child", [])
    assert message.startswith(
        "`configs/child.yaml` resets the config values in 'model' to the values "
        "given by 'cnn'. This drops a value that was set for 'model' before:"
    )
    assert "'model.activation=selu', set in `configs/good_run.yaml`" in message


def test_selecting_the_same_config_again_warns(config_dir: Path) -> None:
    # `good_run` selects `single_linear_layer` too; its `activation: relu`
    # replaces `selu`, and does not count as setting it again.
    [message] = _dropped(config_dir, "good_run", ["model=single_linear_layer"])
    assert "model.activation=selu" in message


def test_no_drop_warning_when_set_again(config_dir: Path) -> None:
    assert (
        _dropped(config_dir, "good_run", ["model=fcn", "model.activation=selu"]) == []
    )
    assert (
        _dropped(config_dir, "good_run", ["model=cnn", "model.activation=gelu"]) == []
    )
    write_files(
        config_dir,
        {
            "child.yaml": "_defaults_:\n  _self_: good_run\n  model: cnn\n"
            "model:\n  activation: selu\n"
        },
    )
    assert _dropped(config_dir, "child", []) == []


def test_no_drop_warning_when_the_value_is_the_same(config_dir: Path) -> None:
    write_files(config_dir, {"gelu.yaml": "model:\n  activation: gelu\n"})
    # `gelu` is the default of `Fcn.activation`, but not of `Cnn.activation`.
    assert _dropped(config_dir, "gelu", ["model=fcn"]) == []
    [message] = _dropped(config_dir, "gelu", ["model=cnn"])
    assert "'model.activation=gelu', set in `configs/gelu.yaml`" in message


def test_no_drop_warning_for_a_field_the_new_config_lacks(config_dir: Path) -> None:
    write_files(
        config_dir,
        {"cnn3.yaml": "_defaults_:\n  model: cnn\nmodel:\n  kernel_size: 3\n"},
    )
    assert _dropped(config_dir, "cnn3", ["model=fcn"]) == []


def test_drop_warning_for_a_dropped_selection(config_dir: Path) -> None:
    write_files(
        config_dir,
        {"ln.yaml": "_defaults_:\n  model: cnn\n  model.norm: layernorm\n"},
    )
    [message] = _dropped(config_dir, "ln", ["model=cnn"])
    assert "'model.norm=layernorm', selected in `configs/ln.yaml`" in message
    # `my_cnn` selects `layernorm` itself, so nothing is lost.
    assert _dropped(config_dir, "ln", ["model=my_cnn"]) == []


def test_no_drop_warning_for_what_the_replaced_config_brought(
    config_dir: Path,
) -> None:
    # `my_cnn` selects `norm: layernorm` and sets `stride: 1`; both go with it.
    write_files(config_dir, {"mine.yaml": "_defaults_:\n  model: my_cnn\n"})
    assert _dropped(config_dir, "mine", ["model=cnn"]) == []


def test_one_drop_warning_per_selection(config_dir: Path) -> None:
    write_files(
        config_dir,
        {"two.yaml": "_defaults_:\n  _self_: good_run\nmodel:\n  dropout_prob: 0.5\n"},
    )
    [message] = _dropped(config_dir, "two", ["model=fcn"])
    assert "'model.activation=selu', set in `configs/good_run.yaml`" in message
    assert "'model.dropout_prob=0.5', set in `configs/two.yaml`" in message
    assert "This drops values that were set for 'model' before:" in message
    assert "To keep the values, set them again after the selection" in message
    # Two selections that drop something give two warnings.
    assert len(_dropped(config_dir, "two", ["model=fcn", "dm=celeba_blond"])) == 2


def test_drop_warning_points_at_the_caller(config_dir: Path) -> None:
    with pytest.warns(DroppedValueWarning) as record:
        nematocyte.compose(
            Config, config_dir, config_name="good_run", overrides=["model=fcn"]
        )
    assert record[0].filename == __file__


def test_doc_of_a_file_is_not_a_value(tmp_path: Path) -> None:
    path = tmp_path / "file.yaml"
    path.write_text("_doc_: |\n  What the file is for.\n\n  More.\nseed: 1\n")
    file = load_file(path)
    assert file.doc == "What the file is for.\n\nMore.\n"
    assert file.values == {"seed": 1}


def test_doc_can_be_null(tmp_path: Path) -> None:
    path = tmp_path / "file.yaml"
    path.write_text("_doc_:\nseed: 1\n")
    assert load_file(path).doc == ""


def test_doc_has_to_be_a_string(tmp_path: Path) -> None:
    path = tmp_path / "file.yaml"
    path.write_text("_doc_: 3\n")
    with pytest.raises(ConfigError, match=r"(?s)file\.yaml.*'_doc_'.*string"):
        load_file(path)


def test_files_with_a_doc_compose(config_dir: Path) -> None:
    # `good_run.yaml`, `single_linear_layer.yaml` and `presets/debug.yaml` have
    # a `_doc_`; `experiments/good_run_blond.yaml` extends one that has.
    config = nematocyte.compose(
        Config,
        config_dir,
        config_name="experiments/good_run_blond",
        overrides=["+presets/debug"],
    )
    assert config.seed == 0
    assert config.model == Fcn(num_hidden=0, activation="selu")


def test_doc_below_the_top_level_is_an_error(config_dir: Path) -> None:
    write_files(config_dir, {"nested_doc.yaml": "opt:\n  _doc_: Fast.\n  lr: 1\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, config_name="nested_doc")
    message = str(excinfo.value)
    assert "nested_doc.yaml" in message
    assert "'_doc_'" in message
    assert "Opt" in message
    assert "top level" in message
