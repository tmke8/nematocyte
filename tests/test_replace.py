"""`_replace_: true`, which replaces a TypedDict instead of merging into it."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict
import warnings

from pydantic import Discriminator
import pytest

import nematocyte
from nematocyte import ConfigError, DroppedValueWarning, SchemaError
from nematocyte._schema import class_fields

from .helpers import write_files

# -- classes -------------------------------------------------------------------


class Opt(TypedDict):
    lr: float
    wd: NotRequired[float]


class Sgd(TypedDict):
    _option_: Literal["sgd"]
    momentum: float


class Adam(TypedDict):
    _option_: Literal["adam"]
    beta1: float
    eps: NotRequired[float]


type Optimizer = Annotated[Sgd | Adam, Discriminator("_option_")]


@dataclass(kw_only=True)
class Model:
    width: int = 8
    depth: int = 2


class Training(TypedDict):
    optimizer: Optimizer
    model: Model
    steps: NotRequired[int]


@dataclass(kw_only=True)
class Config:
    seed: int = 0
    opt: Opt = field(default_factory=lambda: Opt(lr=0.1, wd=0.01))
    optimizer: Optimizer = field(
        default_factory=lambda: Adam(_option_="adam", beta1=0.9, eps=1e-8)
    )
    ema: Opt | None = None
    model: Model = field(default_factory=Model)
    training: Training = field(
        default_factory=lambda: Training(
            optimizer=Sgd(_option_="sgd", momentum=0.9),
            model=Model(width=16),
            steps=10,
        )
    )


def compose(files: dict[str, str] | None = None, **kwargs: Any) -> Config:
    config_dir = Path.cwd() / "configs"
    config_dir.mkdir(exist_ok=True)
    write_files(config_dir, files or {})
    return nematocyte.compose(Config, config_dir, **kwargs)


PLAIN = {"presets/plain.yaml": "opt:\n  _replace_: true\n  lr: 0.001\n"}


# -- in a config file ------------------------------------------------------------


def test_replace_drops_the_default_of_the_field() -> None:
    assert compose(PLAIN, overrides=["+presets/plain"]).opt == {"lr": 0.001}


def test_without_replace_the_mapping_is_merged() -> None:
    files = {"presets/merged.yaml": "opt:\n  lr: 0.001\n"}
    config = compose(files, overrides=["+presets/merged"])
    assert config.opt == {"lr": 0.001, "wd": 0.01}


def test_replace_drops_what_a_primary_config_set() -> None:
    files = {**PLAIN, "run.yaml": "opt:\n  lr: 0.5\n  wd: 0.5\n"}
    config = compose(files, config_name="run", overrides=["+presets/plain"])
    assert config.opt == {"lr": 0.001}


def test_where_replace_is_written_in_the_mapping_does_not_matter() -> None:
    files = {"presets/last.yaml": "opt:\n  lr: 0.001\n  _replace_: true\n"}
    assert compose(files, overrides=["+presets/last"]).opt == {"lr": 0.001}


def test_replace_in_a_json5_file() -> None:
    files = {"presets/plain.json5": "{opt: {_replace_: true, lr: 0.001}}"}
    assert compose(files, overrides=["+presets/plain"]).opt == {"lr": 0.001}


def test_a_required_key_has_to_be_set_again() -> None:
    files = {"presets/empty.yaml": "opt:\n  _replace_: true\n"}
    with pytest.raises(ConfigError) as info:
        compose(files, overrides=["+presets/empty"])
    message = str(info.value)
    assert "opt.lr" in message
    assert "a value is required" in message


def test_replace_at_the_top_of_a_preset_for_the_field() -> None:
    files = {"opt/plain.yaml": "_replace_: true\nlr: 0.001\n"}
    assert compose(files, overrides=["opt=plain"]).opt == {"lr": 0.001}


def test_replace_drops_what_the_chain_of_self_set() -> None:
    files = {
        "opt/base.yaml": "lr: 0.5\nwd: 0.5\n",
        "opt/plain.yaml": "_defaults_:\n  _self_: base\n_replace_: true\nlr: 0.001\n",
    }
    assert compose(files, overrides=["opt=plain"]).opt == {"lr": 0.001}


def test_replace_keeps_what_the_file_selects() -> None:
    # The file selects `adam` under `_defaults_`, and replaces the TypedDict
    # that holds it; the selection is applied after `_replace_`.
    files = {
        "presets/adam.yaml": (
            "_defaults_:\n  training.optimizer: adam\n"
            "training:\n  _replace_: true\n  optimizer:\n    beta1: 0.5\n"
        )
    }
    config = compose(files, overrides=["+presets/adam"])
    # `steps` and the width from the default are gone; the dataclass inside
    # starts from its class defaults.
    assert config.training == {
        "optimizer": {"_option_": "adam", "beta1": 0.5},
        "model": Model(),
    }


def test_after_replace_a_group_inside_has_no_option() -> None:
    files = {"presets/bare.yaml": "training:\n  _replace_: true\n"}
    with pytest.raises(ConfigError, match="training.optimizer: no option is selected"):
        compose(files, overrides=["+presets/bare"])


def test_replace_in_the_option_of_a_group_keeps_the_option() -> None:
    files = {"presets/adam.yaml": "optimizer:\n  _replace_: true\n  beta1: 0.5\n"}
    config = compose(files, overrides=["+presets/adam"])
    assert config.optimizer == {"_option_": "adam", "beta1": 0.5}


def test_replace_switches_on_an_optional_subconfig() -> None:
    files = {"presets/ema.yaml": "ema:\n  _replace_: true\n  lr: 0.5\n"}
    assert compose(files, overrides=["+presets/ema"]).ema == {"lr": 0.5}


def test_replace_in_a_nested_mapping() -> None:
    files = {
        "run.yaml": "training:\n  optimizer:\n    momentum: 0.5\n",
        "presets/nested.yaml": (
            "training:\n  optimizer:\n    _replace_: true\n    momentum: 0.1\n"
        ),
    }
    config = compose(files, config_name="run", overrides=["+presets/nested"])
    assert config.training["optimizer"] == {"_option_": "sgd", "momentum": 0.1}
    assert config.training.get("steps") == 10


def test_a_later_layer_sets_values_again() -> None:
    files = {**PLAIN, "presets/wd.yaml": "opt:\n  wd: 0.2\n"}
    config = compose(files, overrides=["+presets/plain", "+presets/wd"])
    assert config.opt == {"lr": 0.001, "wd": 0.2}


def test_an_interpolation_to_a_replaced_key() -> None:
    with pytest.raises(ConfigError, match="'opt.wd', which has no value"):
        compose(PLAIN, overrides=["+presets/plain", "seed=${opt.wd}"])


def test_replace_inside_a_dict_value_is_a_key_of_the_value() -> None:
    @dataclass
    class WithKwargs:
        kwargs: dict[str, Any] = field(default_factory=dict)

    config_dir = write_files(
        Path.cwd() / "configs",
        {"presets/kwargs.yaml": "kwargs:\n  _replace_: true\n"},
    )
    config = nematocyte.compose(WithKwargs, config_dir, overrides=["+presets/kwargs"])
    assert config.kwargs == {"_replace_": True}


# -- the dropped-value warning ----------------------------------------------------


DROP_FILES = {
    "run.yaml": "optimizer:\n  eps: 1e-6\n",
    "presets/adam.yaml": "_defaults_:\n  optimizer: adam\noptimizer:\n  beta1: 0.8\n",
    "presets/plain.yaml": "optimizer:\n  _replace_: true\n  beta1: 0.7\n",
}


def test_a_selection_warns_about_a_dropped_value() -> None:
    with pytest.warns(DroppedValueWarning, match="optimizer.eps"):
        compose(DROP_FILES, config_name="run", overrides=["+presets/adam"])


def test_replace_forgets_what_a_selection_below_dropped() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DroppedValueWarning)
        config = compose(
            DROP_FILES,
            config_name="run",
            overrides=["+presets/adam", "+presets/plain"],
        )
    assert config.optimizer == {"_option_": "adam", "beta1": 0.7}


# -- on the command line ----------------------------------------------------------


def test_replace_in_a_dict_on_the_command_line() -> None:
    files = {"run.yaml": "opt:\n  wd: 0.5\n"}
    config = compose(
        files, config_name="run", overrides=["opt={_replace_:true,lr:0.001}"]
    )
    assert config.opt == {"lr": 0.001}


@pytest.mark.parametrize(
    "overrides",
    [
        ["opt.wd=0.2", "opt={_replace_:true,lr:0.001}"],
        ["opt={_replace_:true,lr:0.001}", "opt.wd=0.2"],
    ],
)
def test_replace_keeps_values_from_the_command_line(overrides: list[str]) -> None:
    assert compose(overrides=overrides).opt == {"lr": 0.001, "wd": 0.2}


@pytest.mark.parametrize(
    "overrides",
    [
        ["opt=fast", "opt={_replace_:true}"],
        ["opt={_replace_:true}", "opt=fast"],
    ],
)
def test_replace_keeps_a_selection_from_the_command_line(
    overrides: list[str],
) -> None:
    files = {"opt/fast.yaml": "lr: 1.0\n"}
    assert compose(files, overrides=overrides).opt == {"lr": 1.0}


def test_replace_on_the_command_line_keeps_a_selection_inside() -> None:
    files = {"run.yaml": "training:\n  steps: 3\n"}
    config = compose(
        files,
        config_name="run",
        overrides=[
            "training={_replace_:true,optimizer:{beta1:0.5}}",
            "training.optimizer=adam",
        ],
    )
    assert config.training == {
        "optimizer": {"_option_": "adam", "beta1": 0.5},
        "model": Model(),
    }


def test_replace_switches_on_an_optional_subconfig_from_the_command_line() -> None:
    config = compose(overrides=["ema={_replace_:true,lr:0.5}"])
    assert config.ema == {"lr": 0.5}


def test_multirun_keeps_the_params_of_the_sweeper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_files(
        tmp_path,
        {
            "nematocyte/sweeper/wds.yaml": (
                "_defaults_:\n  _self_: basic\n\nparams:\n  opt.wd: 0.1,0.2\n"
            )
        },
    )
    seen: list[Config] = []

    @nematocyte.main(Config, config_path=tmp_path)
    def main(config: Config) -> None:
        seen.append(config)

    main(["-m", "nematocyte.sweeper=wds", "opt={_replace_:true,lr:0.5}"])
    capsys.readouterr()
    assert [config.opt for config in seen] == [
        {"lr": 0.5, "wd": 0.1},
        {"lr": 0.5, "wd": 0.2},
    ]


def test_validate_configs_with_replace(tmp_path: Path) -> None:
    write_files(
        tmp_path,
        {
            **PLAIN,
            "opt/plain.yaml": "_replace_: true\nlr: 0.001\n",
            "opt/empty.yaml": "_replace_: true\n",
        },
    )
    results = {
        result.path.as_posix(): result.ok
        for result in nematocyte.validate_configs(Config, tmp_path)
    }
    assert results == {
        "presets/plain.yaml": True,
        "opt/plain.yaml": True,
        "opt/empty.yaml": False,
    }


# -- errors ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "overrides", "parts"),
    [
        (
            {"presets/model.yaml": "model:\n  _replace_: true\n"},
            ["+presets/model"],
            ["'_replace_'", "'model'", "Model", "dataclass", "presets/model.yaml"],
        ),
        (
            {"model/small.yaml": "_replace_: true\nwidth: 1\n"},
            ["model=small"],
            ["'_replace_'", "'model'", "Model", "dataclass", "model/small.yaml"],
        ),
        (
            {},
            ["model={_replace_:true}"],
            ["'_replace_'", "'model'", "Model", "dataclass", "model={_replace_:true}"],
        ),
        (
            {"presets/top.yaml": "_replace_: true\n"},
            ["+presets/top"],
            ["'_replace_'", "top level", "Config", "presets/top.yaml"],
        ),
        (
            {"presets/false.yaml": "opt:\n  _replace_: false\n"},
            ["+presets/false"],
            ["'_replace_'", "'opt'", "false", "Opt", "leave out the key"],
        ),
        (
            {"presets/string.yaml": "opt:\n  _replace_: yes\n"},
            ["+presets/string"],
            ["'_replace_'", "'opt'", "yes", "Opt"],
        ),
        (
            {},
            ["opt._replace_=true"],
            ["'_replace_'", "Opt", ": opt={_replace_:true,...}"],
        ),
        (
            {},
            ["ema={_replace_:1}"],
            ["'_replace_'", "'ema'", "1", "Opt"],
        ),
    ],
)
def test_errors_name_the_key_the_file_and_the_class(
    files: dict[str, str], overrides: list[str], parts: list[str]
) -> None:
    with pytest.raises(ConfigError) as info:
        compose(files, overrides=overrides)
    message = str(info.value)
    for part in parts:
        assert part in message


def test_replace_is_reserved() -> None:
    class WithReplace(TypedDict):
        _replace_: bool

    @dataclass
    class DataclassWithReplace:
        _replace_: bool = False

    for cls in (WithReplace, DataclassWithReplace):
        with pytest.raises(SchemaError, match="'_replace_' is reserved"):
            class_fields(cls)
