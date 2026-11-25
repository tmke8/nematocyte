"""Merge or replace, decided per key.

A key declared in a config class is merged; a key that is not declared, such
as every key of a `dict[...]`, is replaced by the layer that sets it. On the
command line, a dict replaces too.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import pytest

import nematocyte
from nematocyte import ConfigError

from .helpers import write_files


def _sched() -> dict[str, Any]:
    return {"sched": {"gamma": 0.1, "step": 5}, "milestones": [1, 2]}


@dataclass(kw_only=True)
class Opt:
    lr: float = 0.1
    kwargs: dict[str, Any] = field(default_factory=_sched)


class Extra(TypedDict):
    kwargs: NotRequired[dict[str, Any]]
    name: NotRequired[str]


@dataclass
class Point:
    x: int = 0
    y: int = 0


@dataclass(kw_only=True)
class Config:
    opt: Opt = field(default_factory=Opt)
    anything: Any = field(default_factory=lambda: {"a": {"b": 1, "c": 2}})
    maybe: dict[str, int] | None = None
    text: dict[str, int] | str = "none"
    extra: Extra = field(default_factory=lambda: Extra(kwargs={"a": 1}, name="x"))
    points: dict[str, Point] = field(default_factory=lambda: {"p": Point(1, 2)})


def compose(files: dict[str, str] | None = None, **kwargs: Any) -> Config:
    config_dir = Path.cwd() / "configs"
    config_dir.mkdir(exist_ok=True)
    write_files(config_dir, files or {})
    return nematocyte.compose(Config, config_dir, **kwargs)


# -- what is replaced ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Hydra merges: {sched: {gamma: 0.5, step: 5}, milestones: [1, 2], extra: 1}
        (
            "opt:\n  kwargs:\n    sched:\n      gamma: 0.5\n    extra: 1\n",
            {"sched": {"gamma": 0.5}, "extra": 1},
        ),
        # Hydra: {sched: {gamma: 0.1, step: 5}, milestones: [9]}
        ("opt:\n  kwargs:\n    milestones: [9]\n", {"milestones": [9]}),
        # Hydra: {sched: null, milestones: [1, 2]}
        ("opt:\n  kwargs:\n    sched: null\n", {"sched": None}),
        # An empty mapping makes the dict empty.
        ("opt:\n  kwargs: {}\n", {}),
    ],
)
def test_a_dict_in_a_file_replaces_the_default(text: str, expected: Any) -> None:
    files = {"presets/x.yaml": text}
    config = compose(files, overrides=["+presets/x"])
    assert config.opt.kwargs == expected
    # `opt` is a dataclass, whose fields are declared: it is merged.
    assert config.opt.lr == 0.1


def test_a_dict_is_replaced_in_the_order_of_the_layers() -> None:
    files = {
        "run.yaml": "opt:\n  kwargs:\n    sched:\n      gamma: 0.5\n",
        "presets/step.yaml": "opt:\n  kwargs:\n    sched:\n      step: 9\n",
    }
    config = compose(files, config_name="run", overrides=["+presets/step"])
    assert config.opt.kwargs == {"sched": {"step": 9}}


def test_a_field_typed_any_is_replaced() -> None:
    files = {"presets/x.yaml": "anything:\n  a:\n    b: 5\n"}
    config = compose(files, overrides=["+presets/x"])
    assert config.anything == {"a": {"b": 5}}


def test_a_mapping_over_something_else_replaces_it() -> None:
    files = {"presets/x.yaml": "maybe:\n  a: 1\ntext:\n  a: 1\n"}
    config = compose(files, overrides=["+presets/x"])
    assert config.maybe == {"a": 1}
    assert config.text == {"a": 1}


def test_a_mapping_over_an_interpolation_replaces_it() -> None:
    files = {"run.yaml": "anything: ${opt.kwargs}\n"}
    config = compose(files, config_name="run", overrides=["anything={b:1}"])
    assert config.anything == {"b": 1}


def test_a_typeddict_is_merged_and_a_dict_in_it_replaced() -> None:
    files = {"presets/x.yaml": "extra:\n  kwargs:\n    b: 2\n"}
    assert compose(files, overrides=["+presets/x"]).extra == {
        "kwargs": {"b": 2},
        "name": "x",
    }


def test_a_config_class_inside_a_dict_is_replaced_with_it() -> None:
    files = {"presets/x.yaml": "points:\n  p:\n    x: 5\n"}
    config = compose(files, overrides=["+presets/x"])
    # `y` falls back to the class default, not to the default of the field.
    assert config.points == {"p": Point(5, 0)}


# -- the command line ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (["opt.kwargs={sched:{gamma:0.5}}"], {"sched": {"gamma": 0.5}}),
        (["opt.kwargs={sched:{gamma:0.5}}", "opt.kwargs={b:1}"], {"b": 1}),
    ],
)
def test_a_dict_on_the_command_line_replaces_the_value(
    overrides: list[str], expected: dict[str, Any]
) -> None:
    files = {"run.yaml": "opt:\n  kwargs:\n    extra: 1\n"}
    config = compose(files, config_name="run", overrides=overrides)
    assert config.opt.kwargs == expected


def test_a_dict_on_the_command_line_replaces_a_value_of_any_type() -> None:
    assert compose(overrides=["anything={a:{b:5}}"]).anything == {"a": {"b": 5}}
    assert compose(overrides=["maybe={a:1}", "maybe={b:2}"]).maybe == {"b": 2}


def test_a_dotted_path_into_a_dict_points_to_a_mapping() -> None:
    with pytest.raises(ConfigError) as info:
        compose(overrides=["opt.kwargs.sched.gamma=0.5"])
    message = str(info.value)
    assert "'opt.kwargs'" in message
    assert "opt.kwargs={sched:{gamma:0.5}}" in message


def test_the_command_line_replaces_the_params_of_a_sweeper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_files(
        tmp_path,
        {
            "nematocyte/sweeper/kw.yaml": (
                "_defaults_:\n  _self_: basic\n\nparams:\n  opt.kwargs: '{a:1},{a:2}'\n"
            )
        },
    )
    seen: list[Config] = []

    @nematocyte.main(Config, config_path=tmp_path)
    def main(config: Config) -> None:
        seen.append(config)

    main(["-m", "nematocyte.sweeper=kw", "opt.kwargs={b:1}"])
    capsys.readouterr()
    assert [config.opt.kwargs for config in seen] == [{"b": 1}]


# -- _replace_ in a value ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "without", "expected"),
    [
        (
            "opt:\n  kwargs:\n    _replace_: true\n    a: 1\n",
            "opt:\n  kwargs:\n    a: 1\n",
            {"a": 1},
        ),
        (
            "opt:\n  kwargs:\n    sched:\n      _replace_: true\n      step: 1\n",
            "opt:\n  kwargs:\n    sched:\n      step: 1\n",
            {"sched": {"step": 1}},
        ),
        (
            "opt:\n  kwargs:\n    sched:\n      _replace_: true\n",
            "opt:\n  kwargs:\n    sched: {}\n",
            {"sched": {}},
        ),
    ],
)
def test_replace_in_a_value_changes_nothing(
    text: str, without: str, expected: dict[str, Any]
) -> None:
    files = {"presets/x.yaml": text, "presets/y.yaml": without}
    assert compose(files, overrides=["+presets/x"]).opt.kwargs == expected
    assert compose(files, overrides=["+presets/y"]).opt.kwargs == expected


def test_the_values_of_a_file_replace_a_dict_that_it_selects() -> None:
    files = {
        "run.yaml": "opt:\n  kwargs:\n    old: 1\n",
        "opt/heavy.yaml": "kwargs:\n  heavy: 1\n",
        "presets/x.yaml": "_defaults_:\n  opt: heavy\nopt:\n  kwargs:\n    x: 1\n",
    }
    config = compose(files, config_name="run", overrides=["+presets/x"])
    assert config.opt.kwargs == {"x": 1}


@pytest.mark.parametrize(
    "overrides",
    [["extra.kwargs={b:2}", "extra={}"], ["extra={}", "extra.kwargs={b:2}"]],
)
def test_a_dict_for_a_typeddict_drops_its_default(overrides: list[str]) -> None:
    # The dict replaces `extra`, and the dotted key is applied after it.
    assert compose(overrides=overrides).extra == {"kwargs": {"b": 2}}


def test_replace_is_not_true_in_a_value() -> None:
    files = {"presets/x.yaml": "opt:\n  kwargs:\n    sched:\n      _replace_: 1\n"}
    with pytest.raises(ConfigError) as info:
        compose(files, overrides=["+presets/x"])
    message = str(info.value)
    assert "'opt.kwargs.sched'" in message
    assert "presets/x.yaml" in message
    assert "'_replace_'" in message


def test_replace_inside_a_list_is_a_key_of_an_item() -> None:
    @dataclass
    class WithList:
        items: list[dict[str, Any]] = field(default_factory=list)

    config_dir = write_files(
        Path.cwd() / "configs", {"presets/x.yaml": "items:\n  - _replace_: 1\n"}
    )
    config = nematocyte.compose(WithList, config_dir, overrides=["+presets/x"])
    assert config.items == [{"_replace_": 1}]


def test_the_composed_value_is_saved_whole(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Building the config again from the saved one does not merge the class
    # default back in.
    seen: list[tuple[Config, dict[str, Any]]] = []

    @nematocyte.main(Config, config_path=tmp_path)
    def main(config: Config) -> None:
        seen.append((config, cast("dict[str, Any]", nematocyte.run_config())))

    main(["opt.kwargs={a:1}"])
    capsys.readouterr()
    [(config, saved)] = seen
    assert config.opt.kwargs == {"a": 1}
    assert nematocyte.from_run_config(Config, saved) == config
