"""A mapping is merged into a mapping, also in a value such as a `dict[...]`."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast

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


class Extra(TypedDict, total=False):
    kwargs: dict[str, Any]


@dataclass(kw_only=True)
class Config:
    opt: Opt = field(default_factory=Opt)
    anything: Any = field(default_factory=lambda: {"a": {"b": 1, "c": 2}})
    maybe: dict[str, int] | None = None
    text: dict[str, int] | str = "none"
    extra: Extra = field(default_factory=lambda: Extra(kwargs={"a": 1}))


def compose(files: dict[str, str] | None = None, **kwargs: Any) -> Config:
    config_dir = Path.cwd() / "configs"
    config_dir.mkdir(exist_ok=True)
    write_files(config_dir, files or {})
    return nematocyte.compose(Config, config_dir, **kwargs)


# -- what is merged ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Nested mappings are merged too, as in Hydra.
        (
            "opt:\n  kwargs:\n    sched:\n      gamma: 0.5\n    extra: 1\n",
            {"sched": {"gamma": 0.5, "step": 5}, "milestones": [1, 2], "extra": 1},
        ),
        # A list is replaced.
        (
            "opt:\n  kwargs:\n    milestones: [9]\n",
            {"sched": {"gamma": 0.1, "step": 5}, "milestones": [9]},
        ),
        # A value that is not a mapping replaces the mapping, and the other way.
        (
            "opt:\n  kwargs:\n    sched: off\n",
            {"sched": "off", "milestones": [1, 2]},
        ),
        # Null is a value, which replaces.
        (
            "opt:\n  kwargs:\n    sched: null\n",
            {"sched": None, "milestones": [1, 2]},
        ),
    ],
)
def test_a_file_is_merged_into_the_default(text: str, expected: Any) -> None:
    files = {"presets/x.yaml": text}
    assert compose(files, overrides=["+presets/x"]).opt.kwargs == expected


def test_files_are_merged_in_the_order_of_the_layers() -> None:
    files = {
        "run.yaml": "opt:\n  kwargs:\n    sched:\n      gamma: 0.5\n",
        "presets/step.yaml": "opt:\n  kwargs:\n    sched:\n      step: 9\n",
    }
    config = compose(files, config_name="run", overrides=["+presets/step"])
    assert config.opt.kwargs["sched"] == {"gamma": 0.5, "step": 9}


def test_a_field_typed_any_is_merged() -> None:
    config = compose(overrides=["anything={a:{b:5}}"])
    assert config.anything == {"a": {"b": 5, "c": 2}}


def test_a_mapping_over_something_else_replaces_it() -> None:
    assert compose(overrides=["maybe={a:1}"]).maybe == {"a": 1}
    assert compose(overrides=["text={a:1}"]).text == {"a": 1}
    config = compose(overrides=["maybe={a:1}", "maybe={b:2}"])
    assert config.maybe == {"a": 1, "b": 2}


def test_a_mapping_over_an_interpolation_replaces_it() -> None:
    files = {"run.yaml": "anything: ${opt.kwargs}\n"}
    config = compose(files, config_name="run", overrides=["anything={b:1}"])
    assert config.anything == {"b": 1}


def test_a_dict_inside_a_typeddict_is_merged() -> None:
    files = {"presets/x.yaml": "extra:\n  kwargs:\n    b: 2\n"}
    assert compose(files, overrides=["+presets/x"]).extra == {
        "kwargs": {"a": 1, "b": 2}
    }


# -- the command line ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (["opt.kwargs={sched:{gamma:0.5}}"], {"gamma": 0.5, "step": 5}),
        (
            ["opt.kwargs={sched:{gamma:0.5}}", "opt.kwargs={sched:{step:1}}"],
            {"gamma": 0.5, "step": 1},
        ),
        (["opt={kwargs:{sched:{step:1}}}"], {"gamma": 0.1, "step": 1}),
    ],
)
def test_a_dict_on_the_command_line_is_merged(
    overrides: list[str], expected: dict[str, float]
) -> None:
    assert compose(overrides=overrides).opt.kwargs["sched"] == expected


def test_a_dotted_path_into_a_dict_points_to_a_mapping() -> None:
    with pytest.raises(ConfigError) as info:
        compose(overrides=["opt.kwargs.sched.gamma=0.5"])
    message = str(info.value)
    assert "'opt.kwargs'" in message
    assert "opt.kwargs={sched:{gamma:0.5}}" in message


def test_the_params_of_a_sweeper_are_merged_with_the_command_line(
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

    main(["-m", "nematocyte.sweeper=kw", "opt.kwargs={_replace_:true,b:1}"])
    capsys.readouterr()
    assert [config.opt.kwargs for config in seen] == [
        {"a": 1, "b": 1},
        {"a": 2, "b": 1},
    ]


# -- _replace_ in a value ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("opt:\n  kwargs:\n    _replace_: true\n    a: 1\n", {"a": 1}),
        (
            "opt:\n  kwargs:\n    sched:\n      _replace_: true\n      step: 1\n",
            {"sched": {"step": 1}, "milestones": [1, 2]},
        ),
        (
            "opt:\n  kwargs:\n    sched:\n      _replace_: true\n",
            {"sched": {}, "milestones": [1, 2]},
        ),
    ],
)
def test_replace_in_a_file(text: str, expected: dict[str, Any]) -> None:
    files = {"presets/x.yaml": text}
    assert compose(files, overrides=["+presets/x"]).opt.kwargs == expected


def test_replace_in_a_file_keeps_what_the_file_selects() -> None:
    files = {
        "run.yaml": "opt:\n  kwargs:\n    old: 1\n",
        "opt/heavy.yaml": "kwargs:\n  heavy: 1\n",
        "presets/x.yaml": (
            "_defaults_:\n  opt: heavy\nopt:\n  kwargs:\n    _replace_: true\n    x: 1\n"
        ),
    }
    config = compose(files, config_name="run", overrides=["+presets/x"])
    assert config.opt.kwargs == {"heavy": 1, "x": 1}


@pytest.mark.parametrize(
    "overrides",
    [
        ["opt.kwargs={a:1}", "opt.kwargs={_replace_:true,b:2}"],
        ["opt.kwargs={_replace_:true,b:2}", "opt.kwargs={a:1}"],
        ["opt={kwargs:{a:1}}", "opt.kwargs={_replace_:true,b:2}"],
    ],
)
def test_replace_on_the_command_line_keeps_the_command_line(
    overrides: list[str],
) -> None:
    files = {"run.yaml": "opt:\n  kwargs:\n    old: 1\n"}
    config = compose(files, config_name="run", overrides=overrides)
    assert config.opt.kwargs == {"a": 1, "b": 2}


def test_replace_of_a_typeddict_drops_what_was_merged_from_below() -> None:
    # `extra.kwargs` is set on the command line, but merged with the default;
    # replacing `extra` drops the part that the default gave.
    config = compose(overrides=["extra.kwargs={b:2}", "extra={_replace_:true}"])
    assert config.extra == {"kwargs": {"b": 2}}


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

    main(["opt.kwargs={_replace_:true,a:1}"])
    capsys.readouterr()
    [(config, saved)] = seen
    assert config.opt.kwargs == {"a": 1}
    assert nematocyte.from_run_config(Config, saved) == config
