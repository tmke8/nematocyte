"""`+key=value`: one key in a dict on the command line.

A key that no config class declares, such as a key of a `dict[...]`, is set
on its own with `+`, and the other keys of the dict stay. A declared key
takes no `+`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

from pydantic import Discriminator, Tag
import pytest

import nematocyte
from nematocyte import ConfigError

from .helpers import write_files


def _kwargs() -> dict[str, Any]:
    return {"eps": 1.0, "sched": {"gamma": 0.1}, "milestones": [1, 2]}


@dataclass(kw_only=True)
class Opt:
    lr: float = 0.1
    kwargs: dict[str, Any] = field(default_factory=_kwargs)
    maybe: dict[str, Any] | None = None
    counts: dict[str, int] = field(default_factory=dict)
    steps: int = 1


class Wandb(TypedDict):
    name: NotRequired[str]
    kwargs: NotRequired[dict[str, Any]]


@dataclass(kw_only=True)
class Sgd:
    momentum: float = 0.0
    kwargs: dict[str, Any] = field(default_factory=lambda: {"nesterov": False})


@dataclass(kw_only=True)
class Adam:
    kwargs: dict[str, Any] = field(default_factory=lambda: {"amsgrad": False})


type Optimizer = Annotated[
    Annotated[Sgd, Tag("sgd")] | Annotated[Adam, Tag("adam")],
    Discriminator(nematocyte.pick),
]


@dataclass(kw_only=True)
class Config:
    opt: Opt = field(default_factory=Opt)
    wandb: Wandb = field(default_factory=lambda: Wandb(name="run"))
    optimizer: Optimizer = field(default_factory=Sgd)
    anything: Any = field(default_factory=lambda: {"a": 1})
    mapping: Mapping[str, Any] = field(default_factory=lambda: {"a": 1})
    text: Any = "${opt.lr}"


def compose(*overrides: str, files: dict[str, str] | None = None) -> Config:
    config_dir = Path.cwd() / "configs"
    config_dir.mkdir(exist_ok=True)
    write_files(config_dir, files or {})
    return nematocyte.compose(Config, config_dir, overrides=overrides)


def error(*overrides: str, files: dict[str, str] | None = None) -> str:
    with pytest.raises(ConfigError) as excinfo:
        compose(*overrides, files=files)
    return str(excinfo.value)


# -- what it sets --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        # Hydra adds the key too.
        ("+opt.kwargs.new=2", {**_kwargs(), "new": 2}),
        # Hydra: "An item is already at 'opt.kwargs.eps'"; its `++` sets it.
        ("+opt.kwargs.eps=2", {**_kwargs(), "eps": 2}),
        # Hydra merges into the mapping, as here.
        ("+opt.kwargs.sched.step=5", {**_kwargs(), "sched": {"gamma": 0.1, "step": 5}}),
        # A dict replaces, as everywhere on the command line; Hydra merges.
        ("+opt.kwargs.sched={step:5}", {**_kwargs(), "sched": {"step": 5}}),
        ("+opt.kwargs.milestones=[3]", {**_kwargs(), "milestones": [3]}),
        ("+opt.kwargs.eps=null", {**_kwargs(), "eps": None}),
    ],
)
def test_sets_a_key_and_keeps_the_others(override: str, expected: Any) -> None:
    assert compose(override).opt.kwargs == expected


def test_the_default_is_not_changed() -> None:
    compose("+opt.kwargs.sched.step=5")
    assert compose().opt.kwargs == _kwargs()


def test_sets_a_key_in_a_dict_from_a_config_file() -> None:
    files = {"presets/x.yaml": "opt:\n  kwargs:\n    beta: 0.9\n"}
    config = compose("+presets/x", "+opt.kwargs.eps=2", files=files)
    assert config.opt.kwargs == {"beta": 0.9, "eps": 2}


def test_sets_a_key_in_a_dict_from_a_named_config() -> None:
    files = {"opt/fast.yaml": "lr: 1.0\nkwargs:\n  beta: 0.9\n"}
    config = compose("+opt.kwargs.eps=2", "opt=fast", files=files)
    assert config.opt.lr == 1.0
    assert config.opt.kwargs == {"beta": 0.9, "eps": 2}


@pytest.mark.parametrize(
    "overrides",
    [
        ["optimizer=adam", "+optimizer.kwargs.amsgrad=true"],
        ["+optimizer.kwargs.amsgrad=true", "optimizer=adam"],
    ],
)
def test_applies_after_a_selection(overrides: list[str]) -> None:
    config = compose(*overrides)
    assert isinstance(config.optimizer, Adam)
    assert config.optimizer.kwargs == {"amsgrad": True}


@pytest.mark.parametrize(
    "overrides",
    [
        ["opt.kwargs={a:1}", "+opt.kwargs.b=2"],
        ["+opt.kwargs.b=2", "opt.kwargs={a:1}"],
    ],
)
def test_applies_after_a_dict_for_the_field(overrides: list[str]) -> None:
    # Hydra: "Could not override 'opt.kwargs'".
    assert compose(*overrides).opt.kwargs == {"a": 1, "b": 2}


@pytest.mark.parametrize(
    "overrides",
    [
        ["+opt.kwargs.sched={step:5}", "+opt.kwargs.sched.gamma=0.5"],
        ["+opt.kwargs.sched.gamma=0.5", "+opt.kwargs.sched={step:5}"],
    ],
)
def test_the_shorter_key_applies_first(overrides: list[str]) -> None:
    assert compose(*overrides).opt.kwargs["sched"] == {"step": 5, "gamma": 0.5}


def test_the_last_one_wins() -> None:
    # Hydra: "An item is already at 'opt.kwargs.y'".
    assert compose("+opt.kwargs.y=2", "+opt.kwargs.y=3").opt.kwargs["y"] == 3


def test_in_a_typeddict_and_in_any_and_mapping_fields() -> None:
    files = {"presets/x.yaml": "wandb:\n  kwargs:\n    a: 1\n"}
    config = compose(
        "+presets/x",
        "+wandb.kwargs.b=2",
        "+anything.b=2",
        "+mapping.b=2",
        files=files,
    )
    assert config.wandb == {"name": "run", "kwargs": {"a": 1, "b": 2}}
    assert config.anything == {"a": 1, "b": 2}
    assert config.mapping == {"a": 1, "b": 2}


def test_the_value_is_checked_with_the_dict() -> None:
    message = error("+opt.counts.x=abc")
    assert "opt.counts.x" in message
    assert "+opt.counts.x=abc" in message


def test_replace_in_the_value_is_an_error() -> None:
    message = error("+opt.kwargs.sched={_replace_:true,step:1}")
    assert "_replace_" in message
    assert "opt.kwargs.sched" in message


# -- what is not a key in a dict ---------------------------------------------------


@pytest.mark.parametrize(
    ("override", "key", "cls", "without"),
    [
        # Hydra: "An item is already at 'opt.lr'".
        ("+opt.lr=0.5", "opt.lr", "Opt", "opt.lr=0.5"),
        ("+opt.kwargs={a:1}", "opt.kwargs", "Opt", "opt.kwargs={a:1}"),
        ("+wandb.name=x", "wandb.name", "Wandb", "wandb.name=x"),
        (
            "+optimizer.momentum=0.5",
            "optimizer.momentum",
            "Sgd",
            "optimizer.momentum=0.5",
        ),
    ],
)
def test_a_declared_key_takes_no_plus(
    override: str, key: str, cls: str, without: str
) -> None:
    message = error(override)
    assert repr(override) in message
    assert repr(key) in message
    assert cls in message
    assert f"'+' is not needed: {without}" in message


@pytest.mark.parametrize(
    ("override", "key", "cls", "closed"),
    [
        # Hydra adds the field to the dataclass, without a warning.
        ("+opt.newfield=1", "newfield", "Opt", "dataclass"),
        ("+opt.kwarg.eps=1", "kwarg", "Opt", "dataclass"),
        ("+wandb.extra=1", "extra", "Wandb", "TypedDict"),
    ],
)
def test_a_class_has_only_its_declared_keys(
    override: str, key: str, cls: str, closed: str
) -> None:
    message = error(override)
    assert repr(override) in message
    assert repr(key) in message
    assert f"{cls} is a {closed}" in message


def test_plus_with_a_key_of_the_root_points_to_an_added_config() -> None:
    # Hydra's `+presets=debug`: the name of the file is written without `=`.
    message = error("+presets=debug")
    assert repr("+presets=debug") in message
    assert "'presets' is not a field of Config" in message
    assert "Config is a dataclass" in message
    assert "+presets/debug" in message


def test_plus_with_a_key_of_a_typeddict_root() -> None:
    class Root(TypedDict):
        seed: int

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root, overrides=["seed=1", "+presets=debug"])
    message = str(excinfo.value)
    assert "'presets' is not a field of Root" in message
    assert "Root is a TypedDict, and a TypedDict is closed" in message
    assert "+presets/debug" in message


@pytest.mark.parametrize(
    ("override", "without"), [("+opt=fast", "opt=fast"), ("+text=1", "text=1")]
)
def test_plus_with_a_declared_key_of_the_root(override: str, without: str) -> None:
    # Declared: the key is set without `+`, and it is not a primary config.
    message = error(override)
    assert "a field of the dataclass Config" in message
    assert f"'+' is not needed: {without}" in message
    assert "primary config" not in message


def test_an_unknown_field_suggests_the_closest() -> None:
    message = error("+opt.kwarg.eps=1")
    assert "Did you mean 'kwargs'?" in message
    # Only a key of the root can have been meant as an added config.
    assert "primary config" not in message


# -- the dict has to be there ------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "field", "holds", "whole"),
    [
        # Hydra: "Error merging override".
        ("+opt.maybe.eps=1", "Opt (at 'opt')", "is null", "opt.maybe={eps:1}"),
        (
            "+wandb.kwargs.eps=1",
            "Wandb (at 'wandb')",
            "is not set",
            "wandb.kwargs={eps:1}",
        ),
        ("+opt.steps.x=1", "Opt (at 'opt')", "holds 1, not a dict", "opt.steps={x:1}"),
        ("+text.x=1", "Config", "holds '${opt.lr}', not a dict", "text={x:1}"),
        ("+opt.maybe.a.b=1", "Opt (at 'opt')", "is null", "opt.maybe={a:{b:1}}"),
    ],
)
def test_a_field_without_a_dict_is_an_error(
    override: str, field: str, holds: str, whole: str
) -> None:
    message = error(override)
    key = whole.partition("=")[0]
    assert repr(override) in message
    assert f"{key!r} of {field} {holds}" in message
    assert f"To set the whole dict: {whole}" in message


@pytest.mark.parametrize(
    ("override", "here", "holds", "whole"),
    [
        # Hydra creates the mapping.
        ("+opt.kwargs.a.b=5", "opt.kwargs.a", "is not set", "+opt.kwargs.a={b:5}"),
        (
            "+opt.kwargs.a.b.c=5",
            "opt.kwargs.a",
            "is not set",
            "+opt.kwargs.a={b:{c:5}}",
        ),
        # Hydra replaces the number with a mapping.
        ("+opt.kwargs.eps.x=1", "opt.kwargs.eps", "holds 1.0", "+opt.kwargs.eps={x:1}"),
        (
            "+opt.kwargs.sched.gamma.x=1",
            "opt.kwargs.sched.gamma",
            "holds 0.1",
            "+opt.kwargs.sched.gamma={x:1}",
        ),
    ],
)
def test_every_mapping_on_the_way_has_to_be_there(
    override: str, here: str, holds: str, whole: str
) -> None:
    message = error(override)
    assert repr(override) in message
    assert f"{here!r}, in the dict 'opt.kwargs' of Opt (at 'opt'), {holds}" in message
    assert f"To set it as a whole: {whole}" in message


# -- the error for a dotted key into a value -----------------------------------------


def test_a_dotted_key_into_a_dict_points_to_plus() -> None:
    message = error("opt.kwargs.eps=1e-6")
    assert "opt.kwargs={eps:1e-06}" in message
    assert "+opt.kwargs.eps=1e-06" in message


def test_a_dotted_key_into_a_number_does_not_point_to_plus() -> None:
    message = error("opt.steps.x=1")
    assert "opt.steps={x:1}" in message
    assert "+opt" not in message
