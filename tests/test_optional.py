"""Optional subconfigs, typed `X | None` (`design/unions.md`)."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Annotated, Any, NotRequired, Optional, TypedDict, cast
import warnings

from pydantic import Discriminator, Tag
import pytest

import nematocyte
from nematocyte import ConfigError, SchemaError
from nematocyte._schema import Kind, class_fields

from .helpers import empty_config_dir, write_files
from .test_help import cfg_text, help_text


@dataclass
class Inner:
    size: int = 1


@dataclass
class EmaCfg:
    """Exponential moving average of the weights."""

    decay: float = 0.99
    every: int = 1
    inner: Inner = field(default_factory=Inner)


@dataclass
class Plain:
    lr: float = 0.1
    momentum: float = 0.0
    kwargs: dict[str, int] = field(default_factory=dict)


@dataclass
class A:
    x: int = 0


@dataclass
class B:
    y: int = 0


type Group = Annotated[
    Annotated[A, Tag("a")] | Annotated[B, Tag("b")], Discriminator(nematocyte.pick)
]


@dataclass
class Config:
    ema: EmaCfg | None = field(default=None, doc="Off unless switched on.")
    on: EmaCfg | None = field(default_factory=lambda: EmaCfg(decay=0.8))
    plain: Plain = field(default_factory=Plain)
    group: Group = field(default_factory=A)
    seed: int = 0


def compose(
    config_dir: Path | None = None,
    overrides: tuple[str, ...] = (),
    config_name: str | None = None,
) -> Config:
    return nematocyte.compose(
        Config, config_dir, config_name=config_name, overrides=overrides
    )


def configs(tmp_path: Path, files: dict[str, str]) -> Path:
    return write_files(tmp_path / "configs", files)


# -- the type ---------------------------------------------------------------


def test_x_or_none_is_an_optional_subconfig() -> None:
    @dataclass
    class C:
        union: EmaCfg | None = None
        optional: Optional[EmaCfg] = None  # noqa: UP045

    for info in class_fields(C).values():
        assert info.kind is Kind.CONFIG
        assert info.config_cls is EmaCfg
        assert info.optional


def test_plain_subconfig_is_not_optional() -> None:
    assert not class_fields(Config)["plain"].optional


def test_typeddict_can_be_an_optional_subconfig() -> None:
    class Td(TypedDict):
        a: int

    @dataclass
    class C:
        td: Td | None = None

    assert compose_with(C, ("td={a:3}",)) == C(td={"a": 3})
    assert compose_with(C) == C(td=None)


def compose_with[T](cls: type[T], overrides: tuple[str, ...] = ()) -> T:
    return nematocyte.compose(cls, overrides=overrides)


class InnerTd(TypedDict):
    size: NotRequired[int]


class EmaTd(TypedDict):
    decay: NotRequired[float]
    every: NotRequired[int]
    inner: InnerTd


@dataclass
class TdConfig:
    ema: EmaTd | None = None
    on: EmaTd | None = field(default_factory=lambda: EmaTd(decay=0.8, inner={}))


def compose_td(
    overrides: tuple[str, ...] = (), config_dir: Path | None = None
) -> TdConfig:
    return nematocyte.compose(TdConfig, config_dir, overrides=overrides)


def test_none_next_to_a_group_with_one_option_is_an_error() -> None:
    @dataclass
    class C:
        data: Annotated[A, Tag("a")] | None = None

    with pytest.raises(SchemaError, match=r"C\.data.*one option.*A \| None"):
        class_fields(C)


def test_not_required_subconfig_points_to_x_or_none() -> None:
    class Td(TypedDict):
        ema: NotRequired[EmaCfg]

    with pytest.raises(SchemaError, match=r"Td\.ema.*X \| None"):
        class_fields(Td)


def test_default_of_another_class_is_an_error() -> None:
    @dataclass
    class C:
        ema: EmaCfg | None = field(default_factory=cast("Any", Inner))

    with pytest.raises(SchemaError, match=r"C\.ema.*Inner.*EmaCfg"):
        compose_with(C)


# -- the default -------------------------------------------------------------


def test_default_none_is_off_and_default_instance_is_on() -> None:
    config = compose()
    assert config.ema is None
    assert config.on == EmaCfg(decay=0.8)


# -- null switches it off ------------------------------------------------------


def test_null_on_the_command_line_switches_it_off() -> None:
    assert compose(overrides=("on=null",)).on is None


def test_null_under_defaults_switches_it_off(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path,
        {"off.yaml": "_defaults_:\n  on: null\n", "set.yaml": "on:\n  every: 5\n"},
    )
    assert compose(config_dir, config_name="off").on is None
    # What a lower layer set under it goes.
    assert compose(config_dir, ("+off",), config_name="set").on is None


@pytest.mark.parametrize("text", ["on: null\n", "on:\n  # decay: 0.3\n"])
def test_null_among_the_values_points_to_defaults(tmp_path: Path, text: str) -> None:
    # YAML reads a key with only comments under it as null. Hydra switches
    # the subconfig off, so a silent "nothing" would surprise.
    config_dir = configs(tmp_path, {"c.yaml": text})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'on' is an optional subconfig of Config" in message
    assert "_defaults_:\n      on: null" in message
    assert "leave out the key" in message
    assert "_replace_" not in message


@pytest.mark.parametrize("text", ["plain: null\n", "plain:\n  # lr: 0.3\n"])
def test_null_for_a_plain_subconfig_in_a_file_is_an_error(
    tmp_path: Path, text: str
) -> None:
    # As in Hydra: "field 'plain' is not Optional".
    config_dir = configs(tmp_path, {"c.yaml": text})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'plain' is a subconfig of Config" in message
    assert "Plain | None" in message
    assert "nothing under it is null too" in message
    assert "leave out the key" in message


def test_null_for_a_typeddict_in_a_file_points_to_replace(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "ema:\n  # decay: 0.3\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(TdConfig, config_dir, config_name="c")
    message = str(excinfo.value)
    assert "'ema' is an optional subconfig of TdConfig" in message
    assert "_defaults_:\n      ema: null" in message
    assert "ema: {_replace_: true}" in message


@pytest.mark.parametrize("text", ["group: null\n", "group:\n  # x: 3\n"])
def test_null_for_a_group_in_a_file_is_an_error(tmp_path: Path, text: str) -> None:
    # As in Hydra, for a group field typed as a base class.
    config_dir = configs(tmp_path, {"c.yaml": text})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'group' is a group of Config" in message
    assert "Options: a, b" in message


@dataclass
class WithDicts:
    counts: dict[str, int] = field(default_factory=lambda: {"a": 1})
    maybe: dict[str, int] | None = field(default_factory=lambda: {"a": 1})
    anything: Any = field(default_factory=lambda: {"a": 1})


@pytest.mark.parametrize("text", ["counts: null\n", "counts:\n  # a: 2\n"])
def test_null_for_a_dict_in_a_file_is_an_error(tmp_path: Path, text: str) -> None:
    config_dir = configs(tmp_path, {"c.yaml": text})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(WithDicts, config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'counts' of WithDicts holds a dict" in message
    assert "counts: {_replace_: true}" in message
    assert "leave out the key" in message


def test_null_for_a_dict_that_takes_none_is_a_value(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "maybe: null\nanything:\n"})
    config = nematocyte.compose(WithDicts, config_dir, config_name="c")
    assert config.maybe is None
    assert config.anything is None


def test_an_empty_dict_is_written_with_replace(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "counts: {_replace_: true}\nmaybe: {}\n"})
    config = nematocyte.compose(WithDicts, config_dir, config_name="c")
    assert config.counts == {}
    # An empty mapping is merged, and changes nothing.
    assert config.maybe == {"a": 1}


def test_null_for_a_nested_subconfig_in_a_dict_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose_td(("ema={inner:null}",))
    message = str(excinfo.value)
    assert "ema={inner:null}" in message
    assert "'ema.inner' is a subconfig of EmaTd" in message
    assert "InnerTd | None" in message


def test_switching_off_drops_what_was_set_without_a_warning(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"set.yaml": "on:\n  every: 5\n"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = compose(config_dir, ("on=null",), config_name="set")
    assert config.on is None


def test_null_for_a_plain_subconfig_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("plain=null",))
    message = str(excinfo.value)
    assert "'plain=null'" in message
    assert "Plain | None" in message


def test_null_for_a_group_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("group=null",))
    message = str(excinfo.value)
    assert "'group=null'" in message
    assert "Discriminator(...)] | None" in message
    assert "Options: a, b" in message


@pytest.mark.parametrize("key", ["plain", "group"])
def test_null_under_defaults_for_a_field_that_is_not_optional_is_an_error(
    tmp_path: Path, key: str
) -> None:
    config_dir = configs(tmp_path, {"c.yaml": f"_defaults_:\n  {key}: null\n"})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert f"'{key}' is a" in message
    assert "cannot be null" in message


@pytest.mark.parametrize(
    "body", ["ema:\n  decay: 0.5\n", "ema: {}\n", "ema: null\n", "ema:\n"]
)
def test_values_after_null_under_defaults_are_an_error(
    tmp_path: Path, body: str
) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "_defaults_:\n  ema: null\n" + body})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'ema' is switched off in the entry 'ema' of '_defaults_'" in message
    assert "Remove one of the two" in message


def test_a_higher_file_switches_on_what_a_lower_one_switched_off(
    tmp_path: Path,
) -> None:
    config_dir = configs(
        tmp_path,
        {
            "off.yaml": "_defaults_:\n  ema: null\n",
            "on_again.yaml": (
                "_defaults_:\n  _self_: off\n  ema: _some_\nema:\n  every: 3\n"
            ),
        },
    )
    assert compose(config_dir, config_name="on_again").ema == EmaCfg(every=3)


# -- _some_ switches it on ------------------------------------------------------


def test_some_under_defaults_switches_it_on(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path,
        {
            "switch_on.yaml": "_defaults_:\n  ema: _some_\n",
            "with_values.yaml": "_defaults_:\n  ema: _some_\nema:\n  decay: 0.5\n",
        },
    )
    assert compose(config_dir, config_name="switch_on").ema == EmaCfg()
    config = compose(config_dir, config_name="with_values")
    assert config.ema == EmaCfg(decay=0.5)


def test_some_on_the_command_line_switches_it_on() -> None:
    assert compose(overrides=("ema=_some_",)).ema == EmaCfg()
    assert compose_td(("ema=_some_",)).ema == {"inner": {}}


@pytest.mark.parametrize(
    "overrides",
    [("ema=_some_", "ema.decay=0.5"), ("ema.decay=0.5", "ema=_some_")],
)
def test_some_applies_before_the_dotted_keys(overrides: tuple[str, ...]) -> None:
    assert compose(overrides=overrides).ema == EmaCfg(decay=0.5)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [(("ema=null", "ema=_some_"), EmaCfg()), (("ema=_some_", "ema=null"), None)],
)
def test_null_and_some_are_the_same_key(
    overrides: tuple[str, ...], expected: EmaCfg | None
) -> None:
    assert compose(overrides=overrides).ema == expected


def test_some_starts_from_the_class_defaults(tmp_path: Path) -> None:
    # Not from the default instance of the field, `EmaCfg(decay=0.8)`.
    config_dir = configs(tmp_path, {"off.yaml": "_defaults_:\n  on: null\n"})
    assert compose(config_dir, ("on=_some_",), config_name="off").on == EmaCfg()


def test_some_leaves_it_as_it_is_when_on(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path,
        {
            "low.yaml": "_defaults_:\n  ema: _some_\nema:\n  every: 4\n",
            "high.yaml": "_defaults_:\n  ema: _some_\n",
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = compose(config_dir, ("+high",), config_name="low")
        assert config.ema == EmaCfg(every=4)
        assert compose(overrides=("on=_some_",)).on == EmaCfg(decay=0.8)


def test_dotted_some_entry_switches_on_a_nested_subconfig(tmp_path: Path) -> None:
    @dataclass
    class Outer:
        ema: EmaCfg | None = None

    @dataclass
    class Nested:
        outer: Outer = field(default_factory=Outer)

    config_dir = configs(tmp_path, {"c.yaml": "_defaults_:\n  outer.ema: _some_\n"})
    config = nematocyte.compose(Nested, config_dir, config_name="c")
    assert config.outer.ema == EmaCfg()


def test_some_for_a_subconfig_that_is_not_optional_is_an_error(
    tmp_path: Path,
) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "_defaults_:\n  plain: _some_\n"})
    for overrides, config_name in [(("plain=_some_",), None), ((), "c")]:
        with pytest.raises(ConfigError) as excinfo:
            compose(config_dir, overrides, config_name=config_name)
        message = str(excinfo.value)
        assert "'plain' is a subconfig of Config" in message
        assert "Plain, not Plain | None" in message
    assert "c.yaml" in message


def test_some_for_a_group_is_an_error(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "_defaults_:\n  group: _some_\n"})
    for overrides, config_name in [(("group=_some_",), None), ((), "c")]:
        with pytest.raises(ConfigError) as excinfo:
            compose(config_dir, overrides, config_name=config_name)
        message = str(excinfo.value)
        assert "'group' is a group of Config" in message
        assert "group=a" in message
        assert "Options: a, b" in message


def test_some_as_self_of_a_preset_is_an_error(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"ema/fast.yaml": "_defaults_:\n  _self_: _some_\n"})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, ("ema=fast",))
    message = str(excinfo.value)
    assert "fast.yaml" in message
    assert "'_self_' cannot name it" in message
    assert "ema: _some_" in message


def test_some_cannot_be_the_name_of_an_option() -> None:
    @dataclass
    class C:
        group: Annotated[
            Annotated[A, Tag("_some_")] | Annotated[B, Tag("b")],
            Discriminator(nematocyte.pick),
        ] = field(default_factory=A)

    with pytest.raises(SchemaError, match="'_some_' is not a valid name"):
        nematocyte.compose(C)


def test_a_config_file_called_some_is_not_a_preset(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path, {"ema/_some_.yaml": "decay: 0.5\n", "ema/fast.yaml": "every: 2\n"}
    )
    # `_some_` switches it on; the file is not read.
    assert compose(config_dir, ("ema=_some_",)).ema == EmaCfg()
    assert "_some_" not in help_text(Config, config_dir)
    results = {
        r.path.as_posix(): r.error
        for r in nematocyte.validate_configs(Config, config_dir)
    }
    assert results["ema/fast.yaml"] is None
    error = str(results["ema/_some_.yaml"])
    assert "ema/_some_.yaml cannot be selected" in error


# -- a preset or a dict switches it on --------------------------------------------


def test_dict_for_an_optional_dataclass_is_an_error(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"ema/fast.yaml": "decay: 0.5\n"})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, ("ema={decay:0.5}",))
    message = str(excinfo.value)
    assert "'ema={decay:0.5}'" in message
    assert "'ema' is a subconfig of Config, and EmaCfg is a dataclass" in message
    assert "switched off" in message
    assert "ema=_some_ ema.decay=0.5" in message
    assert "ema=NAME" in message
    assert "Presets: fast" in message


def test_dict_switches_on_an_optional_typeddict() -> None:
    assert compose_td(("ema={decay:0.5}",)).ema == {"decay": 0.5, "inner": {}}
    assert compose_td(("ema={}",)).ema == {"inner": {}}


def test_nested_dict_goes_into_the_nested_subconfig() -> None:
    config = compose_td(("ema={inner:{size:4}}",))
    assert config.ema == {"inner": {"size": 4}}


@pytest.mark.parametrize(
    "text", ["ema:\n  decay: 0.5\n", "ema: {}\n", "ema:\n  _replace_: true\n"]
)
def test_mapping_in_a_file_does_not_switch_it_on(tmp_path: Path, text: str) -> None:
    # Hydra switches it on; nematocyte switches it on under `_defaults_`.
    config_dir = configs(tmp_path, {"c.yaml": text})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'ema' is an optional subconfig of Config (EmaCfg | None)" in message
    assert "switched off" in message
    assert "_defaults_:\n      ema: _some_" in message


def test_mapping_in_a_preset_shows_the_path_from_the_preset(tmp_path: Path) -> None:
    @dataclass
    class Outer:
        ema: EmaCfg | None = None

    @dataclass
    class Nested:
        outer: Outer = field(default_factory=Outer)

    config_dir = configs(tmp_path, {"outer/fast.yaml": "ema:\n  decay: 0.5\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Nested, config_dir, overrides=("outer=fast",))
    message = str(excinfo.value)
    assert "fast.yaml" in message
    assert "'outer.ema' is an optional subconfig of Outer" in message
    assert "_defaults_:\n      ema: _some_" in message


def test_mapping_in_a_file_merges_when_it_is_on(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "on:\n  every: 3\n"})
    assert compose(config_dir, config_name="c").on == EmaCfg(decay=0.8, every=3)


def test_switching_on_starts_from_the_class_defaults(tmp_path: Path) -> None:
    # Not from the default instance of the field, `EmaCfg(decay=0.8)`.
    config_dir = configs(
        tmp_path,
        {
            "off.yaml": "_defaults_:\n  on: null\n",
            "on_again.yaml": (
                "_defaults_:\n  _self_: off\n  on: _some_\non:\n  every: 3\n"
            ),
            "on/three.yaml": "every: 3\n",
        },
    )
    assert compose(config_dir, config_name="on_again").on == EmaCfg(every=3)
    assert compose(config_dir, ("on=three",), config_name="off").on == EmaCfg(every=3)


def test_dict_for_an_optional_typeddict_that_is_on_replaces_it() -> None:
    # Not merged into the default, `{decay: 0.8}`.
    assert compose_td(("on={every:3}",)).on == {"every": 3, "inner": {}}


def test_preset_switches_it_on(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path,
        {
            "ema/fast.yaml": "decay: 0.5\n",
            "with_fast.yaml": "_defaults_:\n  ema: fast\n",
        },
    )
    assert compose(config_dir, ("ema=fast",)).ema == EmaCfg(decay=0.5)
    assert compose(config_dir, config_name="with_fast").ema == EmaCfg(decay=0.5)


def test_preset_that_extends_a_preset_switches_it_on(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path,
        {
            "ema/fast.yaml": "decay: 0.5\n",
            "ema/faster.yaml": "_defaults_:\n  _self_: fast\nevery: 2\n",
        },
    )
    config = compose(config_dir, ("ema=faster",))
    assert config.ema == EmaCfg(decay=0.5, every=2)


def test_preset_starts_an_optional_subconfig_that_is_on_again(tmp_path: Path) -> None:
    # `on` is on, with the default instance `EmaCfg(decay=0.8)`, and `run`
    # sets `decay` again. A preset starts from the class defaults, as an
    # option of a group does, and warns about the value that it drops.
    config_dir = configs(
        tmp_path, {"on/fast.yaml": "every: 5\n", "run.yaml": "on:\n  decay: 0.5\n"}
    )
    with pytest.warns(nematocyte.DroppedValueWarning, match="on.decay=0.5"):
        config = compose(config_dir, ("on=fast",), config_name="run")
    assert config.on == EmaCfg(every=5)


def test_some_keeps_an_optional_subconfig_that_is_on(tmp_path: Path) -> None:
    # `_some_` only switches it on; what was set stays, without a warning.
    config_dir = configs(tmp_path, {"run.yaml": "on:\n  every: 5\n"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = compose(config_dir, ("on=_some_",), config_name="run")
    assert config.on == EmaCfg(decay=0.8, every=5)


# -- a dotted key does not switch it on -------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [("ema.decay=0.5",), ("ema=null", "ema.decay=0.5"), ("ema.decay=0.5", "ema=null")],
)
def test_dotted_key_under_a_subconfig_that_is_off_is_an_error(
    overrides: tuple[str, ...],
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=overrides)
    message = str(excinfo.value)
    assert "'ema.decay=0.5'" in message
    assert "'ema.decay' cannot be set" in message
    assert "EmaCfg" in message
    assert "Config" in message
    # A dataclass is switched on by `_some_` or a named config, not a dict.
    assert "ema=_some_ ema.decay=0.5" in message
    assert "ema: _some_ under '_defaults_'" in message
    assert "ema=NAME" in message
    assert "ema={" not in message


def test_dotted_key_under_a_typeddict_that_is_off_points_to_a_dict() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose_td(("ema.decay=0.5",))
    message = str(excinfo.value)
    assert "'ema.decay' cannot be set" in message
    assert "EmaTd | None" in message
    assert "ema={decay:0.5}" in message
    assert "ema=_some_" in message


def test_dotted_selection_under_a_subconfig_that_is_off_is_an_error(
    tmp_path: Path,
) -> None:
    config_dir = configs(
        tmp_path, {"ema/inner/big.yaml": "size: 9\n", "ema/on.yaml": "every: 1\n"}
    )
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, ("ema.inner=big",))
    assert "ema=NAME" in str(excinfo.value)
    # Switched on, the preset of the nested subconfig applies.
    config = compose(config_dir, ("ema=on", "ema.inner=big"))
    assert config.ema == EmaCfg(inner=Inner(size=9))


def test_dotted_defaults_entry_under_a_subconfig_that_is_off_is_an_error(
    tmp_path: Path,
) -> None:
    config_dir = configs(
        tmp_path,
        {
            "ema/inner/big.yaml": "size: 9\n",
            "c.yaml": "_defaults_:\n  ema.inner: big\n",
        },
    )
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'ema' is an optional subconfig" in message


@pytest.mark.parametrize(
    "overrides",
    [("ema={decay:0.5}", "ema.every=4"), ("ema.every=4", "ema={decay:0.5}")],
)
def test_dict_switches_on_before_the_dotted_keys(overrides: tuple[str, ...]) -> None:
    # A dict is applied with the selections, before every value.
    assert compose_td(overrides).ema == {"decay": 0.5, "every": 4, "inner": {}}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (("ema={decay:0.2}", "ema={decay:0.5}"), {"decay": 0.5, "inner": {}}),
        (("ema={every:2}", "ema={decay:0.5}"), {"decay": 0.5, "inner": {}}),
    ],
)
def test_two_dicts_for_the_same_key_the_last_one_wins(
    overrides: tuple[str, ...], expected: dict[str, Any]
) -> None:
    assert compose_td(overrides).ema == expected


def test_preset_and_dict_are_the_same_key(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"ema/fast.yaml": "decay: 0.5\nevery: 7\n"})
    config = compose_td(("ema=fast", "ema={every:3}"), config_dir)
    assert config.ema == {"every": 3, "inner": {}}
    config = compose_td(("ema={every:3}", "ema=fast"), config_dir)
    assert config.ema == {"decay": 0.5, "every": 7, "inner": {}}


@pytest.mark.parametrize(
    "overrides", [("~td.name", "td={name:y}"), ("td={name:y}", "~td.name")]
)
def test_unset_applies_after_a_dict_for_the_typeddict(
    overrides: tuple[str, ...],
) -> None:
    class Td(TypedDict, total=False):
        name: str

    @dataclass
    class C:
        td: Td = field(default_factory=lambda: Td(name="x"))

    assert compose_with(C, overrides).td == {}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (("ema=null", "ema={}"), {"inner": {}}),
        (("ema={every:2}", "ema=null"), None),
    ],
)
def test_null_and_a_dict_are_the_same_key(
    overrides: tuple[str, ...], expected: dict[str, Any] | None
) -> None:
    assert compose_td(overrides).ema == expected


@pytest.mark.parametrize(
    "overrides",
    [
        ("outer.ema=null", "outer={ema:{decay:0.5}}"),
        ("outer={ema:{decay:0.5}}", "outer.ema=null"),
    ],
)
def test_null_under_a_dict_applies_after_it(overrides: tuple[str, ...]) -> None:
    class Outer(TypedDict):
        ema: EmaTd | None

    @dataclass
    class C:
        outer: Outer = field(default_factory=lambda: Outer(ema=None))

    assert compose_with(C, overrides).outer == {"ema": None}


def test_null_and_preset_are_the_same_key(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"ema/fast.yaml": "decay: 0.5\n"})
    assert compose(config_dir, ("ema=fast", "ema=null")).ema is None
    assert compose(config_dir, ("ema=null", "ema=fast")).ema == EmaCfg(decay=0.5)


def test_tilde_on_an_optional_subconfig_points_to_null() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("~on",))
    message = str(excinfo.value)
    assert "'~on'" in message
    assert "on=null" in message


# -- a dict for a subconfig means what it means in a file ---------------------------


def test_dict_for_a_plain_subconfig_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("plain={lr:0.5,kwargs:{a:1}}",))
    message = str(excinfo.value)
    assert "'plain={lr:0.5,kwargs:{a:1}}'" in message
    assert "'plain' is a subconfig of Config, and Plain is a dataclass" in message
    # The dotted keys stop at the dict value.
    assert "plain.lr=0.5 plain.kwargs={a:1}" in message
    assert "plain=NAME" in message


def test_dict_for_a_dataclass_inside_a_typeddict_is_an_error() -> None:
    class Td(TypedDict):
        name: NotRequired[str]
        plain: Plain

    @dataclass
    class C:
        td: Td = field(default_factory=lambda: Td(plain=Plain()))

    with pytest.raises(ConfigError) as excinfo:
        compose_with(C, ("td={name:a,plain:{lr:0.5}}",))
    message = str(excinfo.value)
    assert "'td={name:a,plain:{lr:0.5}}'" in message
    assert "'td.plain' is a subconfig of Td (at 'td'), and Plain is a dataclass" in (
        message
    )
    assert "td.plain.lr=0.5" in message
    # Its fields are set with dotted keys, after the dict.
    config = compose_with(C, ("td.plain.lr=0.5", "td={name:a}"))
    assert config.td == {"name": "a", "plain": Plain(lr=0.5)}


def test_unknown_key_in_the_dict_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose_td(("ema={decayy:0.5}",))
    message = str(excinfo.value)
    assert "ema={decayy:0.5}" in message
    assert "'decayy' is not a field of EmaTd" in message
    assert "Did you mean 'decay'?" in message


def test_string_for_a_nested_subconfig_in_the_dict_points_to_a_selection() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose_td(("ema={inner:big}",))
    message = str(excinfo.value)
    assert "ema={inner:big}" in message
    assert "ema.inner=big" in message
    assert "_defaults_" not in message


def test_dict_for_a_group_of_dataclasses_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("group={x:1}",))
    message = str(excinfo.value)
    assert "'group={x:1}'" in message
    assert "'group' is a group of Config, whose options are dataclasses" in message
    assert "group.x=1" in message
    assert "Options: a, b" in message


# -- the rest of the node ---------------------------------------------------------


def test_interpolation_under_a_subconfig_that_is_off_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("seed=${ema.every}",))
    message = str(excinfo.value)
    assert "${ema.every}" in message
    assert "switched off" in message


def test_interpolation_under_a_subconfig_that_is_on() -> None:
    assert compose(overrides=("seed=${on.every}", "on.every=3")).seed == 3


def test_to_dict_writes_null() -> None:
    data = nematocyte.to_dict(compose())
    assert data["ema"] is None
    assert data["on"] == {"decay": 0.8, "every": 1, "inner": {"size": 1}}


def test_from_run_config_reads_null(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"ema/two.yaml": "every: 2\n"})
    config = compose(config_dir, ("on=null", "ema=two"))
    data = nematocyte.to_dict(config)
    assert nematocyte.from_run_config(Config, data) == config


# -- --help and --cfg -------------------------------------------------------------


def test_help_shows_the_type_and_null_when_off() -> None:
    text = help_text(Config)
    assert re.search(
        r"^ema=null +\[EmaCfg \| null\] Off unless switched on\.$", text, re.M
    )
    assert "ema.decay" not in text
    assert re.search(r"^on: +\[EmaCfg \| null\] Exponential moving", text, re.M)
    assert re.search(r"^on\.decay=0\.8 ", text, re.M)


def test_help_lists_the_presets_when_off(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"ema/fast.yaml": "_doc_: Decays fast.\n"})
    text = help_text(Config, config_dir)
    assert re.search(r"^ +Presets:\n +fast +Decays fast\.$", text, re.M)


def test_cfg_shows_null_when_off(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"ema/on.yaml": "every: 1\n"})
    text = cfg_text(Config, config_dir, overrides=("on=null", "ema=on"))
    assert (
        "# Exponential moving average of the weights. (EmaCfg | null)\non: null\n"
        in text
    )
    assert (
        "# Off unless switched on. (EmaCfg | null)\nema:\n  # (float)\n  decay: 0.99\n"
        in text
    )


# -- multirun ---------------------------------------------------------------------


def run_main(config_dir: Path, args: list[str]) -> tuple[int, list[Config]]:
    seen: list[Config] = []

    @nematocyte.main(Config, config_path=config_dir)
    def main(config: Config) -> None:
        seen.append(config)

    try:
        main(args)
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        return exc.code, seen
    return 0, seen


def test_sweep_over_null_and_presets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_dir = configs(
        tmp_path, {"ema/fast.yaml": "every: 7\n", "ema/slow.yaml": "decay: 0.5\n"}
    )
    code, seen = run_main(config_dir, ["-m", "ema=null,slow,fast"])
    capsys.readouterr()
    assert code == 0
    assert [c.ema for c in seen] == [None, EmaCfg(decay=0.5), EmaCfg(every=7)]


def test_sweep_over_null_and_some(capsys: pytest.CaptureFixture[str]) -> None:
    code, seen = run_main(empty_config_dir(), ["-m", "ema=null,_some_"])
    capsys.readouterr()
    assert code == 0
    assert [c.ema for c in seen] == [None, EmaCfg()]


def test_sweep_under_a_subconfig_that_is_off_is_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, seen = run_main(empty_config_dir(), ["-m", "ema.decay=0.5,0.9"])
    assert code == 1
    assert not seen
    err = capsys.readouterr().err
    assert "ema.decay=0.5" in err
    assert "ema=NAME" in err


def test_optuna_key_under_a_subconfig_that_is_off_is_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ["-m", "nematocyte.sweeper=optuna", "ema.decay=interval(0.5, 0.9)"]
    code, seen = run_main(empty_config_dir(), args)
    assert code == 1
    assert not seen
    err = capsys.readouterr().err
    assert "'ema.decay'" in err
    assert "switched off" in err


# -- validation -------------------------------------------------------------------


def test_validator_checks_presets_of_a_subconfig_that_is_off(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path,
        {
            "ema/fast.yaml": "decay: 0.5\n",
            "ema/inner/big.yaml": "size: 9\n",
            "ema/inner/bad.yaml": "sise: 9\n",
        },
    )
    results = {
        r.path.as_posix(): r.error
        for r in nematocyte.validate_configs(Config, config_dir)
    }
    assert results["ema/fast.yaml"] is None
    assert results["ema/inner/big.yaml"] is None
    error = str(results["ema/inner/bad.yaml"])
    assert "ema/inner/bad.yaml" in error
    assert "'sise' is not a field of Inner" in error
