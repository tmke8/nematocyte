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


def test_null_in_a_file_switches_it_off(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"off.yaml": "on: null\n"})
    assert compose(config_dir, config_name="off").on is None


def test_key_with_nothing_under_it_switches_it_off(tmp_path: Path) -> None:
    # YAML reads a key with only comments under it as null.
    config_dir = configs(tmp_path, {"off.yaml": "on:\n  # decay: 0.3\n"})
    assert compose(config_dir, config_name="off").on is None


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
    assert "plain: {}" in message


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


def test_null_entry_in_defaults_points_to_the_values(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "_defaults_:\n  ema: null\n"})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'ema' is an optional subconfig" in message
    assert "among the values, outside '_defaults_': ema: null" in message


# -- a mapping switches it on ---------------------------------------------------


def test_dict_for_an_optional_dataclass_is_an_error(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"ema/fast.yaml": "decay: 0.5\n"})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, ("ema={decay:0.5}",))
    message = str(excinfo.value)
    assert "'ema={decay:0.5}'" in message
    assert "'ema' is a subconfig of Config, and EmaCfg is a dataclass" in message
    assert "switched off" in message
    assert "ema=NAME" in message
    assert str(config_dir / "ema") in message
    assert "Presets: fast" in message


def test_dict_switches_on_an_optional_typeddict() -> None:
    assert compose_td(("ema={decay:0.5}",)).ema == {"decay": 0.5, "inner": {}}
    assert compose_td(("ema={}",)).ema == {"inner": {}}


def test_nested_dict_goes_into_the_nested_subconfig() -> None:
    config = compose_td(("ema={inner:{size:4}}",))
    assert config.ema == {"inner": {"size": 4}}


def test_mapping_in_a_file_switches_it_on(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"switch_on.yaml": "ema:\n  decay: 0.5\n"})
    assert compose(config_dir, config_name="switch_on").ema == EmaCfg(decay=0.5)


def test_switching_on_starts_from_the_class_defaults(tmp_path: Path) -> None:
    # Not from the default instance of the field, `EmaCfg(decay=0.8)`.
    config_dir = configs(
        tmp_path,
        {
            "off.yaml": "on: null\n",
            "on_again.yaml": "_defaults_:\n  _self_: off\non:\n  every: 3\n",
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
    # A dataclass is switched on by a named config, not a dict.
    assert "ema=NAME" in message
    assert "ema={" not in message


def test_dotted_key_under_a_typeddict_that_is_off_points_to_a_dict() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose_td(("ema.decay=0.5",))
    message = str(excinfo.value)
    assert "'ema.decay' cannot be set" in message
    assert "EmaTd | None" in message
    assert "ema={decay:0.5}" in message


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
