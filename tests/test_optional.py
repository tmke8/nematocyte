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
        compose(overrides=("ema={inner:null}",))
    message = str(excinfo.value)
    assert "ema={inner:null}" in message
    assert "'ema.inner' is a subconfig of EmaCfg" in message
    assert "Inner | None" in message


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


def test_empty_dict_switches_it_on_with_the_class_defaults() -> None:
    assert compose(overrides=("ema={}",)).ema == EmaCfg()


def test_dict_switches_it_on() -> None:
    assert compose(overrides=("ema={decay:0.5}",)).ema == EmaCfg(decay=0.5)


def test_nested_dict_goes_into_the_nested_subconfig() -> None:
    config = compose(overrides=("ema={inner:{size:4}}",))
    assert config.ema == EmaCfg(inner=Inner(size=4))


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
        },
    )
    assert compose(config_dir, config_name="on_again").on == EmaCfg(every=3)
    assert compose(config_dir, ("on={}",), config_name="off").on == EmaCfg()


def test_dict_is_merged_into_a_subconfig_that_is_on(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"set.yaml": "on:\n  every: 5\n"})
    config = compose(config_dir, ("on={decay:0.5}",), config_name="set")
    assert config.on == EmaCfg(decay=0.5, every=5)


def test_empty_dict_leaves_a_subconfig_that_is_on_as_it_is() -> None:
    assert compose(overrides=("on={}",)).on == EmaCfg(decay=0.8)


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
    assert "ema={decay:0.5}" in message


def test_dotted_selection_under_a_subconfig_that_is_off_is_an_error(
    tmp_path: Path,
) -> None:
    config_dir = configs(tmp_path, {"ema/inner/big.yaml": "size: 9\n"})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, ("ema.inner=big",))
    assert "ema={inner:big}" in str(excinfo.value)
    # Switched on, the preset of the nested subconfig applies.
    config = compose(config_dir, ("ema={}", "ema.inner=big"))
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


def test_dotted_key_and_dict_work_in_either_order() -> None:
    expected = EmaCfg(decay=0.5, every=4)
    assert compose(overrides=("ema={decay:0.5}", "ema.every=4")).ema == expected
    assert compose(overrides=("ema.every=4", "ema={decay:0.5}")).ema == expected


@pytest.mark.parametrize(
    ("overrides", "decay"),
    [
        (("ema.decay=0.2", "ema={decay:0.5}"), 0.5),
        (("ema={decay:0.5}", "ema.decay=0.2"), 0.2),
        (("ema={decay:0.2}", "ema={decay:0.5}"), 0.5),
    ],
)
def test_dict_stands_for_its_dotted_keys_the_later_one_wins(
    overrides: tuple[str, ...], decay: float
) -> None:
    assert compose(overrides=overrides).ema == EmaCfg(decay=decay)


def test_two_dicts_for_the_same_subconfig_both_apply() -> None:
    config = compose(overrides=("plain={lr:0.5}", "plain={momentum:0.9}"))
    assert config.plain == Plain(lr=0.5, momentum=0.9)


@pytest.mark.parametrize(
    ("overrides", "kwargs"),
    [
        (("plain.kwargs={a:1}", "plain={kwargs:{b:2}}"), {"b": 2}),
        (("plain={kwargs:{b:2}}", "plain.kwargs={a:1}"), {"a": 1}),
    ],
)
def test_dict_stops_at_a_field_that_holds_a_value(
    overrides: tuple[str, ...], kwargs: dict[str, int]
) -> None:
    # A `dict[...]` field in the dict is replaced as a whole, by the later one.
    assert compose(overrides=overrides).plain.kwargs == kwargs


@pytest.mark.parametrize(
    "overrides",
    [("ema=fast", "ema={every:3}"), ("ema={every:3}", "ema=fast")],
)
def test_preset_and_dict_both_apply_in_either_order(
    tmp_path: Path, overrides: tuple[str, ...]
) -> None:
    config_dir = configs(tmp_path, {"ema/fast.yaml": "decay: 0.5\nevery: 7\n"})
    assert compose(config_dir, overrides).ema == EmaCfg(decay=0.5, every=3)


def test_dict_switches_on_before_a_dotted_key_given_earlier() -> None:
    # Switching on is a selection, which comes before every value.
    config = compose(overrides=("ema.every=4", "ema={decay:0.5}"))
    assert config.ema == EmaCfg(decay=0.5, every=4)


def test_unset_and_dict_for_a_typeddict_key_the_later_one_wins() -> None:
    class Td(TypedDict, total=False):
        name: str

    @dataclass
    class C:
        td: Td = field(default_factory=lambda: Td(name="x"))

    assert compose_with(C, ("~td.name", "td={name:y}")).td == {"name": "y"}
    assert compose_with(C, ("td={name:y}", "~td.name")).td == {}


@pytest.mark.parametrize(
    "overrides",
    [("ema=null", "ema={}"), ("ema={every:2}", "ema=null")],
)
def test_null_and_a_dict_for_the_same_subconfig_is_an_error(
    overrides: tuple[str, ...],
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=overrides)
    message = str(excinfo.value)
    assert "'ema=null'" in message
    assert "'ema' is switched off by 'ema=null' and switched on by a dict" in message


def test_null_and_a_nested_dict_for_the_same_subconfig_is_an_error() -> None:
    @dataclass
    class Outer:
        ema: EmaCfg | None = None

    @dataclass
    class C:
        outer: Outer = field(default_factory=Outer)

    with pytest.raises(ConfigError) as excinfo:
        compose_with(C, ("outer.ema=null", "outer={ema:{decay:0.5}}"))
    message = str(excinfo.value)
    assert "'outer={ema:{decay:0.5}}'" in message
    assert "'outer.ema' is switched off by 'outer.ema=null'" in message


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


def test_dict_for_a_plain_subconfig_is_merged() -> None:
    config = compose(overrides=("plain={lr:0.5}", "plain.momentum=0.9"))
    assert config.plain == Plain(lr=0.5, momentum=0.9)


def test_dict_for_a_plain_subconfig_keeps_values_set_before(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "plain:\n  momentum: 0.9\n"})
    config = compose(config_dir, ("plain={lr:0.5}",), config_name="c")
    assert config.plain == Plain(lr=0.5, momentum=0.9)


def test_dict_field_inside_the_dict_is_replaced(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "plain:\n  kwargs:\n    a: 1\n"})
    config = compose(config_dir, ("plain={kwargs:{b:2}}",), config_name="c")
    assert config.plain.kwargs == {"b": 2}


def test_unknown_key_in_the_dict_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("plain={lrr:0.5}",))
    message = str(excinfo.value)
    assert "plain={lrr:0.5}" in message
    assert "'lrr' is not a field of Plain" in message
    assert "Did you mean 'lr'?" in message


def test_string_for_a_nested_subconfig_in_the_dict_points_to_a_selection() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("ema={inner:big}",))
    message = str(excinfo.value)
    assert "ema={inner:big}" in message
    assert "ema.inner=big" in message
    assert "_defaults_" not in message


def test_dict_for_a_group_goes_into_the_selected_option() -> None:
    assert compose(overrides=("group={x:1}",)).group == A(x=1)


@pytest.mark.parametrize(
    "overrides", [("group=b", "group={y:3}"), ("group={y:3}", "group=b")]
)
def test_selection_and_dict_for_a_group_both_apply_in_either_order(
    overrides: tuple[str, ...],
) -> None:
    assert compose(overrides=overrides).group == B(y=3)


def test_dict_with_a_key_of_another_option_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("group={y:3}",))
    message = str(excinfo.value)
    assert "'group={y:3}'" in message
    assert "'y' is not a field of A (the option 'a' of 'group')" in message


def test_dict_for_a_group_without_a_selected_option_is_an_error() -> None:
    @dataclass
    class C:
        group: Group

    with pytest.raises(ConfigError) as excinfo:
        compose_with(C, ("group={x:1}",))
    message = str(excinfo.value)
    assert "'group={x:1}'" in message
    assert "No option is selected for the group 'group'" in message
    assert "Options: a, b" in message


# -- the rest of the node ---------------------------------------------------------


def test_interpolation_under_a_subconfig_that_is_off_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        compose(overrides=("seed=${ema.every}",))
    message = str(excinfo.value)
    assert "${ema.every}" in message
    assert "switched off" in message


def test_interpolation_under_a_subconfig_that_is_on() -> None:
    assert compose(overrides=("seed=${on.every}", "on={every:3}")).seed == 3


def test_to_dict_writes_null() -> None:
    data = nematocyte.to_dict(compose())
    assert data["ema"] is None
    assert data["on"] == {"decay": 0.8, "every": 1, "inner": {"size": 1}}


def test_from_run_config_reads_null() -> None:
    config = compose(overrides=("on=null", "ema={every:2}"))
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


def test_cfg_shows_null_when_off() -> None:
    text = cfg_text(Config, overrides=("on=null", "ema={}"))
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


def test_sweep_over_null_a_dict_and_a_preset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_dir = configs(tmp_path, {"ema/fast.yaml": "every: 7\n"})
    code, seen = run_main(config_dir, ["-m", "ema=null,{decay:0.5},fast"])
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
    assert "ema={decay:0.5}" in err


@pytest.mark.parametrize(
    ("args", "lrs"),
    [
        (["plain.lr=0.1,0.2", "plain={lr:0.5}"], [0.5, 0.5]),
        (["plain={lr:0.5}", "plain.lr=0.1,0.2"], [0.1, 0.2]),
        (["plain={lr:0.1},{lr:0.2}", "plain.lr=0.5"], [0.5, 0.5]),
        (["plain.lr=0.5", "plain={lr:0.1},{lr:0.2}"], [0.1, 0.2]),
    ],
)
def test_swept_value_keeps_the_place_of_its_sweep(
    capsys: pytest.CaptureFixture[str], args: list[str], lrs: list[float]
) -> None:
    # As in Hydra.
    code, seen = run_main(empty_config_dir(), ["-m", *args])
    capsys.readouterr()
    assert code == 0
    assert [c.plain.lr for c in seen] == lrs


def test_key_swept_twice_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["-m", "plain={lr:0.1},{lr:0.2}", "plain={momentum:0.1},{momentum:0.2}"]
    code, seen = run_main(empty_config_dir(), args)
    assert code == 1
    assert not seen
    err = capsys.readouterr().err
    assert "'plain' is swept twice" in err
    assert "plain={lr:0.1},{lr:0.2}" in err


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
