"""Optional groups, typed `Model | None` (`design/unions.md`)."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Annotated, Literal, NotRequired, Optional, TypedDict
import warnings

from pydantic import Discriminator, Tag
import pytest

import nematocyte
from nematocyte import ConfigError, SchemaError
from nematocyte._schema import Kind, class_fields

from .helpers import empty_config_dir, write_files
from .test_help import cfg_text, help_text


@dataclass
class Norm:
    eps: float = 1e-5


@dataclass
class Fcn:
    """A fully connected network."""

    width: int = 1
    norm: Norm = field(default_factory=Norm)


@dataclass
class Cnn:
    """A convolutional network."""

    kernel: int = 3
    norm: Norm = field(default_factory=Norm)


type Model = Annotated[
    Annotated[Fcn, Tag("fcn")] | Annotated[Cnn, Tag("cnn")],
    Discriminator(nematocyte.pick),
]


class BatchNorm(TypedDict):
    _option_: Literal["BN"]
    momentum: NotRequired[float]


class LayerNorm(TypedDict):
    _option_: Literal["LN"]
    eps: NotRequired[float]


type NormGroup = Annotated[BatchNorm | LayerNorm, Discriminator("_option_")]


@dataclass
class Config:
    model: Model | None = field(default=None, doc="Off unless selected.")
    on: Model | None = field(default_factory=lambda: Cnn(kernel=5))
    norm: NormGroup | None = None
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


def error_of(*overrides: str, config_dir: Path | None = None) -> str:
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, overrides)
    return str(excinfo.value)


# -- the type ---------------------------------------------------------------


def test_group_or_none_is_an_optional_group() -> None:
    @dataclass
    class C:
        alias: Model | None = None
        inline: (
            Annotated[
                Annotated[Fcn, Tag("fcn")] | Annotated[Cnn, Tag("cnn")],
                Discriminator(nematocyte.pick),
            ]
            | None
        ) = None
        optional: Optional[Model] = None  # noqa: UP045
        typeddicts: NormGroup | None = None

    fields = class_fields(C)
    for name in ("alias", "inline", "optional"):
        info = fields[name]
        assert info.kind is Kind.GROUP
        assert info.optional
        assert info.tagged
        assert info.options == {"fcn": Fcn, "cnn": Cnn}
    assert fields["typeddicts"].kind is Kind.GROUP
    assert fields["typeddicts"].optional
    assert fields["typeddicts"].options == {"BN": BatchNorm, "LN": LayerNorm}


def test_group_without_none_is_not_optional() -> None:
    @dataclass
    class C:
        model: Model = field(default_factory=Fcn)

    assert not class_fields(C)["model"].optional


def test_none_inside_the_union_of_a_group_is_an_error() -> None:
    @dataclass
    class C:
        model: Annotated[
            Annotated[Fcn, Tag("fcn")] | Annotated[Cnn, Tag("cnn")] | None,
            Discriminator(nematocyte.pick),
        ] = None

    with pytest.raises(SchemaError, match=r"C\.model.*inside the union.*\] \| None"):
        class_fields(C)


def test_none_inside_a_union_of_typeddicts_is_an_error() -> None:
    # pydantic would accept it, but there is one spelling.
    @dataclass
    class C:
        norm: Annotated[BatchNorm | LayerNorm | None, Discriminator("_option_")] = None

    with pytest.raises(SchemaError, match=r"C\.norm.*inside the union.*\| None"):
        class_fields(C)


def test_optional_group_without_a_discriminator_is_an_error() -> None:
    @dataclass
    class C:
        model: Annotated[Fcn, Tag("fcn")] | Annotated[Cnn, Tag("cnn")] | None = None

    with pytest.raises(
        SchemaError, match=r"C\.model.*Discriminator\(nematocyte\.pick\)\] \| None"
    ):
        class_fields(C)


def test_optional_typeddict_with_one_option_is_an_error() -> None:
    @dataclass
    class C:
        norm: BatchNorm | None = None

    with pytest.raises(
        SchemaError,
        match=r"C\.norm.*one option.*BatchNorm \| None.*without the key '_option_'",
    ):
        class_fields(C)


def test_not_required_group_points_to_x_or_none() -> None:
    class Td(TypedDict):
        model: NotRequired[Model]

    with pytest.raises(SchemaError, match=r"Td\.model.*group.*X \| None"):
        class_fields(Td)


def test_optional_group_in_a_typeddict() -> None:
    class Td(TypedDict):
        model: Model | None

    assert nematocyte.compose(Td, overrides=("model=null",)) == {"model": None}
    assert nematocyte.compose(Td, overrides=("model=fcn",)) == {"model": Fcn()}


# -- the default -------------------------------------------------------------


def test_default_none_is_off_and_default_instance_is_on() -> None:
    config = compose()
    assert config.model is None
    assert config.on == Cnn(kernel=5)
    assert config.norm is None


def test_no_default_and_no_selection_is_an_error_that_points_to_null() -> None:
    @dataclass
    class C:
        model: Model | None

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(C)
    message = str(excinfo.value)
    assert "model: no option is selected" in message
    assert "model=null" in message
    assert "fcn, cnn" in message
    assert nematocyte.compose(C, overrides=("model=null",)) == C(model=None)
    assert nematocyte.compose(C, overrides=("model=cnn",)) == C(model=Cnn())


def test_values_apply_on_top_of_the_default_instance() -> None:
    assert compose(overrides=("on.norm.eps=0.1",)).on == Cnn(
        kernel=5, norm=Norm(eps=0.1)
    )


# -- null switches it off ------------------------------------------------------


def test_null_on_the_command_line_switches_it_off() -> None:
    assert compose(overrides=("on=null",)).on is None


def test_null_under_defaults_switches_it_off(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"off.yaml": "_defaults_:\n  on: null\n"})
    assert compose(config_dir, config_name="off").on is None


def test_null_under_defaults_of_a_named_config(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"outer/off.yaml": "_defaults_:\n  model: null\n"})
    assert nematocyte.compose(WithOuter, config_dir).outer.model == Fcn()
    config = nematocyte.compose(WithOuter, config_dir, overrides=("outer=off",))
    assert config.outer.model is None


@pytest.mark.parametrize("text", ["on: null\n", "on:\n  # kernel: 7\n"])
def test_null_among_the_values_points_to_defaults(tmp_path: Path, text: str) -> None:
    # A key with nothing under it is null to YAML, so commenting out every
    # line under it does not switch the group off.
    config_dir = configs(tmp_path, {"c.yaml": text})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'on' is an optional group of Config" in message
    assert "_defaults_:\n      on: null" in message
    assert "leave out the key" in message


@dataclass
class Outer:
    model: Model | None = field(default_factory=Fcn)


@dataclass
class WithOuter:
    outer: Outer = field(default_factory=Outer)


def test_null_among_the_values_shows_the_path_from_the_file(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "outer:\n  model: null\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(WithOuter, config_dir, config_name="c")
    message = str(excinfo.value)
    assert "'outer.model' is an optional group of Outer" in message
    assert "_defaults_:\n      outer.model: null" in message


def test_null_in_a_dict_on_the_command_line_switches_it_off() -> None:
    # The dict stands for its dotted keys, and `outer.model=null` is allowed.
    config = nematocyte.compose(WithOuter, overrides=("outer={model:null}",))
    assert config.outer.model is None


def test_values_after_null_under_defaults_are_an_error(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path, {"c.yaml": "_defaults_:\n  on: null\non:\n  kernel: 2\n"}
    )
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'on' is an optional group of Config" in message
    assert "switched off" in message


def test_null_for_self_is_an_error(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "_defaults_:\n  _self_: null\n"})
    with pytest.raises(ConfigError, match=r"c\.yaml.*'_self_'.*None"):
        compose(config_dir, config_name="c")


def test_null_switches_off_a_group_of_typeddicts() -> None:
    config = compose(overrides=("norm=LN", "norm.eps=0.1"))
    assert config.norm == {"_option_": "LN", "eps": 0.1}
    assert compose(overrides=("norm=null",)).norm is None


def test_switching_off_drops_what_was_set_without_a_warning(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path,
        {
            "set.yaml": "_defaults_:\n  on: fcn\non:\n  width: 5\n",
            "off.yaml": "_defaults_:\n  _self_: set\n  on: null\n",
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compose(config_dir, ("on=null",), config_name="set").on is None
        assert compose(config_dir, config_name="off").on is None


def test_null_under_defaults_for_a_group_that_is_not_optional(
    tmp_path: Path,
) -> None:
    @dataclass
    class C:
        model: Model = field(default_factory=Fcn)

    config_dir = configs(tmp_path, {"c.yaml": "_defaults_:\n  model: null\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(C, config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'model' is a group of C" in message
    assert "not needed in nematocyte; remove it" in message


# -- selecting an option switches it on ---------------------------------------------


def test_selection_on_the_command_line_switches_it_on() -> None:
    assert compose(overrides=("model=cnn",)).model == Cnn()


def test_selection_in_a_file_switches_it_on(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path, {"c.yaml": "_defaults_:\n  model: fcn\nmodel:\n  width: 4\n"}
    )
    assert compose(config_dir, config_name="c").model == Fcn(width=4)


def test_config_in_the_groups_directory_switches_it_on(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path, {"model/wide.yaml": "_defaults_:\n  _self_: fcn\nwidth: 9\n"}
    )
    assert compose(config_dir, ("model=wide",)).model == Fcn(width=9)


def test_selection_after_null_in_a_lower_layer(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"off.yaml": "_defaults_:\n  on: null\n"})
    config = compose(config_dir, ("on=fcn", "on.width=3"), config_name="off")
    assert config.on == Fcn(width=3)


def test_selection_starts_from_the_class_defaults() -> None:
    # Not from the default instance, as for any group.
    assert compose(overrides=("on=null", "on=cnn")).on == Cnn()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (("model=null", "model=cnn"), Cnn()),
        (("model=cnn", "model=null"), None),
    ],
)
def test_null_and_a_selection_are_the_same_key(
    overrides: tuple[str, ...], expected: Cnn | None
) -> None:
    assert compose(overrides=overrides).model == expected


def test_selection_and_dict_both_apply_in_either_order() -> None:
    expected = Cnn(kernel=7)
    assert compose(overrides=("model=cnn", "model={kernel:7}")).model == expected
    assert compose(overrides=("model={kernel:7}", "model=cnn")).model == expected


# -- setting a field while it is off ------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [("model.kernel=2",), ("on=null", "on.kernel=2"), ("on.kernel=2", "on=null")],
)
def test_dotted_key_under_a_group_that_is_off_is_an_error(
    overrides: tuple[str, ...],
) -> None:
    message = error_of(*overrides)
    key = overrides[0].split(".")[0].split("=")[0]
    assert f"'{key}' is an optional group of Config" in message
    assert f"'{key}.kernel' cannot be set" in message
    assert f"{key}=fcn" in message
    assert "Options: fcn, cnn" in message


def test_dict_does_not_switch_on_a_group() -> None:
    message = error_of("model={kernel:2}")
    assert "'model={kernel:2}'" in message
    assert "'model' is an optional group of Config" in message
    assert "model=fcn" in message


def test_null_and_a_dict_for_the_same_group_is_an_error() -> None:
    message = error_of("on={kernel:2}", "on=null")
    assert "'on' is an optional group of Config" in message


def test_mapping_in_a_file_does_not_switch_on_a_group(tmp_path: Path) -> None:
    config_dir = configs(tmp_path, {"c.yaml": "model:\n  kernel: 2\n"})
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'model' is an optional group of Config" in message
    assert "model: fcn" in message


def test_dotted_defaults_entry_under_a_group_that_is_off_is_an_error(
    tmp_path: Path,
) -> None:
    config_dir = configs(
        tmp_path,
        {"c.yaml": "_defaults_:\n  model.norm: small\n", "model/norm/small.yaml": ""},
    )
    with pytest.raises(ConfigError) as excinfo:
        compose(config_dir, config_name="c")
    message = str(excinfo.value)
    assert "c.yaml" in message
    assert "'model.norm' cannot be set" in message


def test_tilde_on_an_optional_group_points_to_null() -> None:
    message = error_of("~model")
    assert "'~model'" in message
    assert "model=null" in message


# -- the rest of the node ---------------------------------------------------------


def test_error_from_pydantic_names_the_key() -> None:
    message = error_of("model=cnn", "model.kernel=big")
    assert "model.kernel" in message
    assert "Cnn" in message


def test_interpolation_under_a_group_that_is_off_is_an_error() -> None:
    message = error_of("seed=${model.kernel}")
    assert "${model.kernel}" in message
    assert "switched off" in message


def test_interpolation_under_a_group_that_is_on() -> None:
    assert compose(overrides=("seed=${on.kernel}",)).seed == 5


def test_to_dict_writes_null() -> None:
    data = nematocyte.to_dict(compose())
    assert data["model"] is None
    assert data["norm"] is None
    assert data["on"]["_option_"] == "cnn"


def test_from_run_config_reads_null() -> None:
    config = compose(overrides=("on=null", "model=fcn"))
    data = nematocyte.to_dict(config)
    assert nematocyte.from_run_config(Config, data) == config


# -- --help and --cfg -------------------------------------------------------------


def test_help_shows_null_when_off() -> None:
    text = help_text(Config)
    assert re.search(r"^model=null +Off unless selected\.$", text, re.M)
    assert re.search(r"^ +fcn +A fully connected network\.$", text, re.M)
    assert "model.width" not in text
    assert re.search(r"^on=cnn", text, re.M)
    assert re.search(r"^on\.kernel=5 ", text, re.M)


def test_cfg_shows_null_when_off() -> None:
    text = cfg_text(Config, overrides=("on=null",))
    assert "# Options: fcn, cnn. Selected: null.\non: null\n" in text
    assert "Off unless selected. Options: fcn, cnn. Selected: null." in text


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


def test_sweep_over_null_and_options(capsys: pytest.CaptureFixture[str]) -> None:
    code, seen = run_main(empty_config_dir(), ["-m", "model=null,fcn,cnn"])
    capsys.readouterr()
    assert code == 0
    assert [c.model for c in seen] == [None, Fcn(), Cnn()]


def test_sweep_under_a_group_that_is_off_is_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, seen = run_main(empty_config_dir(), ["-m", "model.kernel=1,2"])
    assert code == 1
    assert not seen
    err = capsys.readouterr().err
    assert "'model.kernel'" in err
    assert "model=fcn" in err


# -- validation -------------------------------------------------------------------


def test_validator_checks_configs_of_a_group_that_is_off(tmp_path: Path) -> None:
    config_dir = configs(
        tmp_path,
        {
            "model/wide.yaml": "_defaults_:\n  _self_: fcn\nwidth: 9\n",
            "model/bad.yaml": "_defaults_:\n  _self_: cnn\nwidht: 9\n",
            "model/norm/small.yaml": "eps: 0.1\n",
        },
    )
    results = {
        r.path.as_posix(): r.error
        for r in nematocyte.validate_configs(Config, config_dir)
    }
    assert results["model/wide.yaml"] is None
    assert results["model/norm/small.yaml"] is None
    error = str(results["model/bad.yaml"])
    assert "model/bad.yaml" in error
    assert "widht" in error
