"""TypedDicts as config classes (`design/deserializer.md`, "TypedDict configs")."""

from dataclasses import dataclass, field, make_dataclass
import logging
from pathlib import Path
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    Literal,
    NotRequired,
    ReadOnly,
    Required,
    TypedDict,
    cast,
)
import typing_extensions as te

import pytest

import nematocyte
from nematocyte import ConfigError, SchemaError, option
from nematocyte._compose import Composer
from nematocyte._help import format_cfg, format_help
from nematocyte._overrides import parse_override
from nematocyte._schema import Kind, class_fields
from nematocyte._settings import Settings

from .helpers import write_files

# -- classes -------------------------------------------------------------------


class Opt(TypedDict):
    """The optimizer."""

    lr: float
    wd: NotRequired[float]


class Partial(TypedDict, total=False):
    a: Required[int]
    b: ReadOnly[str]


class Child(Partial):
    c: Annotated[list[int], "kept"]


@dataclass(kw_only=True)
class Fcn:
    num_hidden: int = 1


@dataclass(kw_only=True)
class Cnn:
    kernel_size: int = 5


class Sgd(TypedDict):
    """Stochastic gradient descent."""

    _option_: Literal["sgd"]
    momentum: float


class Adam(TypedDict):
    _option_: ReadOnly[Literal["adam"]]
    beta1: float
    eps: NotRequired[float]


type Optimizer = Sgd | Adam
type Model = Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")]


class Root(TypedDict):
    """Train a model."""

    seed: int
    name: NotRequired[str]
    opt: Opt
    model: Model


@dataclass(kw_only=True)
class DataclassRoot:
    seed: int = 0
    opt: Opt = field(default_factory=lambda: Opt(lr=0.1))


def compose_root(cls: Any, files: dict[str, str] | None = None, **kwargs: Any) -> Any:
    config_dir = Path.cwd() / "configs"
    config_dir.mkdir(exist_ok=True)
    write_files(config_dir, files or {})
    return nematocyte.compose(cls, config_dir, **kwargs)


# -- the schema ----------------------------------------------------------------


def test_keys_of_a_typeddict_are_its_fields() -> None:
    fields = class_fields(Root)
    assert list(fields) == ["seed", "name", "opt", "model"]
    assert fields["seed"].kind is Kind.VALUE
    assert fields["opt"].kind is Kind.CONFIG
    assert fields["opt"].config_cls is Opt
    assert fields["model"].kind is Kind.GROUP
    assert fields["seed"].required
    assert not fields["name"].required
    assert fields["name"].annotation is str


def test_total_required_readonly_and_inheritance() -> None:
    fields = class_fields(Child)
    assert list(fields) == ["a", "b", "c"]
    assert [info.required for info in fields.values()] == [True, False, True]
    assert fields["b"].annotation is str
    assert fields["c"].annotation == Annotated[list[int], "kept"]


def test_typeddict_options_of_a_group() -> None:
    class Config(TypedDict):
        optimizer: Optimizer

    info = class_fields(Config)["optimizer"]
    assert info.kind is Kind.GROUP
    assert dict(info.options) == {"sgd": Sgd, "adam": Adam}
    # `_option_` is not a config key.
    assert list(class_fields(Sgd)) == ["momentum"]


def test_one_typeddict_with_an_option_key_is_a_group_of_one() -> None:
    class Config(TypedDict):
        optimizer: Sgd
        opt: Opt

    fields = class_fields(Config)
    assert fields["optimizer"].kind is Kind.GROUP
    assert dict(fields["optimizer"].options) == {"sgd": Sgd}
    assert fields["opt"].kind is Kind.CONFIG


def test_option_is_an_error_for_a_typeddict() -> None:
    @dataclass
    class Config:
        optimizer: Annotated[Sgd, option("sgd")] | Adam

    with pytest.raises(SchemaError) as info:
        class_fields(Config)
    message = str(info.value)
    assert "Config.optimizer" in message
    assert "option() is not used for a TypedDict (Sgd)" in message


def test_a_typeddict_in_a_group_needs_an_option_key() -> None:
    @dataclass
    class Config:
        optimizer: Sgd | Opt

    with pytest.raises(SchemaError) as info:
        class_fields(Config)
    message = str(info.value)
    assert "Config.optimizer" in message
    assert "Opt" in message
    assert "_option_" in message


class TwoNames(TypedDict):
    _option_: Literal["a", "b"]


class NotALiteral(TypedDict):
    _option_: str


class BadName(TypedDict):
    _option_: Literal["no spaces"]


class Optional(TypedDict):
    _option_: NotRequired[Literal["o"]]


@pytest.mark.parametrize(
    ("union", "name", "problem"),
    [
        (Sgd | TwoNames, "TwoNames", "a Literal of one string"),
        (Sgd | NotALiteral, "NotALiteral", "a Literal of one string"),
        (Sgd | BadName, "BadName", "not a valid name"),
        (Sgd | Optional, "Optional", "cannot be NotRequired"),
    ],
)
def test_option_key_has_to_be_one_literal_name(
    union: Any, name: str, problem: str
) -> None:
    config_cls = make_dataclass("Config", [("thing", union)])
    with pytest.raises(SchemaError) as info:
        class_fields(config_cls)
    message = str(info.value)
    assert f"{name}._option_" in message
    assert problem in message


def test_a_group_cannot_mix_dataclasses_and_typeddicts() -> None:
    @dataclass
    class Mixed:
        thing: Annotated[Fcn, option("fcn")] | Sgd

    with pytest.raises(SchemaError) as info:
        class_fields(Mixed)
    message = str(info.value)
    assert "Mixed.thing" in message
    assert "Fcn" in message
    assert "Sgd" in message


class OptionalSub(TypedDict):
    sub: NotRequired[Opt]


class OptionalGroup(TypedDict):
    sub: NotRequired[Optimizer]


@pytest.mark.parametrize("cls", [OptionalSub, OptionalGroup])
def test_a_subconfig_or_group_cannot_be_notrequired(cls: type) -> None:
    with pytest.raises(SchemaError, match=rf"{cls.__name__}\.sub: .*NotRequired"):
        class_fields(cls)


class WithDefaults(TypedDict):
    _defaults_: int


class WithSelf(TypedDict):
    _self_: int


@pytest.mark.parametrize("cls", [WithDefaults, WithSelf])
def test_reserved_names_are_rejected(cls: type) -> None:
    with pytest.raises(SchemaError, match=f"{cls.__name__}.* is reserved"):
        class_fields(cls)


def test_option_key_is_not_allowed_at_the_root() -> None:
    with pytest.raises(SchemaError, match=r"Sgd\._option_: the root"):
        nematocyte.compose(Sgd)


def test_option_key_cannot_be_set() -> None:
    with pytest.raises(ConfigError, match="'_option_' is not a field of Sgd"):
        compose_root(
            WithOptimizer,
            overrides=["optimizer=sgd", "optimizer.momentum=1", "optimizer._option_=x"],
        )


def test_reserved_key_at_a_typeddict_root() -> None:
    class Config(TypedDict):
        nematocyte: int

    with pytest.raises(SchemaError, match=r"Config\.nematocyte"):
        nematocyte.compose(Config)


@pytest.mark.parametrize("not_a_class", [int, list[int], Opt | None])
def test_compose_rejects_other_type_forms(not_a_class: Any) -> None:
    with pytest.raises(SchemaError, match="dataclass or a TypedDict"):
        nematocyte.compose(not_a_class)


# -- building ------------------------------------------------------------------


def test_compose_gives_plain_dicts() -> None:
    config = compose_root(Root, overrides=["seed=1", "opt.lr=0.5", "model=cnn"])
    assert type(config) is dict
    assert config == {"seed": 1, "opt": {"lr": 0.5}, "model": Cnn()}
    assert type(config["opt"]) is dict


def test_a_notrequired_key_is_present_once_set() -> None:
    files = {
        "run.yaml": "_defaults_:\n  model: fcn\nseed: 1\nname: x\nopt:\n  lr: 1e-3\n  wd: 0\n"
    }
    config = compose_root(Root, files, config_name="run")
    assert config == {
        "seed": 1,
        "name": "x",
        "opt": {"lr": 0.001, "wd": 0.0},
        "model": Fcn(),
    }


def test_a_missing_required_key_names_the_key_and_the_class() -> None:
    with pytest.raises(ConfigError) as info:
        compose_root(Root, overrides=["seed=1", "model=fcn"])
    message = str(info.value)
    assert "opt.lr: a value is required" in message
    assert "Opt" in message
    assert "seed" not in message
    assert "name" not in message


def test_an_invalid_value_names_the_key_the_file_and_the_class() -> None:
    files = {"run.yaml": "_defaults_:\n  model: fcn\nseed: abc\nopt:\n  lr: 0.1\n"}
    with pytest.raises(ConfigError) as info:
        compose_root(Root, files, config_name="run")
    message = str(info.value)
    assert "seed:" in message
    assert "run.yaml" in message
    assert "Root.seed (int)" in message


def test_a_bool_is_not_a_number_in_a_typeddict() -> None:
    with pytest.raises(ConfigError, match="a bool is not a number"):
        compose_root(Root, overrides=["seed=true", "opt.lr=1", "model=fcn"])


def test_dataclass_inside_typeddict_and_the_reverse() -> None:
    config = compose_root(DataclassRoot, overrides=["opt.wd=0.5"])
    assert config == DataclassRoot(opt={"lr": 0.1, "wd": 0.5})


# -- defaults ------------------------------------------------------------------


def test_dict_default_of_a_typeddict_subconfig() -> None:
    config = compose_root(DataclassRoot)
    assert config.opt == {"lr": 0.1}
    config = compose_root(DataclassRoot, overrides=["opt.lr=0.2"])
    assert config.opt == {"lr": 0.2}


def test_key_left_out_of_a_dict_default_is_not_set() -> None:
    @dataclass
    class Config:
        opt: Opt = field(default_factory=lambda: Opt(lr=0.1))
        empty: Opt = field(default_factory=lambda: cast("Opt", {}))

    with pytest.raises(ConfigError, match="empty.lr: a value is required"):
        compose_root(Config)


def test_default_of_a_typeddict_subconfig_has_to_be_a_dict() -> None:
    @dataclass
    class Wrong:
        opt: Opt = field(default_factory=Fcn)  # ty: ignore[invalid-assignment]  # pyright: ignore[reportAssignmentType]

    with pytest.raises(SchemaError, match=r"Wrong\.opt: the default is an instance"):
        compose_root(Wrong)


def test_another_mapping_is_not_a_default_of_a_typeddict_subconfig() -> None:
    """Only a dict or a `frozendict`; a dataclass takes a `MappingProxyType`."""

    @dataclass
    class Wrong:
        opt: Opt = MappingProxyType({"lr": 0.1})  # ty: ignore[invalid-assignment]  # pyright: ignore[reportAssignmentType]

    with pytest.raises(SchemaError) as info:
        compose_root(Wrong)
    message = str(info.value)
    assert "Wrong.opt" in message
    assert "mappingproxy" in message
    assert "Opt" in message


@dataclass
class WithDefaultOptimizer:
    optimizer: Optimizer = field(
        default_factory=lambda: Sgd(_option_="sgd", momentum=0.9)
    )


def test_default_of_a_group_of_typeddicts_names_its_option() -> None:
    config = compose_root(WithDefaultOptimizer)
    assert config.optimizer == {"_option_": "sgd", "momentum": 0.9}
    config = compose_root(WithDefaultOptimizer, overrides=["optimizer.momentum=0.5"])
    assert config.optimizer == {"_option_": "sgd", "momentum": 0.5}
    # Selecting the option by name starts from nothing: Sgd has no defaults.
    with pytest.raises(ConfigError, match="optimizer.momentum: a value is required"):
        compose_root(WithDefaultOptimizer, overrides=["optimizer=sgd"])
    config = compose_root(
        WithDefaultOptimizer, overrides=["optimizer=adam", "optimizer.beta1=0.8"]
    )
    assert config.optimizer == {"_option_": "adam", "beta1": 0.8}


@pytest.mark.parametrize("default", [{"momentum": 0.9}, {"_option_": "rmsprop"}])
def test_default_of_a_group_of_typeddicts_without_its_option(
    default: dict[str, Any],
) -> None:
    @dataclass
    class Config:
        optimizer: Optimizer = field(default_factory=lambda: cast("Sgd", default))

    with pytest.raises(SchemaError) as info:
        compose_root(Config)
    message = str(info.value)
    assert "Config.optimizer" in message
    assert "'_option_'" in message
    assert "sgd, adam" in message


# -- groups --------------------------------------------------------------------


@dataclass
class WithOptimizer:
    optimizer: Optimizer


def test_select_a_typeddict_option_by_name() -> None:
    config = compose_root(
        WithOptimizer, overrides=["optimizer=adam", "optimizer.beta1=0.8"]
    )
    assert config.optimizer == {"_option_": "adam", "beta1": 0.8}
    # The option comes first, as in the output of `to_dict`.
    assert next(iter(config.optimizer)) == "_option_"


def test_select_a_typeddict_option_from_a_group_file() -> None:
    files = {"optimizer/fast.yaml": "_defaults_:\n  _self_: sgd\nmomentum: 0.99\n"}
    config = compose_root(WithOptimizer, files, overrides=["optimizer=fast"])
    # The option is the one from the annotations, not the name of the file.
    assert config.optimizer == {"_option_": "sgd", "momentum": 0.99}


def test_match_on_the_option() -> None:
    config = compose_root(
        WithOptimizer, overrides=["optimizer=adam", "optimizer.beta1=0.8"]
    )
    match config.optimizer:
        case {"_option_": "sgd"}:
            pytest.fail("not sgd")
        case {"_option_": "adam", **rest}:
            assert rest == {"beta1": 0.8}


def test_a_group_of_typeddicts_needs_a_selection() -> None:
    with pytest.raises(ConfigError, match="optimizer: no option is selected"):
        compose_root(WithOptimizer)


# -- the entry point -----------------------------------------------------------


def test_main_with_a_typeddict_root(tmp_path: Path) -> None:
    seen: list[tuple[Root, int]] = []

    @nematocyte.main(Root, config_path=tmp_path)
    def main(config: Root) -> None:
        seen.append((config, nematocyte.log_level()))

    main(["seed=3", "opt.lr=0.1", "model=fcn", "nematocyte.log=debug"])
    assert seen == [({"seed": 3, "opt": {"lr": 0.1}, "model": Fcn()}, logging.DEBUG)]


def test_multirun_with_a_typeddict_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Root] = []

    @nematocyte.main(Root, config_path=tmp_path)
    def main(config: Root) -> None:
        seen.append(config)

    main(["-m", "seed=1,2", "opt.lr=0.1", "model=fcn,cnn"])
    capsys.readouterr()
    assert [(config["seed"], config["model"]) for config in seen] == [
        (1, Fcn()),
        (1, Cnn()),
        (2, Fcn()),
        (2, Cnn()),
    ]
    assert all("name" not in config for config in seen)
    assert all(config["opt"] == {"lr": 0.1} for config in seen)


def test_multirun_over_a_group_of_typeddicts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[WithDefaultOptimizer] = []

    @nematocyte.main(WithDefaultOptimizer, config_path=tmp_path)
    def main(config: WithDefaultOptimizer) -> None:
        seen.append(config)

    main(["-m", "optimizer.momentum=0.1,0.2"])
    capsys.readouterr()
    # Each job rebuilds its config from the resolved tree, `_option_` included.
    assert [config.optimizer for config in seen] == [
        {"_option_": "sgd", "momentum": 0.1},
        {"_option_": "sgd", "momentum": 0.2},
    ]


def test_validate_configs_with_a_typeddict_root(tmp_path: Path) -> None:
    write_files(
        tmp_path,
        {
            "good.yaml": "_defaults_:\n  model: fcn\nseed: 1\nopt:\n  lr: 0.1\n",
            "bad.yaml": "_defaults_:\n  model: fcn\nseed: 1\nopt:\n  lr: x\n",
        },
    )
    results = {
        str(result.path): result.ok
        for result in nematocyte.validate_configs(Root, tmp_path)
    }
    assert results == {"bad.yaml": False, "good.yaml": True}


def test_interpolation_to_an_unset_key() -> None:
    with pytest.raises(ConfigError, match="'name', which has no value"):
        compose_root(
            Root,
            overrides=["seed=1", "opt.lr=0.1", "model=fcn", "opt.wd=${name}"],
        )


# -- to_dict -------------------------------------------------------------------


@dataclass(kw_only=True)
class Logged:
    model: Model = field(default_factory=Cnn)
    opt: Opt = field(default_factory=lambda: Opt(lr=0.1))
    optimizer: Optimizer = field(
        default_factory=lambda: Adam(_option_="adam", beta1=0.9)
    )


def test_to_dict_keeps_typeddicts_as_they_are() -> None:
    config = compose_root(Logged)
    assert nematocyte.to_dict(config) == {
        "model": {"_option_": "cnn", "kernel_size": 5},
        "opt": {"lr": 0.1},
        "optimizer": {"_option_": "adam", "beta1": 0.9},
    }
    config = compose_root(Logged, overrides=["opt.wd=0.5", "optimizer.eps=1e-8"])
    result = nematocyte.to_dict(config)
    assert result["opt"] == {"lr": 0.1, "wd": 0.5}
    assert result["optimizer"] == {"_option_": "adam", "beta1": 0.9, "eps": 1e-8}


def test_to_dict_converts_the_values_of_a_typeddict() -> None:
    class Paths(TypedDict):
        root: Path
        sizes: tuple[int, ...]

    @dataclass
    class Config:
        paths: Paths = field(
            default_factory=lambda: Paths(root=Path("/data"), sizes=(1, 2))
        )

    assert nematocyte.to_dict(compose_root(Config)) == {
        "paths": {"root": "/data", "sizes": [1, 2]}
    }


def test_to_dict_refuses_a_typeddict_root() -> None:
    config = compose_root(Root, overrides=["seed=1", "opt.lr=0.1", "model=fcn"])
    with pytest.raises(TypeError, match="TypedDict as the root"):
        nematocyte.to_dict(config)


class WithGroup(TypedDict):
    model: Model


class WithSubconfig(TypedDict):
    opt: Opt


class Nested(TypedDict):
    _option_: Literal["nested"]
    inner: WithSubconfig


@dataclass(kw_only=True)
class GroupInTypedDict:
    section: WithGroup = field(default_factory=lambda: WithGroup(model=Cnn()))


@dataclass(kw_only=True)
class SubconfigInTypedDict:
    section: WithSubconfig = field(
        default_factory=lambda: WithSubconfig(opt=Opt(lr=0.1))
    )


@dataclass(kw_only=True)
class DeeplyNested:
    model: Model = field(default_factory=Cnn)
    thing: Nested = field(
        default_factory=lambda: Nested(
            _option_="nested", inner=WithSubconfig(opt=Opt(lr=0.1))
        )
    )


@pytest.mark.parametrize(
    ("cls", "where", "kind"),
    [
        (GroupInTypedDict, "WithGroup.model", "group"),
        (SubconfigInTypedDict, "WithSubconfig.opt", "subconfig"),
        (DeeplyNested, "Nested.inner", "subconfig"),
    ],
)
def test_to_dict_refuses_a_typeddict_that_holds_more_than_values(
    cls: type, where: str, kind: str
) -> None:
    config = compose_root(cls)
    with pytest.raises(TypeError) as info:
        nematocyte.to_dict(config)
    message = str(info.value)
    assert where in message
    assert f"is a {kind}" in message


# -- --help and --cfg ------------------------------------------------------------


def _texts(cls: type, overrides: tuple[str, ...]) -> tuple[str, str]:
    composer = Composer(cls, None, Settings)
    root = composer.compose(None, [parse_override(text) for text in overrides])
    return (
        format_help(composer, root, 88),
        format_cfg(composer, root, own_settings=False, width=88),
    )


def test_help_and_cfg_show_unset_keys() -> None:
    help_text, cfg_text = _texts(Root, ("model=fcn",))
    assert help_text.startswith("Train a model.\n")
    lines = help_text.splitlines()
    assert any(line.startswith("seed=???") and "(required)" in line for line in lines)
    assert any(line.startswith("name=") and "(not set)" in line for line in lines)
    assert any(line.startswith("opt:") and "The optimizer." in line for line in lines)
    assert "# name: not set" in cfg_text
    assert "  # wd: not set" in cfg_text
    assert "seed: ???  # a value is required" in cfg_text


def test_help_lists_the_options_of_a_group_of_typeddicts() -> None:
    help_text, cfg_text = _texts(WithDefaultOptimizer, ())
    lines = help_text.splitlines()
    assert any(
        line.startswith("optimizer=sgd")
        and "* sgd" in line
        and "Stochastic gradient descent." in line
        for line in lines
    )
    assert any(line.strip() == "adam" for line in lines)
    assert "optimizer.momentum=0.9" in help_text
    assert "Options: sgd, adam. Selected: sgd." in cfg_text
    assert "_option_" not in help_text


def test_cfg_of_a_typeddict_with_nothing_set() -> None:
    class Empty(TypedDict):
        a: NotRequired[int]

    @dataclass
    class Config:
        sub: Empty = field(default_factory=Empty)

    _, cfg_text = _texts(Config, ())
    assert "sub: {}" in cfg_text


# -- typing_extensions.TypedDict -------------------------------------------------------


class ClosedOpt(te.TypedDict, closed=True):
    """An optimizer, as a TypedDict from typing_extensions."""

    lr: float
    wd: NotRequired[float]


class ClosedSgd(te.TypedDict, closed=True):
    _option_: Literal["sgd"]
    momentum: NotRequired[float]


class ClosedAdam(te.TypedDict, closed=True):
    _option_: ReadOnly[Literal["adam"]]
    beta1: float


@dataclass(kw_only=True)
class WithClosed:
    opt: ClosedOpt = field(default_factory=lambda: ClosedOpt(lr=0.1))
    optimizer: ClosedSgd | ClosedAdam = field(
        default_factory=lambda: ClosedSgd(_option_="sgd")
    )


def test_typing_extensions_typeddicts_are_config_classes() -> None:
    fields = class_fields(WithClosed)
    assert fields["opt"].kind is Kind.CONFIG
    assert fields["optimizer"].kind is Kind.GROUP
    assert dict(fields["optimizer"].options) == {"sgd": ClosedSgd, "adam": ClosedAdam}
    assert [info.required for info in class_fields(ClosedOpt).values()] == [
        True,
        False,
    ]


def test_build_and_to_dict_with_typing_extensions_typeddicts() -> None:
    config = compose_root(
        WithClosed, overrides=["opt.wd=0.5", "optimizer=adam", "optimizer.beta1=0.9"]
    )
    assert config.opt == {"lr": 0.1, "wd": 0.5}
    assert config.optimizer == {"_option_": "adam", "beta1": 0.9}
    assert nematocyte.to_dict(config) == {
        "opt": {"lr": 0.1, "wd": 0.5},
        "optimizer": {"_option_": "adam", "beta1": 0.9},
    }
    assert compose_root(WithClosed).optimizer == {"_option_": "sgd"}


def test_typing_extensions_typeddict_as_the_root() -> None:
    config = compose_root(ClosedOpt, overrides=["lr=0.2"])
    assert config == {"lr": 0.2}
    help_text, _ = _texts(ClosedOpt, ())
    assert help_text.startswith("An optimizer, as a TypedDict from typing_extensions.")
