"""Reading the schema off the dataclasses (`design/unions.md`, `helptext.md`)."""

from dataclasses import InitVar, dataclass, field
from typing import Annotated, ClassVar

import pytest

from nematocyte import ConfigError, SchemaError, option
from nematocyte._schema import Kind, class_doc, class_fields, validate_schema

from .helpers import BatchNorm, Celeba, Cmnist, Cnn, Config, DataModule, Fcn, LayerNorm


@dataclass
class A:
    x: int = 1


@dataclass
class B:
    y: int = 2


def test_kinds_of_fields() -> None:
    fields = class_fields(Config)
    assert list(fields) == ["dm", "model", "opt", "wandb", "seed", "gpu", "flag"]
    assert fields["dm"].kind is Kind.GROUP
    assert fields["model"].kind is Kind.GROUP
    assert fields["opt"].kind is Kind.CONFIG
    assert fields["seed"].kind is Kind.VALUE


def test_group_declared_inline() -> None:
    assert class_fields(Config)["dm"].options == {"celeba": Celeba, "cmnist": Cmnist}


def test_group_declared_through_an_alias() -> None:
    # `type Model = ...` has to be unwrapped through `__value__`.
    assert class_fields(Config)["model"].options == {"fcn": Fcn, "cnn": Cnn}
    assert class_fields(Cnn)["norm"].options == {
        "batchnorm": BatchNorm,
        "layernorm": LayerNorm,
    }


def test_members_need_no_common_base() -> None:
    assert not issubclass(Fcn, DataModule)
    assert not issubclass(Cnn, Fcn)


def test_ordinary_union_stays_a_value() -> None:
    @dataclass
    class C:
        name: str | None = None
        size: int | list[int] = 1

    assert all(info.kind is Kind.VALUE for info in class_fields(C).values())


def test_group_with_one_option() -> None:
    @dataclass
    class C:
        a: Annotated[A, option("only")]

    assert class_fields(C)["a"].kind is Kind.GROUP
    assert class_fields(C)["a"].options == {"only": A}


def test_unnamed_member_is_an_error() -> None:
    @dataclass
    class C:
        data: A | B

    with pytest.raises(SchemaError) as excinfo:
        class_fields(C)
    message = str(excinfo.value)
    assert "C.data" in message
    assert "name" in message


def test_partly_named_union_is_an_error() -> None:
    @dataclass
    class C:
        data: Annotated[A, option("a")] | B

    with pytest.raises(SchemaError, match="B"):
        class_fields(C)


def test_none_in_a_group_is_an_error() -> None:
    @dataclass
    class C:
        data: Annotated[A, option("a")] | None = None

    with pytest.raises(SchemaError, match="None"):
        class_fields(C)


def test_optional_dataclass_field_is_an_error() -> None:
    @dataclass
    class C:
        ema: A | None = None

    with pytest.raises(SchemaError, match=r"C\.ema"):
        class_fields(C)


def test_duplicate_option_name_is_an_error() -> None:
    @dataclass
    class C:
        data: Annotated[A, option("a")] | Annotated[B, option("a")]

    with pytest.raises(SchemaError, match="'a'"):
        class_fields(C)


def test_option_class_cannot_have_a_field_called_option() -> None:
    @dataclass
    class Bad:
        _option_: str = "x"

    @dataclass
    class C:
        data: Annotated[Bad, option("bad")] | Annotated[A, option("a")]

    with pytest.raises(SchemaError, match=r"Bad\._option_"):
        validate_schema(C)


@pytest.mark.parametrize("name", ["_self_", "_defaults_", "_option_"])
def test_reserved_field_names(name: str) -> None:
    # Reserved on every class, not only on options of a group.
    cls = dataclass(type("C", (), {"__annotations__": {name: int}, name: 1}))
    with pytest.raises(SchemaError, match=rf"C\.{name}: .* is reserved"):
        class_fields(cls)


def test_reserved_field_name_in_a_nested_class() -> None:
    @dataclass
    class Inner:
        _option_: int = 1

    @dataclass
    class Outer:
        inner: Inner = field(default_factory=Inner)

    with pytest.raises(SchemaError, match=r"Inner\._option_"):
        validate_schema(Outer)


def test_hydras_name_defaults_is_a_field_like_any_other() -> None:
    @dataclass
    class C:
        defaults: int = 1

    validate_schema(C)
    assert class_fields(C)["defaults"].kind is Kind.VALUE


def test_root_cannot_have_a_field_called_nematocyte() -> None:
    @dataclass
    class C:
        nematocyte: int = 1

    with pytest.raises(SchemaError, match="nematocyte"):
        validate_schema(C)


def test_schema_errors_are_config_errors() -> None:
    assert issubclass(SchemaError, ConfigError)


def test_invalid_option_name() -> None:
    with pytest.raises(SchemaError):
        option("not a name")


def test_initvar_classvar_and_init_false() -> None:
    @dataclass
    class C:
        x: int = 1
        scale: InitVar[int] = 2
        registry: ClassVar[int] = 3
        derived: int = field(default=0, init=False)

    fields = class_fields(C)
    assert list(fields) == ["x", "scale"]
    assert fields["scale"].initvar
    assert fields["scale"].annotation is int


def test_field_doc() -> None:
    assert class_fields(Config)["seed"].doc == "Random seed."
    assert class_fields(Config)["flag"].doc == ""


def test_field_doc_survives_redeclaration_in_a_subclass() -> None:
    # `default_res: int = 28` in the subclass has `doc=None`.
    assert Cmnist.__dataclass_fields__["default_res"].doc is None
    assert class_fields(Cmnist)["default_res"].doc == "Resolution of the images."
    assert class_fields(Cmnist)["root"].doc == "Where the datasets are stored."


def test_class_doc_is_the_first_paragraph() -> None:
    assert class_doc(Fcn) == "Fully-connected network."


def test_generated_class_docstring_is_ignored() -> None:
    assert class_doc(Cnn) == ""

    @dataclass
    class Sub(Fcn):
        pass

    # A subclass does not inherit the docstring of its base.
    assert class_doc(Sub) == ""
