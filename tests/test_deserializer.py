"""Building dataclass instances from the tree (`design/deserializer.md`)."""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import (
    KW_ONLY,
    FrozenInstanceError,
    InitVar,
    dataclass,
    field,
    make_dataclass,
)
import datetime
from decimal import Decimal
import enum
import math
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, TypedDict
from typing_extensions import sentinel

from pydantic import AfterValidator, ConfigDict, Discriminator, Field, Tag, with_config
import pytest

import nematocyte
from nematocyte import ConfigError, SchemaError
from nematocyte._json_form import is_json_form

from .helpers import Celeba, Cmnist, Config, Fcn


def test_result_is_an_ordinary_instance() -> None:
    config = nematocyte.compose(Config)
    assert type(config) is Config
    assert type(config.dm) is Cmnist
    assert type(config.model) is Fcn
    assert config.dm.root == Path("/data")
    assert config.dm.in_dim == 28


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"


@dataclass
class Types:
    i: int = 0
    f: float = 0.0
    b: bool = False
    s: str = ""
    p: Path = Path()
    lit: Literal["a", "b"] = "a"
    opt_i: int | None = None
    ints: list[int] | None = None
    strs: list[str] | None = None
    int_map: dict[int, int] | None = None
    any_map: dict[str, Any] | None = None
    either: int | str = 0
    auto: float | Literal["auto"] = "auto"
    color: Color = Color.RED
    pair: tuple[int, str] = (0, "")
    prob: Annotated[float, Field(ge=0, le=1)] = 0.5


@pytest.mark.parametrize(
    ("override", "name", "expected"),
    [
        # The table in design/cli.md, against pydantic's strict mode on JSON.
        ("i=123", "i", 123),
        ("f=1e-3", "f", 0.001),
        ("f=1", "f", 1.0),
        ("b=true", "b", True),
        ("s=run1", "s", "run1"),
        ("opt_i=null", "opt_i", None),
        ("s='123'", "s", "123"),
        ("strs=[a,'1']", "strs", ["a", "1"]),
        ("p=/srv/data", "p", Path("/srv/data")),
        ("int_map={1:2}", "int_map", {1: 2}),
        (
            "any_map={eps:1e-8,amsgrad:true}",
            "any_map",
            {"eps": 1e-08, "amsgrad": True},
        ),
        # Further types from FEATURES.md and the comparison table.
        ("lit=b", "lit", "b"),
        ("ints=[1,2]", "ints", [1, 2]),
        ("either=abc", "either", "abc"),
        ("either=3", "either", 3),
        ("auto=0.5", "auto", 0.5),
        ("auto=auto", "auto", "auto"),
        ("color=green", "color", Color.GREEN),
        ("pair=[1,a]", "pair", (1, "a")),
        ("prob=1", "prob", 1.0),
    ],
)
def test_accepted_values(override: str, name: str, expected: Any) -> None:
    config = nematocyte.compose(Types, overrides=[override])
    value = getattr(config, name)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ("i=1.5", "integer"),
        ("s=123", "string"),
        ("strs=[a,1]", "string"),
        ("i=abc", "integer"),
        ("lit=c", "'a' or 'b'"),
        ("color=RED", "red"),
        ("prob=1.5", "less than or equal to 1"),
        ("ints=abc", "list"),
        ("opt_i=x", "integer"),
        # Strict validation through JSON (design/deserializer.md).
        ("i=true", "integer"),
        ("f=false", "number"),
        ("opt_i=true", "integer"),
        ("ints=[1,true]", "integer"),
        ("pair=[true,a]", "integer"),
        ("int_map={1:true}", "integer"),
        ("i='123'", "integer"),
        ("i=1.0", "integer"),
        ("b='false'", "boolean"),
        ("b=0", "boolean"),
        ("f='1e-3'", "number"),
    ],
)
def test_rejected_values(override: str, fragment: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Types, overrides=[override])
    message = str(excinfo.value)
    assert fragment in message
    assert override.split("=", maxsplit=1)[0] in message


def test_error_for_number_as_string_explains_the_quoting() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, overrides=["wandb.name=123"])
    message = str(excinfo.value)
    assert "wandb.name" in message
    assert "\"wandb.name='123'\"" in message


def test_error_for_invalid_value_includes_the_doc() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, overrides=["opt.lr=abc"])
    message = str(excinfo.value)
    assert "opt.lr" in message
    assert "'abc'" in message
    # The doc of the field, its class and its type.
    assert "Opt.lr (float). Learning rate." in message
    assert "opt.lr=abc" in message


def test_union_reports_every_member() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Types, overrides=["either=[1]"])
    message = str(excinfo.value)
    assert "integer" in message
    assert "string" in message


def test_all_errors_are_collected() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, overrides=["seed=abc", "opt.lr=x", "dm.val_prop=y"])
    message = str(excinfo.value)
    assert "seed" in message
    assert "opt.lr" in message
    assert "dm.val_prop" in message


def test_post_init_error_keeps_its_message() -> None:
    with pytest.raises(ConfigError, match="Invalid GPU index -2"):
        nematocyte.compose(Config, overrides=["gpu=-2"])


def test_post_init_error_in_a_subconfig_names_the_path() -> None:
    @dataclass
    class Sub:
        x: int = 1

        def __post_init__(self) -> None:
            if self.x < 0:
                raise ValueError("x must not be negative")

    @dataclass
    class Root:
        sub: Sub = field(default_factory=Sub)

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root, overrides=["sub.x=-1"])
    message = str(excinfo.value)
    assert "x must not be negative" in message
    assert "sub" in message
    assert "Sub" in message


def test_post_init_error_of_another_kind_keeps_its_message() -> None:
    # pydantic collects only a `ValueError` or an `AssertionError`; anything
    # else stops the validation, and its location is lost.
    @dataclass
    class Sub:
        x: int = 1

        def __post_init__(self) -> None:
            if self.x < 0:
                raise TypeError("x must not be negative")

    @dataclass
    class Root:
        sub: Sub = field(default_factory=Sub)

    with pytest.raises(ConfigError, match="TypeError: x must not be negative"):
        nematocyte.compose(Root, overrides=["sub.x=-1"])


def test_field_without_default_is_required() -> None:
    @dataclass
    class Root:
        name: str
        size: int = 1

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root)
    message = str(excinfo.value)
    assert "name" in message
    assert "Root" in message
    assert nematocyte.compose(Root, overrides=["name=x"]) == Root(name="x")


def test_triple_question_mark_is_an_ordinary_string(tmp_path: Path) -> None:
    @dataclass
    class Root:
        name: str = "x"

    (tmp_path / "primary.yaml").write_text("name: ???\n")
    config = nematocyte.compose(Root, tmp_path, config_name="primary")
    assert config.name == "???"


def test_group_without_default_needs_a_selection() -> None:
    @dataclass
    class Root:
        dm: Annotated[
            Annotated[Celeba, Tag("celeba")] | Annotated[Cmnist, Tag("cmnist")],
            Discriminator(nematocyte.pick),
        ]

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root)
    message = str(excinfo.value)
    assert "dm" in message
    assert "celeba" in message
    assert type(nematocyte.compose(Root, overrides=["dm=celeba"]).dm) is Celeba


def test_hand_written_init_is_not_called() -> None:
    # pydantic sets the fields and calls `__post_init__`, but not `__init__`.
    @dataclass
    class Root:
        lr: float = 0.1

        def __init__(self, lr: float = 0.1) -> None:
            self.lr = lr
            self.scaled = lr * 10

    config = nematocyte.compose(Root, overrides=["lr=0.5"])
    assert config.lr == 0.5
    assert not hasattr(config, "scaled")


def test_initvar_and_init_false_fields() -> None:
    @dataclass
    class Root:
        base: int = 1
        scale: InitVar[int] = 2
        scaled: int = field(default=0, init=False)

        def __post_init__(self, scale: int) -> None:
            self.scaled = self.base * scale

    assert nematocyte.compose(Root).scaled == 2
    assert nematocyte.compose(Root, overrides=["base=3", "scale=5"]).scaled == 15
    with pytest.raises(ConfigError, match="scaled"):
        nematocyte.compose(Root, overrides=["scaled=1"])


@dataclass(frozen=True, slots=True)
class FrozenSub:
    a: int = 1


@dataclass(kw_only=True, frozen=True)
class FrozenSgd:
    lr: float = 0.1


@dataclass(kw_only=True, frozen=True)
class FrozenAdam:
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(kw_only=True, frozen=True)
class FrozenRoot:
    seed: int = 0
    sub: FrozenSub = field(default_factory=FrozenSub)
    opt: Annotated[
        Annotated[FrozenSgd, Tag("sgd")] | Annotated[FrozenAdam, Tag("adam")],
        Discriminator(nematocyte.pick),
    ] = field(default_factory=FrozenSgd)
    doubled: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "doubled", 2 * self.seed)


def test_frozen_dataclasses(tmp_path: Path) -> None:
    (tmp_path / "primary.yaml").write_text("seed: 3\nsub:\n  a: 4\n")
    config = nematocyte.compose(
        FrozenRoot,
        tmp_path,
        config_name="primary",
        overrides=["sub.a=5", "opt=adam", "opt.betas=[0.5,0.6]"],
    )
    assert config == FrozenRoot(
        seed=3, sub=FrozenSub(5), opt=FrozenAdam(betas=(0.5, 0.6))
    )
    assert config.doubled == 6
    # The classes are used as they are, so the instances stay frozen.
    with pytest.raises(FrozenInstanceError):
        config.seed = 1  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        config.sub.a = 1  # ty: ignore[invalid-assignment]


@dataclass
class KwOnlySub:
    a: int
    _: KW_ONLY
    b: int = 2


@dataclass
class KwOnlyRoot:
    name: str
    _: KW_ONLY
    size: int = 1
    sub: KwOnlySub = field(default_factory=lambda: KwOnlySub(1))
    dm: Annotated[
        Annotated[Celeba, Tag("celeba")] | Annotated[Cmnist, Tag("cmnist")],
        Discriminator(nematocyte.pick),
    ] = field(default_factory=Cmnist)


def test_kw_only_marker(tmp_path: Path) -> None:
    # The `_: KW_ONLY` pseudo-field is not a key; the fields on either side
    # of it are (issue #18).
    (tmp_path / "primary.yaml").write_text("name: x\nsub:\n  a: 3\n")
    config = nematocyte.compose(
        KwOnlyRoot,
        tmp_path,
        config_name="primary",
        overrides=["size=4", "sub.b=5", "dm=celeba"],
    )
    assert config == KwOnlyRoot("x", size=4, sub=KwOnlySub(3, b=5), dm=Celeba())
    for override, cls in [("_=1", "KwOnlyRoot"), ("sub._=1", "KwOnlySub")]:
        with pytest.raises(ConfigError) as excinfo:
            nematocyte.compose(KwOnlyRoot, overrides=["name=x", override])
        message = str(excinfo.value)
        assert "'_'" in message
        assert cls in message


def test_bool_is_kept_where_the_type_allows_it() -> None:
    @dataclass
    class Root:
        flag: int | bool = 0
        anything: Any = None
        mixed: list[int | bool] = field(default_factory=list)

    config = nematocyte.compose(
        Root, overrides=["flag=true", "anything=true", "mixed=[1,true]"]
    )
    assert config.flag is True
    assert config.anything is True
    assert config.mixed == [1, True]
    assert config.mixed[1] is True


def test_value_changed_by_post_init_is_not_an_error() -> None:
    # The default check compares with the value before the constructor ran.
    @dataclass
    class Root:
        root: Path = Path("data")

        def __post_init__(self) -> None:
            self.root = self.root.resolve()

    config = nematocyte.compose(Root)
    assert config.root == Path("data").resolve()


def test_bool_default_for_a_number_is_rejected() -> None:
    @dataclass
    class Root:
        seed: int = True

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root)
    message = str(excinfo.value)
    assert "seed: Input should be a valid integer" in message
    assert "the default of Root.seed" in message


def test_unknown_key_in_a_dataclass_value_is_ignored() -> None:
    # Composition checks the keys of the config classes; inside a value,
    # the pydantic config of the class decides, and pydantic ignores them.
    @dataclass
    class Item:
        x: int = 0

    @dataclass
    class Root:
        items: list[Item] = field(default_factory=list)

    config = nematocyte.compose(Root, overrides=["items=[{x:1}]"])
    assert config.items == [Item(x=1)]
    assert nematocyte.compose(Root, overrides=["items=[{y:1}]"]).items == [Item()]


@with_config(ConfigDict(extra="forbid"))
@dataclass
class StrictItem:
    x: int = 0


@with_config(ConfigDict(extra="forbid"))
class StrictPoint(TypedDict):
    x: int


@dataclass
class WithStrictItems:
    items: list[StrictItem] = field(default_factory=list)
    points: list[StrictPoint] = field(default_factory=list)


@pytest.mark.parametrize(
    ("override", "where"),
    [("items=[{x:1},{y:1}]", "items: "), ("points=[{x:1,y:2}]", "points: ")],
)
def test_unknown_key_in_a_value_with_forbid(override: str, where: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(WithStrictItems, overrides=[override])
    message = str(excinfo.value)
    assert where in message
    assert "(at " in message
    assert "WithStrictItems." in message
    assert "the override" in message


def test_default_values_are_validated_too() -> None:
    # A default of the wrong type, as with `size: int = None`.
    root = make_dataclass("Root", [("size", int, field(default=None))])

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(root)
    message = str(excinfo.value)
    assert "size" in message
    assert "default" in message


# -- defaults that lose their type in JSON form -------------------------------


class Level(enum.IntEnum):
    LOW = 1


@dataclass
class Adam:
    lr: float = 0.1


@dataclass(eq=False)
class NoEq:
    lr: float = 0.1


TOKEN = object()


@pytest.mark.parametrize(
    ("annotation", "default", "where", "before", "after"),
    [
        (Any, Color.RED, "", "<Color.RED: 'red'>", "'red'"),
        (Color | str, Color.RED, "", "<Color.RED: 'red'>", "'red'"),
        (Any, Path("/tmp"), "", "PosixPath('/tmp')", "'/tmp'"),
        (Any, Adam(), "", "Adam(lr=0.1)", "{'lr': 0.1}"),
        (object, Path("/tmp"), "", "PosixPath('/tmp')", "'/tmp'"),
        (list[Any], [Path("/tmp")], " (at 0)", "PosixPath('/tmp')", "'/tmp'"),
        (dict[str, Any], {"a": Adam()}, " (at a)", "Adam(lr=0.1)", "{'lr': 0.1}"),
        (Any, ((1, Color.RED),), " (at 0.1)", "<Color.RED: 'red'>", "'red'"),
        (
            Any,
            datetime.date(2026, 1, 1),
            "",
            "datetime.date(2026, 1, 1)",
            "'2026-01-01'",
        ),
        (Level | int, Level.LOW, "", "<Level.LOW: 1>", "1"),
    ],
)
def test_default_that_loses_its_type_is_rejected(
    annotation: Any, default: object, where: str, before: str, after: str
) -> None:
    root = make_dataclass(
        "Root", [("value", annotation, field(default_factory=lambda: default))]
    )
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(root)
    message = str(excinfo.value)
    assert f"value: the default loses its type{where}:" in message
    assert f"{before} becomes {after}." in message
    assert "the default of Root.value" in message


@pytest.mark.parametrize(
    ("annotation", "default", "expected"),
    [
        # A config file cannot tell a tuple, list or set apart.
        (Any, (1, 2), [1, 2]),
        (Any, {3, 1}, [1, 3]),
        (Any, frozenset({(1, 2)}), [[1, 2]]),
        (Sequence[int], (1, 2), [1, 2]),
        (Sequence[Color], (Color.RED,), [Color.RED]),
        # The type says what to build, so nothing is lost.
        (Color, Color.RED, Color.RED),
        (Path, Path("/tmp"), Path("/tmp")),
        (tuple[int, ...], (1, 2), (1, 2)),
        (list[Adam], [Adam()], [Adam()]),
        (dict[Color, int], {Color.RED: 1}, {Color.RED: 1}),
        # Conversions that pydantic makes of the default itself.
        (float, 1, 1.0),
        (Path, "/tmp", Path("/tmp")),
        # Values in JSON form.
        (Any, {"a": [1, None]}, {"a": [1, None]}),
        # In strict mode on JSON, the first member that takes the string wins.
        (Path | str, Path("/tmp"), Path("/tmp")),
        (Path | Any, Path("/tmp"), Path("/tmp")),
        (datetime.date, datetime.date(2026, 1, 1), datetime.date(2026, 1, 1)),
    ],
)
def test_default_that_keeps_its_type_is_accepted(
    annotation: Any, default: object, expected: object
) -> None:
    root = make_dataclass(
        "Root", [("value", annotation, field(default_factory=lambda: default))]
    )
    value = nematocyte.compose(root).value
    assert value == expected
    assert type(value) is type(expected)


def test_default_in_json_form_is_validated_once() -> None:
    # A default of plain values loses nothing, so it is not checked again.
    calls: list[object] = []

    def count(value: object) -> object:
        calls.append(value)
        return value

    counted = Annotated[Any, AfterValidator(count)]
    root = make_dataclass(
        "Root",
        [
            (
                "value",
                counted,
                field(default_factory=lambda: {"a": (1, {3, 2.5}), "b": None}),
            )
        ],
    )
    assert nematocyte.compose(root).value == {"a": [1, [2.5, 3]], "b": None}
    assert len(calls) == 1


class Pair(NamedTuple):
    first: int
    second: int


@pytest.mark.parametrize(
    "value",
    [
        {"a": (1, {2.5, None}), "b": "x"},
        # No JSON form; the build reports it.
        [Decimal("0.5"), TOKEN],
    ],
)
def test_value_that_json_form_keeps_is_json_form(value: object) -> None:
    # `to_json_form` changes only the kinds of its containers.
    assert is_json_form(value)


@pytest.mark.parametrize(
    "value",
    [
        Color.RED,
        [Path("x")],
        {"opt": Adam()},
        OrderedDict(a=1),
        Pair(1, 2),
        [frozenset({Color.RED})],
        datetime.date(2026, 1, 1),
        {"at": datetime.time(12, 30)},
    ],
)
def test_value_that_json_form_converts_is_not_json_form(value: object) -> None:
    # Mirrors the branches of `to_json_form` that convert a value.
    assert not is_json_form(value)


def test_default_in_json_form_is_not_checked_even_where_the_union_differs() -> None:
    # The price of not checking defaults in JSON form (design/deserializer.md):
    # the default itself would stay a set of strings.
    root = make_dataclass(
        "Root",
        [("value", list[Path] | set[str], field(default_factory=lambda: {"x"}))],
    )
    assert nematocyte.compose(root).value == [Path("x")]


def test_set_default_in_json_form_for_a_list_is_accepted() -> None:
    # The JSON form of a set is sorted; the order of a set means nothing.
    root = make_dataclass(
        "Root", [("value", list[str], field(default_factory=lambda: {"b", "a"}))]
    )
    assert nematocyte.compose(root).value == ["a", "b"]


def test_named_tuple_default_is_checked() -> None:
    # A subclass of a plain container is not plain.
    class Pair(NamedTuple):
        color: Color
        size: int

    root = make_dataclass("Root", [("value", Any, field(default=Pair(Color.RED, 1)))])
    with pytest.raises(
        ConfigError, match=r"value: the default loses its type \(at 0\)"
    ):
        nematocyte.compose(root)


def test_dataclass_default_without_eq_keeps_its_type() -> None:
    # Compared field by field: a rebuilt instance is not `==` to it.
    root = make_dataclass(
        "Root", [("items", list[NoEq], field(default_factory=lambda: [NoEq()]))]
    )
    [item] = nematocyte.compose(root).items
    assert type(item) is NoEq


def test_nan_default_keeps_its_type() -> None:
    root = make_dataclass("Root", [("value", Any, field(default=float("nan")))])
    assert math.isnan(nematocyte.compose(root).value)


def test_default_that_loses_its_type_can_be_set(tmp_path: Path) -> None:
    # Only a default that nothing sets is checked.
    @dataclass
    class Root:
        value: Any = Color.RED

    (tmp_path / "primary.yaml").write_text("value: red\n")
    assert nematocyte.compose(Root, overrides=["value=blue"]).value == "blue"
    assert nematocyte.compose(Root, tmp_path, config_name="primary").value == "red"


def test_default_instance_that_loses_its_type_is_rejected() -> None:
    # The value comes from the default instance of a subconfig.
    @dataclass
    class Sub:
        color: Color | str = "red"

    @dataclass
    class Root:
        sub: Sub = field(default_factory=lambda: Sub(color=Color.GREEN))

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root)
    message = str(excinfo.value)
    assert "sub.color: the default loses its type" in message
    assert "Sub.color" in message


# -- types that pydantic cannot read -----------------------------------------


class Device:
    """A class that pydantic does not know, like `torch.device`."""


BN = sentinel("BN")
LN = sentinel("LN")


@dataclass
class WithDevice:
    device: Device = field(default_factory=Device, doc="Where to run.")
    seed: int = 0


@dataclass
class WithSentinel:
    norm: BN | LN | None = LN


@dataclass
class WithHiddenDevice:
    seed: int = 0
    _device: Device | None = field(default=None, init=False)


@pytest.mark.parametrize(
    ("cls", "where"),
    [
        (WithDevice, "WithDevice.device"),
        (WithSentinel, "WithSentinel.norm"),
        (WithHiddenDevice, "WithHiddenDevice._device"),
    ],
)
def test_type_pydantic_cannot_read_is_a_schema_error(cls: type, where: str) -> None:
    with pytest.raises(SchemaError) as excinfo:
        nematocyte.compose(cls)
    message = str(excinfo.value)
    assert where in message
    assert "property" in message


def test_init_false_field_of_a_type_pydantic_cannot_read_without_a_default() -> None:
    # pydantic leaves out a field with init=False and no default.
    @dataclass
    class Root:
        seed: int = 0
        _device: Device = field(init=False)

        def __post_init__(self) -> None:
            self._device = Device()

    assert isinstance(nematocyte.compose(Root)._device, Device)


@dataclass
class ArbitraryTypes:
    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    device: Device = field(default_factory=Device, doc="Where to run.")
    seed: int = 0


def test_default_without_json_form_is_an_error() -> None:
    # Even where the pydantic config of the class lets the type through.
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(ArbitraryTypes, overrides=["seed=1"])
    message = str(excinfo.value)
    assert "device: JSON has no form for" in message
    assert "(Device)" in message
    assert "the default of ArbitraryTypes.device" in message
    assert "ArbitraryTypes.device (Device). Where to run." in message
    assert "init=False" in message


@pytest.mark.parametrize(
    ("annotation", "default", "fragment"),
    [
        (Any, TOKEN, "(object)"),
        (Any, Decimal("0.5"), "(Decimal)"),
        (list[Any], [1, TOKEN], "(object) (at 1)"),
        (dict[Any, int], {(1, 2): 3}, "(tuple)"),
    ],
)
def test_value_without_json_form_is_an_error(
    annotation: Any, default: object, fragment: str
) -> None:
    root = make_dataclass(
        "Root", [("value", annotation, field(default_factory=lambda: default))]
    )
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(root)
    message = str(excinfo.value)
    assert "value: JSON has no form for" in message
    assert fragment in message
    assert "Root.value" in message


def test_value_from_a_file_without_json_form_is_an_error(tmp_path: Path) -> None:
    @dataclass
    class Root:
        blob: Any = None
        table: dict[Any, str] = field(default_factory=dict)

    (tmp_path / "c.yaml").write_text("blob: !!binary aGVsbG8=\ntable: {? [1, 2] : x}\n")
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root, tmp_path, config_name="c")
    message = str(excinfo.value)
    assert "blob: JSON has no form for b'hello' (bytes)" in message
    assert "table: JSON has no form for (1, 2) (tuple)" in message
    assert "c.yaml" in message
    assert "Root.blob" in message
    assert "Root.table" in message
