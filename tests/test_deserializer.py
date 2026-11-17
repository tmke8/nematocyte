"""Building dataclass instances from the tree (`design/deserializer.md`)."""

from dataclasses import InitVar, dataclass, field, make_dataclass
import enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field
import pytest

import nematocyte
from nematocyte import ConfigError

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
        # The table in design/cli.md, against pydantic's lax mode.
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
        ("f='1e-3'", "f", 0.001),
        ("b='false'", "b", False),
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
        dm: (
            Annotated[Celeba, nematocyte.option("celeba")]
            | Annotated[Cmnist, nematocyte.option("cmnist")]
        )

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root)
    message = str(excinfo.value)
    assert "dm" in message
    assert "celeba" in message
    assert type(nematocyte.compose(Root, overrides=["dm=celeba"]).dm) is Celeba


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


def test_bool_is_accepted_for_int() -> None:
    # A known wart of lax mode, kept as it is.
    assert nematocyte.compose(Config, overrides=["seed=true"]).seed == 1


def test_unknown_key_in_a_nested_dataclass_value() -> None:
    @dataclass
    class Item:
        x: int = 0

    @dataclass
    class Root:
        items: list[Item] = field(default_factory=list)

    config = nematocyte.compose(Root, overrides=["items=[{x:1}]"])
    assert config.items == [Item(x=1)]
    with pytest.raises(ConfigError, match="items"):
        nematocyte.compose(Root, overrides=["items=[{y:1}]"])


def test_default_values_are_validated_too() -> None:
    # A default of the wrong type, as with `size: int = None`.
    root = make_dataclass("Root", [("size", int, field(default=None))])

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(root)
    message = str(excinfo.value)
    assert "size" in message
    assert "default" in message
