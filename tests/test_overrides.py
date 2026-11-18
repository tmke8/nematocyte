"""The override parser (`design/cli.md`)."""

import math
from typing import Any

import pytest

from nematocyte import ConfigError
from nematocyte._overrides import (
    AddedConfig,
    ChoiceSweep,
    IntervalSweep,
    RangeSweep,
    format_value,
    parse_argument,
    parse_override,
    parse_value,
    split_arguments,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The table of parsed values in design/cli.md.
        ("seed=123", 123),
        ("seed=1_000", 1000),
        ("opt.lr=1e-3", 0.001),
        ("opt.lr=.5", 0.5),
        ("flag=true", True),
        ("flag=TRUE", True),
        ("flag=yes", "yes"),
        ("flag=on", "on"),
        ("wandb.name=null", None),
        ("wandb.name='null'", "null"),
        ('wandb.name="123"', "123"),
        ("wandb.name=my run", "my run"),
        ("dm.root=/srv/data", "/srv/data"),
        ("seed=007", "007"),
        ("wandb.name=", ""),
        ("wandb.tags=[a,b]", ["a", "b"]),
        ("dm.colors=[1,[2,3]]", [1, [2, 3]]),
        ("dm.label_map={1:2}", {1: 2}),
        ("x={a:{b:c}}", {"a": {"b": "c"}}),
        ("wandb.group=${now:%Y-%m-%d}-x", "${now:%Y-%m-%d}-x"),
        # Where the grammar differs from YAML.
        ("x=2026-11-16", "2026-11-16"),
        ("x=1:30", "1:30"),
        # Further cases, checked against Hydra's parser.
        ("x=-1", -1),
        ("x=+1", 1),
        ("x=1.", 1.0),
        ("x=1.5e3", 1500.0),
        ("x=0x10", "0x10"),
        ("x=1__0", "1__0"),
        ("x=01", "01"),
        ("x=0.1.2", "0.1.2"),
        ("x=[ a , b ]", ["a", "b"]),
        ("x={ a : 1 , b : [1, 2] }", {"a": 1, "b": [1, 2]}),
        ("x=a\\,b", "a,b"),
        ("x=\\[a\\]", "[a]"),
        ("x='a\\'b'", "a'b"),
        ("x='a,b'", "a,b"),
        ("x=[]", []),
        ("x={}", {}),
        ("x= 5", 5),
        ("x=5 ", 5),
        ("x=tRue", True),
        ("x=NuLL", None),
        ("x=a/b@c", "a/b@c"),
        ("x=[${a},1]", ["${a}", 1]),
        ("x=a\\=b", "a=b"),
        ("x=3-4", "3-4"),
        ("x={a:b:c}", {"a": "b:c"}),
        ("x={a:1,b:[x,{c:null}]}", {"a": 1, "b": ["x", {"c": None}]}),
    ],
)
def test_values(text: str, expected: Any) -> None:
    override = parse_override(text)
    assert override.value == expected
    assert type(override.value) is type(expected)
    assert override.text == text
    assert not override.is_sweep


def test_inf_and_nan_are_floats() -> None:
    assert parse_override("opt.lr=inf").value == math.inf
    assert parse_override("opt.lr=-inf").value == -math.inf
    assert math.isnan(parse_override("opt.lr=nan").value)


def test_key_is_split_at_dots() -> None:
    override = parse_override("model.norm.momentum=0.2")
    assert override.key == ("model", "norm", "momentum")
    assert override.key_text == "model.norm.momentum"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("seed=42,43", ChoiceSweep((42, 43))),
        ("seed=choice(1,2)", ChoiceSweep((1, 2))),
        ("x=choice(a)", ChoiceSweep(("a",))),
        ("x=a b , c", ChoiceSweep(("a b", "c"))),
        ("x=1,[2,3]", ChoiceSweep((1, [2, 3]))),
        ("seed=range(1,4)", RangeSweep(1, 4)),
        ("seed=range(4)", RangeSweep(0, 4)),
        ("seed=range(1,10,2)", RangeSweep(1, 10, 2)),
        ("seed=range(start=1,stop=4)", RangeSweep(1, 4)),
        ("opt.lr=interval(1e-5,1e-3)", IntervalSweep(1e-5, 1e-3)),
        (
            "opt.lr=tag(log, interval(1e-5, 1e-3))",
            IntervalSweep(1e-5, 1e-3, frozenset({"log"})),
        ),
        ("x=tag(a,b,choice(1,2))", ChoiceSweep((1, 2), frozenset({"a", "b"}))),
    ],
)
def test_sweeps(text: str, expected: Any) -> None:
    override = parse_override(text)
    assert override.value == expected
    assert override.is_sweep


def test_range_values() -> None:
    assert RangeSweep(1, 4).values() == [1, 2, 3]
    assert RangeSweep(0, 1, 0.25).values() == [0.0, 0.25, 0.5, 0.75]
    assert RangeSweep(0, 0.3, 0.1).values() == [0.0, 0.1, 0.2]


def test_interval_keeps_integers() -> None:
    sweep = parse_value("interval(1, 12)")
    assert isinstance(sweep, IntervalSweep)
    assert isinstance(sweep.start, int)
    assert isinstance(sweep.end, int)


@pytest.mark.parametrize(
    "text",
    [
        # The overrides that nematocyte removes.
        "+seed=1",
        "++seed=1",
        "~seed",
        "~seed=1",
        # Grammar that is left out.
        "model@pkg=cnn",
        "hydra/launcher=x",
        "a[0]=1",
        "x=int(1)",
        "x=float(1)",
        "x=str(1)",
        "x=glob(*)",
        "x=sort(1,2)",
        "x=shuffle(1,2)",
        "x=extend_list(1)",
        # What Hydra rejects stays rejected.
        "dm.root=~/data",
        "x=ü",
        "x=a#b",
        "x=a&b",
        "x=a!b",
        "x=a=b",
        "x=a'b'",
        "x='a'b",
        "x=(1)",
        "x={'a b':1}",
        "x=[a,",
        "x={a:1",
        "x='abc",
        "x=1,2,",
        "x=[a,,b]",
        "x=choice(1,choice(2,3))",
        "x=[choice(1,2)]",
        "x=tag(log)",
        "x=range(1,2,0)",
        "x=interval(a,b)",
        "seed",
        "=1",
        "1a=2",
    ],
)
def test_rejected(text: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_override(text)
    assert text in str(excinfo.value)


# -- +NAME ---------------------------------------------------------------------


@pytest.mark.parametrize("text", ["+presets/debug", "+good_run", "+presets/debug.yaml"])
def test_plus_names_a_primary_config(text: str) -> None:
    assert parse_argument(text) == AddedConfig(text[1:], text)


def test_other_arguments_are_overrides() -> None:
    assert parse_argument("seed=1") == parse_override("seed=1")


@pytest.mark.parametrize(
    ("text", "added", "key"),
    [
        ("+presets=debug", "+presets/debug", "presets=debug"),
        (
            "+opt=weight_decay_adamw",
            "+opt/weight_decay_adamw",
            "opt=weight_decay_adamw",
        ),
        ("+a.b=c", "+a/b/c", "a.b=c"),
    ],
)
def test_plus_with_equals_shows_both_forms(text: str, added: str, key: str) -> None:
    # Hydra's `+presets=debug`: the name of the file is written without `=`.
    with pytest.raises(ConfigError) as excinfo:
        parse_argument(text)
    message = str(excinfo.value)
    assert repr(text) in message
    assert added in message
    assert f"{key}" in message


@pytest.mark.parametrize("text", ["+", "++presets/debug", "+ "])
def test_plus_needs_a_name(text: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_argument(text)
    assert repr(text) in str(excinfo.value)


def test_split_arguments_keeps_the_order_of_added_configs() -> None:
    added, overrides = split_arguments(
        ["+b", "seed=1", "+a", "+b", "seed=2", "model=cnn"]
    )
    assert [config.name for config in added] == ["a", "b"]
    assert [override.text for override in overrides] == ["seed=2", "model=cnn"]


def test_error_for_slash_shows_the_dotted_key() -> None:
    with pytest.raises(ConfigError, match=r"nematocyte\.launcher"):
        parse_override("nematocyte/launcher=slurm")


def test_error_for_tilde_says_to_quote() -> None:
    with pytest.raises(ConfigError, match="quoted"):
        parse_override("dm.root=~/data")


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        1,
        -3,
        0.001,
        5e-05,
        "abc",
        "my run",
        "",
        "123",
        "null",
        "true",
        "1e-3",
        "a,b",
        "it's",
        "/srv/data",
        "~/data",
        "über",
        ["a", 1, None],
        {"eps": 1e-8, "amsgrad": True},
        {1: [2, {"a": "b c"}]},
    ],
)
def test_format_value_round_trips(value: Any) -> None:
    text = format_value(value)
    parsed = parse_value(text)
    assert parsed == value
    assert type(parsed) is type(value)


def test_format_value_of_plain_strings_is_unquoted() -> None:
    assert format_value("gelu") == "gelu"
    assert format_value("/srv/data") == "/srv/data"
    assert format_value("123") == "'123'"
