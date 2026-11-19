"""A `frozendict` as the default of a TypedDict field, which is experimental.

See `design/deserializer.md`, "Experimental: a `frozendict` as the default of
a TypedDict field". `frozendict` is new in Python 3.15.
"""

from dataclasses import dataclass
import sys
from typing import NotRequired, TypedDict, cast

import pytest

import nematocyte
from nematocyte import ConfigError, SchemaError
from nematocyte._plain import to_plain

from .test_typeddict import Opt, Optimizer, Sgd, _texts, compose_root

# The tests are in a version check, which every type checker understands, so that
# the name `frozendict` is not checked against Python 3.14.
if sys.version_info < (3, 15):
    pytest.skip("frozendict is new in Python 3.15", allow_module_level=True)
else:
    from builtins import frozendict

    def test_frozendict_default_of_a_typeddict_subconfig() -> None:
        @dataclass
        class Config:
            opt: Opt = cast("Opt", frozendict(lr=0.1))

        config = compose_root(Config)
        assert config.opt == {"lr": 0.1}
        # The built node is a plain dict.
        assert type(config.opt) is dict
        config = compose_root(Config, overrides=["opt.lr=0.2", "opt.wd=0.5"])
        assert config.opt == {"lr": 0.2, "wd": 0.5}
        assert type(config.opt) is dict

    def test_key_left_out_of_a_frozendict_default_is_not_set() -> None:
        @dataclass
        class Config:
            opt: Opt = cast("Opt", frozendict(wd=0.5))

        with pytest.raises(ConfigError) as info:
            compose_root(Config)
        message = str(info.value)
        assert "opt.lr: a value is required" in message
        assert "Opt" in message

    def test_frozendict_default_of_a_group_of_typeddicts_names_its_option() -> None:
        @dataclass
        class Config:
            optimizer: Optimizer = cast("Sgd", frozendict(_option_="sgd", momentum=0.9))

        config = compose_root(Config)
        assert config.optimizer == {"_option_": "sgd", "momentum": 0.9}
        assert type(config.optimizer) is dict
        config = compose_root(
            Config, overrides=["optimizer=adam", "optimizer.beta1=0.8"]
        )
        assert config.optimizer == {"_option_": "adam", "beta1": 0.8}

    def test_frozendict_default_of_a_group_of_typeddicts_without_its_option() -> None:
        @dataclass
        class Config:
            optimizer: Optimizer = cast("Sgd", frozendict(momentum=0.9))

        with pytest.raises(SchemaError) as info:
            compose_root(Config)
        message = str(info.value)
        assert "Config.optimizer" in message
        assert "'_option_'" in message
        assert "sgd, adam" in message

    class Schedule(TypedDict):
        steps: int

    class Training(TypedDict):
        lr: float
        schedule: Schedule
        weights: NotRequired[dict[str, float]]

    @dataclass
    class Nested:
        training: Training = cast(
            "Training",
            frozendict(
                lr=0.1, schedule=frozendict(steps=3), weights=frozendict(a=1.0, b=2.0)
            ),
        )

    def test_frozendicts_inside_a_frozendict_default() -> None:
        """For a nested TypedDict, and as the value of a key typed as a dict."""
        config = compose_root(Nested)
        assert config.training == {
            "lr": 0.1,
            "schedule": {"steps": 3},
            "weights": {"a": 1.0, "b": 2.0},
        }
        assert type(config.training["schedule"]) is dict
        assert type(config.training.get("weights")) is dict
        config = compose_root(Nested, overrides=["training.schedule.steps=5"])
        assert config.training["schedule"] == {"steps": 5}

    def test_cfg_with_frozendict_values() -> None:
        _, cfg_text = _texts(Nested, ())
        assert "frozendict" not in cfg_text
        assert "  weights:\n    a: 1.0\n    b: 2.0\n" in cfg_text
        assert "  schedule:\n    # (int)\n    steps: 3\n" in cfg_text

    def test_to_plain_turns_a_frozendict_into_a_dict() -> None:
        plain = to_plain(frozendict(a=frozendict(b=(1, 2))))
        assert plain == {"a": {"b": [1, 2]}}
        assert type(plain) is dict
        assert type(plain["a"]) is dict

    def test_to_dict_with_a_frozendict_default() -> None:
        @dataclass
        class Config:
            opt: Opt = cast("Opt", frozendict(lr=0.1))
            optimizer: Optimizer = cast("Sgd", frozendict(_option_="sgd", momentum=0.9))

        assert nematocyte.to_dict(compose_root(Config)) == {
            "opt": {"lr": 0.1},
            "optimizer": {"_option_": "sgd", "momentum": 0.9},
        }
