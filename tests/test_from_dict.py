"""Building the config object from plain dicts, such as `config.json` (`design/api.md`)."""

from dataclasses import InitVar, dataclass, field
from datetime import date, datetime
import enum
import json
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast

import pytest

import nematocyte
from nematocyte import ConfigError, SchemaError, option

from .helpers import Celeba, Cnn, Config, Fcn, LayerNorm, Opt, empty_config_dir


def saved_configs(cls: Any, args: list[str]) -> list[tuple[Any, dict[str, Any]]]:
    """Run a program; return what each job received, with its `config.json`."""
    received: list[Any] = []

    @nematocyte.main(cls, config_path=empty_config_dir())
    def main(config: Any) -> None:
        received.append(config)
        nematocyte.output_dir()

    main(args)
    paths = sorted(Path("outputs").glob("*/*/*/.nematocyte/config.json"))
    assert len(paths) == len(received)
    return [
        (config, json.loads(path.read_text()))
        for config, path in zip(received, paths, strict=True)
    ]


# -- the saved config of a run --------------------------------------------------


def test_saved_config_gives_the_object_of_the_run(config_dir: Path) -> None:
    received: list[Config] = []

    @nematocyte.main(Config, config_path=config_dir)
    def main(config: Config) -> None:
        received.append(config)
        nematocyte.output_dir()

    main(["--config-name", "experiments/cmnist_cnn", "model.norm=layernorm"])
    [path] = Path("outputs").glob("*/*/0/.nematocyte/config.json")
    config = nematocyte.from_dict(Config, json.loads(path.read_text()))
    assert config == received[0]
    assert isinstance(config.model, Cnn)
    assert isinstance(config.model.norm, LayerNorm)
    assert config.model.kernel_size == 3


def test_saved_configs_of_a_multirun(capsys: pytest.CaptureFixture[str]) -> None:
    pairs = saved_configs(Config, ["--multirun", "model=fcn,cnn", "seed=1,2"])
    capsys.readouterr()
    assert len(pairs) == 4
    for received, saved in pairs:
        assert nematocyte.from_dict(Config, saved) == received


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


@dataclass(kw_only=True)
class Values:
    path: Path = Path("/data")
    shape: tuple[int, int] = (2, 3)
    names: set[str] = field(default_factory=lambda: {"b", "a"})
    color: Color = Color.BLUE
    label_map: dict[int, int] = field(default_factory=lambda: {1: 2})
    day: date = date(2026, 11, 16)
    moment: datetime = datetime(2026, 11, 16, 16, 49, 10)
    lr: float = 1e-5
    maybe: int | None = None
    scale: InitVar[int] = 1
    scaled: int = field(init=False)

    def __post_init__(self, scale: int) -> None:
        self.scaled = 10 * scale


def test_values_survive_the_json_file() -> None:
    [(received, saved)] = saved_configs(Values, ["scale=3", "color=red"])
    config = nematocyte.from_dict(Values, saved)
    assert config == received
    # The value of an `InitVar` is in the saved config.
    assert saved["scale"] == 3
    assert config.scaled == 30
    assert config.color is Color.RED
    assert config.moment == datetime(2026, 11, 16, 16, 49, 10)


class Sgd(TypedDict):
    _option_: Literal["sgd"]
    momentum: float


class Adam(TypedDict):
    _option_: Literal["adam"]
    beta1: float
    eps: NotRequired[float]


class Plain(TypedDict):
    lr: float


class TypedConfig(TypedDict):
    seed: int
    optimizer: Sgd | Adam
    model: Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")]
    plain: Plain


def test_typeddicts() -> None:
    args = ["seed=1", "optimizer=adam", "optimizer.beta1=0.9", "model=cnn"]
    [(received, saved)] = saved_configs(TypedConfig, [*args, "plain.lr=0.1"])
    config = nematocyte.from_dict(TypedConfig, saved)
    assert config == received
    assert config["optimizer"] == {"_option_": "adam", "beta1": 0.9}


# -- plain dicts -------------------------------------------------------------------


def test_round_trip_through_to_dict(config_dir: Path) -> None:
    config = nematocyte.compose(Config, config_dir, config_name="good_run")
    assert nematocyte.from_dict(Config, nematocyte.to_dict(config)) == config


def test_left_out_keys_take_the_defaults_of_the_class() -> None:
    config = nematocyte.from_dict(Config, {"model": {"_option_": "cnn"}})
    assert config == Config(model=Cnn())
    assert nematocyte.from_dict(Config, {}) == Config()


def test_any_mapping_is_accepted() -> None:
    data = MappingProxyType(
        {"seed": 3, "dm": MappingProxyType({"_option_": "celeba", "download": True})}
    )
    config = nematocyte.from_dict(Config, data)
    assert config == Config(seed=3, dm=Celeba(download=True))


def test_option_key_is_kept_for_a_dataclass_option() -> None:
    # `to_dict` writes it into the same mapping as the fields.
    config = nematocyte.from_dict(
        Config, {"model": {"_option_": "fcn", "num_hidden": 2}}
    )
    assert config.model == Fcn(num_hidden=2)


def test_interpolations_are_not_resolved() -> None:
    config = nematocyte.from_dict(Config, {"wandb": {"group": "${seed}"}})
    assert config.wandb.group == "${seed}"


# -- errors ------------------------------------------------------------------------


def test_unknown_key() -> None:
    with pytest.raises(ConfigError) as info:
        nematocyte.from_dict(Config, {"opt": {"lrr": 0.1}})
    message = str(info.value)
    assert "opt.lrr" in message
    assert "Opt" in message
    assert "Did you mean 'lr'?" in message


def test_settings_are_not_part_of_the_dict() -> None:
    with pytest.raises(ConfigError, match="'nematocyte' is not a field of Config"):
        nematocyte.from_dict(Config, {"nematocyte": {"log": "debug"}})


def test_group_without_option_key() -> None:
    with pytest.raises(ConfigError) as info:
        nematocyte.from_dict(Config, {"model": {"num_hidden": 2}})
    message = str(info.value)
    assert "model" in message
    assert "'_option_'" in message
    assert "Config" in message
    assert "fcn, cnn" in message


def test_unknown_option() -> None:
    with pytest.raises(ConfigError) as info:
        nematocyte.from_dict(Config, {"model": {"_option_": "rnn"}})
    message = str(info.value)
    assert "model" in message
    assert "'rnn'" in message
    assert "Config" in message
    assert "fcn, cnn" in message


def test_option_key_in_a_plain_subconfig() -> None:
    with pytest.raises(ConfigError, match="opt._option_: '_option_' is not a field"):
        nematocyte.from_dict(Config, {"opt": {"_option_": "opt"}})


@pytest.mark.parametrize(("key", "what"), [("opt", "subconfig"), ("model", "group")])
def test_subconfig_that_is_not_a_mapping(key: str, what: str) -> None:
    with pytest.raises(ConfigError) as info:
        nematocyte.from_dict(Config, {key: "fcn"})
    message = str(info.value)
    assert f"{key}: this is a {what} of Config" in message
    assert "'fcn'" in message


def test_invalid_value() -> None:
    with pytest.raises(ConfigError) as info:
        nematocyte.from_dict(Config, {"opt": {"lr": "fast"}})
    message = str(info.value)
    assert "opt.lr" in message
    assert "Opt.lr (float)" in message
    assert "the dict given to from_dict()" in message


def test_post_init_runs() -> None:
    with pytest.raises(ConfigError, match="Invalid GPU index -2"):
        nematocyte.from_dict(Config, {"gpu": -2})


def test_errors_are_collected() -> None:
    # The errors in the structure come first, and then those in the values.
    data = {"seed": "x", "model": {"_option_": "rnn"}, "opt": {"lt": 1}, "wandb": 1}
    with pytest.raises(ConfigError) as info:
        nematocyte.from_dict(Config, data)
    message = str(info.value)
    for key in ("model", "opt.lt", "wandb"):
        assert f"\n{key}: " in message
    with pytest.raises(ConfigError) as info:
        nematocyte.from_dict(Config, {"seed": "x", "gpu": "y"})
    message = str(info.value)
    for key in ("seed", "gpu"):
        assert f"\n{key}: " in message


@dataclass
class NoDefault:
    model: Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")]


def test_left_out_group_without_a_default() -> None:
    with pytest.raises(ConfigError) as info:
        nematocyte.from_dict(NoDefault, {})
    message = str(info.value)
    assert "model" in message
    assert "NoDefault" in message
    assert "no default" in message
    assert nematocyte.from_dict(NoDefault, {"model": {"_option_": "cnn"}}) == NoDefault(
        model=Cnn()
    )


class Device:
    """A type that pydantic cannot read."""


DEVICE = Device()


@dataclass
class WithDevice:
    device: Device = DEVICE
    seed: int = 0


def test_value_of_a_type_that_pydantic_cannot_read() -> None:
    assert nematocyte.from_dict(WithDevice, {"seed": 1}).device is DEVICE
    with pytest.raises(ConfigError) as info:
        nematocyte.from_dict(WithDevice, {"device": "cpu"})
    message = str(info.value)
    assert "device" in message
    assert "WithDevice.device" in message
    assert "from_dict()" in message


def test_not_a_config_class() -> None:
    with pytest.raises(SchemaError):
        nematocyte.from_dict(cast("Any", Opt()), {})
