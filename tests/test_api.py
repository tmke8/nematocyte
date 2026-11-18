"""The public Python API (`design/api.md`)."""

from dataclasses import InitVar, dataclass, field
import enum
import importlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Annotated

import pytest

import nematocyte
from nematocyte import ConfigError, option

from .helpers import Cnn, Config, Fcn, empty_config_dir, write_files


def test_public_names() -> None:
    for name in (
        "main",
        "compose",
        "to_dict",
        "from_run_config",
        "run_config",
        "output_dir",
        "log_level",
        "configure_logging",
        "option",
        "validate_configs",
        "validate_config",
    ):
        assert name in nematocyte.__all__
        assert callable(getattr(nematocyte, name))


def test_compose_returns_an_instance_of_the_class(config_dir: Path) -> None:
    config = nematocyte.compose(Config, config_dir, overrides=["model=cnn", "seed=1"])
    assert isinstance(config, Config)
    assert config.model == Cnn()
    assert config.seed == 1


def test_compose_takes_a_config_name(config_dir: Path) -> None:
    config = nematocyte.compose(Config, config_dir, config_name="good_run")
    assert config.seed == 1


def test_config_path_is_relative_to_the_file_of_the_function(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    write_files(
        project,
        {
            "app.py": """\
from dataclasses import dataclass

import nematocyte

seen = []


@dataclass
class Config:
    seed: int = 0


@nematocyte.main(Config, config_path="conf", config_name="primary")
def main(config: Config) -> None:
    seen.append(config.seed)
""",
            "conf/primary.yaml": "seed: 11\n",
            "conf/other.yaml": "seed: 12\n",
        },
    )
    monkeypatch.syspath_prepend(str(project))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    app = importlib.import_module("app")

    # `config_name` names the default primary config ...
    app.main([])
    # ... which `--config-name` replaces.
    app.main(["--config-name", "other"])
    app.main(["--config-name", ""])
    assert app.seen == [11, 12, 0]


def test_entry_point_with_a_missing_config_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    @nematocyte.main(Config, config_path="no_such_directory")
    def main(config: Config) -> None:
        del config

    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 1
    assert "no_such_directory" in capsys.readouterr().err


def test_decorated_function_keeps_its_name() -> None:
    @nematocyte.main(Config, config_path=empty_config_dir())
    def my_entry_point(config: Config) -> float:
        """Docstring."""
        return float(config.seed)

    assert getattr(my_entry_point, "__name__", None) == "my_entry_point"
    assert my_entry_point.__doc__ == "Docstring."
    assert my_entry_point([]) is None


# -- to_dict -----------------------------------------------------------------


def test_to_dict_records_the_option_of_a_group() -> None:
    result = nematocyte.to_dict(nematocyte.compose(Config))
    assert result["model"] == {
        "_option_": "fcn",
        "num_hidden": 1,
        "hidden_dim": None,
        "activation": "gelu",
        "dropout_prob": 0.0,
    }
    # A plain subconfig gets no `_option_` key.
    assert result["opt"] == {"lr": 5e-05, "weight_decay": 0.0, "optimizer_kwargs": None}
    assert result["seed"] == 42
    assert list(result) == ["dm", "model", "opt", "wandb", "seed", "gpu", "flag"]


def test_to_dict_names_the_option_not_the_file(config_dir: Path) -> None:
    config = nematocyte.compose(
        Config, config_dir, overrides=["model=single_linear_layer"]
    )
    assert nematocyte.to_dict(config)["model"]["_option_"] == "fcn"


def test_to_dict_records_nested_groups(config_dir: Path) -> None:
    config = nematocyte.compose(Config, config_dir, overrides=["model=my_cnn"])
    assert nematocyte.to_dict(config)["model"] == {
        "_option_": "cnn",
        "kernel_size": 5,
        "stride": 1,
        "activation": "relu",
        "norm": {"_option_": "layernorm", "eps": 1e-05},
    }


def test_to_dict_works_on_any_instance() -> None:
    config = Config(model=Cnn(kernel_size=7), seed=3)
    result = nematocyte.to_dict(config)
    assert result["model"]["_option_"] == "cnn"
    assert result["model"]["kernel_size"] == 7


class Color(enum.Enum):
    RED = "red"


@dataclass
class Item:
    path: Path = Path("x")


@dataclass
class Values:
    path: Path = Path("/srv/data")
    pair: tuple[int, ...] = (1, 2)
    unique: set[int] = field(default_factory=lambda: {3, 1, 2})
    color: Color = Color.RED
    nested: dict[str, list[Path]] = field(default_factory=lambda: {"a": [Path("b")]})
    items: list[Item] = field(default_factory=lambda: [Item()])
    scale: InitVar[int] = 1
    derived: int = field(default=0, init=False)


def test_to_dict_converts_values_to_plain_types() -> None:
    result = nematocyte.to_dict(nematocyte.compose(Values))
    assert result == {
        "path": "/srv/data",
        "pair": [1, 2],
        "unique": [1, 2, 3],
        "color": "red",
        "nested": {"a": ["b"]},
        "items": [{"path": "x"}],
    }
    # The dict can always be dumped.
    json.dumps(result)


def test_to_dict_of_something_else_is_an_error() -> None:
    with pytest.raises(TypeError):
        nematocyte.to_dict({"a": 1})
    with pytest.raises(TypeError):
        nematocyte.to_dict(Config)


def test_to_dict_of_an_instance_that_is_no_option() -> None:
    @dataclass
    class Other:
        x: int = 1

    @dataclass
    class Root:
        model: Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")]

    with pytest.raises(ConfigError, match="Other"):
        nematocyte.to_dict(Root(model=Other()))  # ty: ignore[invalid-argument-type]


def test_importing_nematocyte_imports_no_optional_or_reference_package() -> None:
    # submitit and Optuna are imported only when their option is selected,
    # and Hydra, OmegaConf and PyYAML are never needed.
    code = (
        "import sys, nematocyte; "
        "banned = {'hydra', 'omegaconf', 'yaml', 'antlr4', 'submitit', 'optuna'}; "
        "print(sorted(banned & set(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_logging_is_not_set_up() -> None:
    root = logging.getLogger()
    before = (root.level, list(root.handlers))

    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:
        del config

    main([])
    assert (root.level, list(root.handlers)) == before
