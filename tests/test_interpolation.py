"""`${...}` in config values (`design/interpolation.md`)."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import Discriminator, Tag
import pytest

import nematocyte
from nematocyte import ConfigError
from nematocyte._interpolation import resolve

from .helpers import Config, compose_tree, write_files


@dataclass(kw_only=True)
class Small:
    default_res: int = 28


@dataclass(kw_only=True)
class Large:
    default_res: int = 224


@dataclass
class Model:
    hidden_dim: int = 1
    name: str = "model"


@dataclass(kw_only=True)
class Root:
    dm: Annotated[
        Annotated[Small, Tag("small")] | Annotated[Large, Tag("large")],
        Discriminator(nematocyte.pick),
    ] = field(default_factory=Small)
    model: Model = field(default_factory=Model)
    seed: int = 42
    lr: float = 5e-05
    tag: str | None = None
    tags: list[str] | None = None
    extra: dict[str, int | str] | None = None


def test_reference_keeps_the_type_of_its_target() -> None:
    config = nematocyte.compose(
        Root, overrides=["model.hidden_dim=${dm.default_res}", "dm.default_res=64"]
    )
    assert config.model.hidden_dim == 64
    assert isinstance(config.model.hidden_dim, int)


def test_reference_inside_a_string_is_formatted_into_it() -> None:
    config = nematocyte.compose(Root, overrides=["tag=seed${seed}-lr${lr}", "seed=7"])
    assert config.tag == "seed7-lr5e-05"


def test_reference_to_a_value_that_is_only_a_default() -> None:
    # Neither `seed` nor `dm.default_res` is in the raw tree.
    config = nematocyte.compose(
        Root, overrides=["tag=s${seed}", "model.hidden_dim=${dm.default_res}"]
    )
    assert config.tag == "s42"
    assert config.model.hidden_dim == 28


def test_reference_follows_the_selected_option() -> None:
    config = nematocyte.compose(
        Root, overrides=["model.hidden_dim=${dm.default_res}", "dm=large"]
    )
    assert config.model.hidden_dim == 224


def test_reference_picks_up_a_command_line_override(tmp_path: Path) -> None:
    directory = write_files(
        tmp_path / "c", {"primary.yaml": "seed: 1\ntag: run-${seed}\n"}
    )
    config = nematocyte.compose(
        Root, directory, config_name="primary", overrides=["seed=5"]
    )
    assert config.tag == "run-5"


def test_date_from_yaml_is_formatted_as_its_iso_string(tmp_path: Path) -> None:
    # The same text as the value itself arrives as, with a `T` and not the
    # space that `str()` of a datetime gives.
    directory = write_files(
        tmp_path / "c",
        {"primary.yaml": "extra: {at: 2026-11-16 09:30:00}\ntag: at ${extra.at}\n"},
    )
    config = nematocyte.compose(Root, directory, config_name="primary")
    assert config.extra == {"at": "2026-11-16T09:30:00"}
    assert config.tag == "at 2026-11-16T09:30:00"


def test_chain_of_references() -> None:
    config = nematocyte.compose(
        Root, overrides=["tag=${model.name}", "model.name=x${seed}"]
    )
    assert config.tag == "x42"


def test_references_in_lists_and_dicts() -> None:
    config = nematocyte.compose(
        Root, overrides=["tags=[a,s${seed}]", "extra={res:${dm.default_res}}"]
    )
    assert config.tags == ["a", "s42"]
    assert config.extra == {"res": 28}


def test_interpolated_value_is_checked_against_the_field() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root, overrides=["seed=${model.name}"])
    assert "seed" in str(excinfo.value)
    assert "integer" in str(excinfo.value)


def test_now_resolver() -> None:
    root = compose_tree(Root, overrides=["tag=${now:%Y-%m-%d}-cnn-dropout"])
    resolve(root, datetime(2026, 11, 16, 16, 49, 10))
    assert root.values["tag"] == "2026-11-16-cnn-dropout"


def test_now_is_computed_once() -> None:
    root = compose_tree(
        Root, overrides=["tag=${now:%H-%M-%S.%f}", "model.name=${now:%H-%M-%S.%f}"]
    )
    resolve(root, datetime.now())
    assert root.values["tag"] == root.children["model"].values["name"]


def test_now_in_a_config_file(config_dir: Path) -> None:
    write_files(config_dir, {"dated.yaml": "wandb:\n  group: ${now:%Y}-x\n"})
    config = nematocyte.compose(Config, config_dir, config_name="dated")
    assert config.wandb.group == f"{datetime.now():%Y}-x"


def test_cycle_is_detected() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root, overrides=["tag=${model.name}", "model.name=${tag}"])
    message = str(excinfo.value)
    assert "cycle" in message
    assert "tag" in message
    assert "model.name" in message


def test_unknown_target() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root, overrides=["tag=${model.nope}"])
    message = str(excinfo.value)
    assert "${model.nope}" in message
    assert "'nope'" in message
    assert "Model" in message
    assert "tag" in message


def test_target_that_is_a_subconfig() -> None:
    with pytest.raises(ConfigError, match="subconfig"):
        nematocyte.compose(Root, overrides=["tag=${model}"])


def test_relative_reference_is_not_supported() -> None:
    with pytest.raises(ConfigError, match="relative references are not supported"):
        nematocyte.compose(Root, overrides=["model.name=${.hidden_dim}"])


def test_nested_interpolation_is_not_supported(tmp_path: Path) -> None:
    directory = write_files(tmp_path / "c", {"primary.yaml": "tag: ${model.${seed}}\n"})
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root, directory, config_name="primary")
    assert "not supported" in str(excinfo.value)
    assert "primary.yaml" in str(excinfo.value)


def test_only_the_now_resolver_exists() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Root, overrides=["tag=${oc.env:HOME}"])
    assert "'now'" in str(excinfo.value)


def test_interpolation_in_a_dataclass_default_is_not_resolved() -> None:
    @dataclass
    class WithDefault:
        seed: int = 1
        name: str = "${seed}"

    assert nematocyte.compose(WithDefault).name == "${seed}"


def test_reference_into_a_dict_value() -> None:
    config = nematocyte.compose(
        Root, overrides=["extra={res:64}", "model.hidden_dim=${extra.res}"]
    )
    assert config.model.hidden_dim == 64
    with pytest.raises(ConfigError, match="'nope'"):
        nematocyte.compose(Root, overrides=["extra={res:64}", "seed=${extra.nope}"])
