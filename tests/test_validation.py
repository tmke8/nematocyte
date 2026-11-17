"""Checking every config file in a test (`design/validation.md`)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import pytest

import nematocyte
from nematocyte import ConfigError, option

from .helpers import CONFIG_FILES, Config, write_files


def errors(config_dir: Path, *, construct: bool = True) -> dict[str, str]:
    """The files with an error, by their path."""
    results = nematocyte.validate_configs(Config, config_dir, construct=construct)
    return {
        result.path.as_posix(): str(result.error)
        for result in results
        if result.error is not None
    }


def test_one_result_per_file(config_dir: Path) -> None:
    results = nematocyte.validate_configs(Config, config_dir)
    assert sorted(result.path.as_posix() for result in results) == sorted(CONFIG_FILES)
    assert all(result.ok for result in results)
    assert all(result.error is None for result in results)


@pytest.mark.parametrize("path", sorted(CONFIG_FILES))
def test_single_file(config_dir: Path, path: str) -> None:
    # As in the example: one test case per file, with a relative path.
    result = nematocyte.validate_config(Config, config_dir, Path(path))
    assert result.path == Path(path)
    if result.error is not None:
        raise result.error


def test_absolute_path(config_dir: Path) -> None:
    result = nematocyte.validate_config(
        Config, config_dir, config_dir / "good_run.yaml"
    )
    assert result.path == Path("good_run.yaml")
    assert result.ok


def test_other_files_are_ignored(config_dir: Path) -> None:
    write_files(config_dir, {"README.md": "# Configs\n", "dm/notes.txt": "x"})
    assert errors(config_dir) == {}


def test_broken_primary_config(config_dir: Path) -> None:
    write_files(config_dir, {"experiments/broken.yaml": "model:\n  num_hiden: 1\n"})
    found = errors(config_dir)
    assert list(found) == ["experiments/broken.yaml"]
    assert "'num_hiden'" in found["experiments/broken.yaml"]
    assert "Fcn" in found["experiments/broken.yaml"]


def test_broken_named_config_for_a_group(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "dm/renamed.yaml": "defaults:\n  _self_: celeba\nsuperclas: Smiling\n",
            "dm/bad_value.yaml": "defaults:\n  _self_: celeba\nsuperclass: Smilin\n",
            "dm/no_self.yaml": "superclass: Smiling\n",
            "dm/bad_chain.yaml": "defaults:\n  _self_: nope\n",
        },
    )
    found = errors(config_dir)
    assert sorted(found) == [
        "dm/bad_chain.yaml",
        "dm/bad_value.yaml",
        "dm/no_self.yaml",
        "dm/renamed.yaml",
    ]
    assert "'superclas'" in found["dm/renamed.yaml"]
    assert "Celeba" in found["dm/renamed.yaml"]
    assert "'Smilin'" in found["dm/bad_value.yaml"]
    assert "_self_" in found["dm/no_self.yaml"]
    assert "'nope'" in found["dm/bad_chain.yaml"]


def test_broken_preset(config_dir: Path) -> None:
    write_files(config_dir, {"opt/typo.yaml": "learning_rate: 1\n"})
    found = errors(config_dir)
    assert list(found) == ["opt/typo.yaml"]
    assert "'learning_rate'" in found["opt/typo.yaml"]
    assert "Opt" in found["opt/typo.yaml"]


def test_files_under_the_reserved_key(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "nematocyte/launcher/bad.yaml": (
                "defaults:\n  _self_: submitit_slurm\npartitoin: x\n"
            ),
        },
    )
    found = errors(config_dir)
    assert list(found) == ["nematocyte/launcher/bad.yaml"]
    assert "'partitoin'" in found["nematocyte/launcher/bad.yaml"]
    assert "SubmititSlurmLauncher" in found["nematocyte/launcher/bad.yaml"]


def test_named_config_for_a_nested_field_of_another_option(config_dir: Path) -> None:
    # `norm` is a field of `cnn` only, and the default option of `model` is `fcn`.
    result = nematocyte.validate_config(
        Config, config_dir, "model/norm/my_batchnorm.yaml"
    )
    assert result.ok
    write_files(
        config_dir,
        {"model/norm/bad.yaml": "defaults:\n  _self_: batchnorm\nmomentun: 1\n"},
    )
    found = errors(config_dir)
    assert list(found) == ["model/norm/bad.yaml"]
    assert "BatchNorm" in found["model/norm/bad.yaml"]


def test_files_that_belong_nowhere(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            # A directory named after a field that is a value.
            "seed/one.yaml": "seed: 1\n",
            # A subdirectory of a group's directory that matches no field.
            "model/experiments/x.yaml": "seed: 1\n",
        },
    )
    found = errors(config_dir)
    assert sorted(found) == ["model/experiments/x.yaml", "seed/one.yaml"]
    assert "belongs nowhere" in found["seed/one.yaml"]
    assert "'seed'" in found["seed/one.yaml"]
    assert "belongs nowhere" in found["model/experiments/x.yaml"]
    assert "'experiments'" in found["model/experiments/x.yaml"]


def test_directory_that_matches_no_field_holds_primary_configs(
    config_dir: Path,
) -> None:
    write_files(config_dir, {"experiments/deep/er.yaml": "seed: 3\n"})
    assert errors(config_dir) == {}


def test_named_config_is_validated_in_the_whole_tree(config_dir: Path) -> None:
    # An interpolation can refer to a value outside the node of the file.
    write_files(
        config_dir,
        {
            "dm/from_seed.yaml": "defaults:\n  _self_: celeba\ndefault_res: ${seed}\n",
            "dm/bad_ref.yaml": "defaults:\n  _self_: celeba\ndefault_res: ${sed}\n",
        },
    )
    found = errors(config_dir)
    assert list(found) == ["dm/bad_ref.yaml"]
    assert "${sed}" in found["dm/bad_ref.yaml"]


def test_post_init_runs(config_dir: Path) -> None:
    write_files(config_dir, {"bad_gpu.yaml": "gpu: -5\n"})
    found = errors(config_dir)
    assert list(found) == ["bad_gpu.yaml"]
    assert "Invalid GPU index -5" in found["bad_gpu.yaml"]


def test_construct_false_skips_post_init_only(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "bad_gpu.yaml": "gpu: -5\n",
            "bad_type.yaml": "gpu: minus-five\n",
            "bad_key.yaml": "gppu: 1\n",
        },
    )
    found = errors(config_dir, construct=False)
    assert sorted(found) == ["bad_key.yaml", "bad_type.yaml"]


@dataclass
class Sub:
    x: int = 1


@dataclass
class NeedsName:
    name: str
    sub: Sub = field(default_factory=Sub)


def test_field_without_default_needs_a_primary_config(tmp_path: Path) -> None:
    directory = write_files(
        tmp_path / "c",
        {"base.yaml": "name: test\n", "sub/big.yaml": "x: 100\n"},
    )
    results = nematocyte.validate_configs(NeedsName, directory)
    assert [result.ok for result in results] == [True, False]
    assert "name" in str(results[1].error)
    # The counterpart of `config_with_defaults` in hydra-qol.
    results = nematocyte.validate_configs(NeedsName, directory, config_name="base")
    assert all(result.ok for result in results)


def test_error_is_a_config_error(config_dir: Path) -> None:
    write_files(config_dir, {"bad.yaml": "nope: 1\n"})
    result = nematocyte.validate_config(Config, config_dir, "bad.yaml")
    assert isinstance(result.error, ConfigError)
    assert not result.ok


def test_missing_file(config_dir: Path) -> None:
    result = nematocyte.validate_config(Config, config_dir, "nope.yaml")
    assert "nope.yaml" in str(result.error)


# -- sweeper parameters -------------------------------------------------------


def sweeper_file(params: str) -> dict[str, str]:
    return {
        "nematocyte/sweeper/s.yaml": f"defaults:\n  _self_: optuna\nparams:\n{params}"
    }


def test_sweeper_params_are_parsed(config_dir: Path) -> None:
    write_files(config_dir, sweeper_file("  opt.lr: interval(1e-5,\n"))
    found = errors(config_dir)
    assert list(found) == ["nematocyte/sweeper/s.yaml"]
    assert "opt.lr" in found["nematocyte/sweeper/s.yaml"]


def test_sweeper_params_have_to_name_values(config_dir: Path) -> None:
    write_files(config_dir, sweeper_file("  opt.learning_rate: interval(0, 1)\n"))
    found = errors(config_dir)
    assert list(found) == ["nematocyte/sweeper/s.yaml"]
    assert "'opt.learning_rate'" in found["nematocyte/sweeper/s.yaml"]


def test_sweeper_param_of_one_option_is_enough(config_dir: Path) -> None:
    # `kernel_size` exists only for `cnn`, and `num_hidden` only for `fcn`. The
    # file does not say which option of `model` will be selected.
    write_files(
        config_dir,
        sweeper_file(
            "  model.kernel_size: range(3, 6)\n"
            "  model.num_hidden: range(1, 4)\n"
            "  model.norm.momentum: interval(0.1, 0.9)\n"
        ),
    )
    assert errors(config_dir) == {}


def test_basic_sweeper_params_are_checked_too(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "nematocyte/sweeper/seeds.yaml": (
                "defaults:\n  _self_: basic\nparams:\n  sed: 42,43\n"
            ),
        },
    )
    found = errors(config_dir)
    assert list(found) == ["nematocyte/sweeper/seeds.yaml"]


def test_group_through_union_alias(tmp_path: Path) -> None:
    @dataclass
    class A:
        x: int = 1

    @dataclass
    class B:
        y: int = 1

    @dataclass
    class Root:
        part: Annotated[A, option("a")] | Annotated[B, option("b")]

    directory = write_files(
        tmp_path / "c",
        {
            "part/big_a.yaml": "defaults:\n  _self_: a\nx: 9\n",
            "part/big_b.yaml": "defaults:\n  _self_: b\ny: 9\n",
            "primary.yaml": "defaults:\n  part: big_a\n",
        },
    )
    # The group has no default, but a named config for it selects an option.
    assert all(r.ok for r in nematocyte.validate_configs(Root, directory))
