"""Checking every config file in a test (`design/validation.md`)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import pytest

import nematocyte
from nematocyte import ConfigError, SchemaError, option
import nematocyte._validation as nematocyte_validation

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
    result = nematocyte.Validator(Config, config_dir).validate(Path(path))
    assert result.path == Path(path)
    if result.error is not None:
        raise result.error


def test_absolute_path(config_dir: Path) -> None:
    result = nematocyte.Validator(Config, config_dir).validate(
        config_dir / "good_run.yaml"
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
    result = nematocyte.Validator(Config, config_dir).validate(
        "model/norm/my_batchnorm.yaml"
    )
    assert result.ok
    write_files(
        config_dir,
        {"model/norm/bad.yaml": "defaults:\n  _self_: batchnorm\nmomentun: 1\n"},
    )
    found = errors(config_dir)
    assert list(found) == ["model/norm/bad.yaml"]
    assert "BatchNorm" in found["model/norm/bad.yaml"]


@dataclass
class Adam:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class Sgd:
    lr: float = 0.1
    momentum: float = 0.0


@dataclass
class Small:
    opt: Adam = field(default_factory=Adam)


@dataclass
class Large:
    opt: Sgd = field(default_factory=Sgd)
    depth: int = 3


@dataclass
class Shared:
    net: Annotated[Small, option("small")] | Annotated[Large, option("large")] = field(
        default_factory=Small
    )


def test_named_config_for_a_field_of_several_options_fits_each(
    tmp_path: Path,
) -> None:
    # `net.opt` is an `Adam` in `small` and an `Sgd` in `large`; a file in
    # `net/opt/` can be selected under either, so it is checked with both.
    write_files(
        tmp_path,
        {
            "net/opt/fast.yaml": "lr: 0.5\n",
            "net/opt/adam_only.yaml": "betas: [0.8, 0.9]\n",
            "net/opt/sgd_only.yaml": "momentum: 0.9\n",
        },
    )
    results = {
        result.path.as_posix(): result.error
        for result in nematocyte.validate_configs(Shared, tmp_path)
    }
    assert results["net/opt/fast.yaml"] is None
    adam_only = str(results["net/opt/adam_only.yaml"])
    assert "net=large" in adam_only
    assert "'betas'" in adam_only
    assert "Sgd" in adam_only
    sgd_only = str(results["net/opt/sgd_only.yaml"])
    # The option that is selected by default is the one without an override.
    assert "the option that is selected" in sgd_only
    assert "'momentum'" in sgd_only
    assert "Adam" in sgd_only
    assert "net=large:" not in sgd_only


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
    validator = nematocyte.Validator(Config, config_dir, construct=False)
    assert validator.validate("bad_gpu.yaml").ok
    assert not validator.validate("bad_key.yaml").ok


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
    validator = nematocyte.Validator(NeedsName, directory, config_name="base")
    assert validator.validate("sub/big.yaml").ok


def test_error_is_a_config_error(config_dir: Path) -> None:
    write_files(config_dir, {"bad.yaml": "nope: 1\n"})
    result = nematocyte.Validator(Config, config_dir).validate("bad.yaml")
    assert isinstance(result.error, ConfigError)
    assert not result.ok


def test_missing_file(config_dir: Path) -> None:
    result = nematocyte.Validator(Config, config_dir).validate("nope.yaml")
    assert "nope.yaml" in str(result.error)


# -- one validator for many files ---------------------------------------------


def test_validator_agrees_with_validate_configs(config_dir: Path) -> None:
    write_files(config_dir, {"experiments/broken.yaml": "model:\n  num_hiden: 1\n"})
    validator = nematocyte.Validator(Config, config_dir)
    expected = nematocyte.validate_configs(Config, config_dir)
    found = [validator.validate(result.path) for result in expected]
    assert [(r.path, str(r.error)) for r in found] == [
        (r.path, str(r.error)) for r in expected
    ]
    assert not all(result.ok for result in expected)


def test_validator_composes_with_one_composer(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[object] = []
    real = nematocyte_validation.Composer

    def counting(*args: Any, **kwargs: Any) -> object:
        composer = real(*args, **kwargs)
        made.append(composer)
        return composer

    monkeypatch.setattr(nematocyte_validation, "Composer", counting)
    validator = nematocyte.Validator(Config, config_dir)
    assert all(validator.validate(path).ok for path in sorted(CONFIG_FILES))
    assert len(made) == 1
    made.clear()
    nematocyte.validate_configs(Config, config_dir)
    assert len(made) == 1


def test_validator_for_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="nowhere"):
        nematocyte.Validator(Config, tmp_path / "nowhere")


def test_validator_for_a_top_level_file_named_after_a_field(
    config_dir: Path,
) -> None:
    # A problem with the directory, not with one file: it is raised once.
    write_files(config_dir, {"model.yaml": "num_hidden: 1\n"})
    with pytest.raises(ConfigError, match=r"model\.yaml") as info:
        nematocyte.Validator(Config, config_dir)
    assert "'model'" in str(info.value)


def test_validator_for_an_unsupported_class(config_dir: Path) -> None:
    @dataclass
    class Reserved:
        nematocyte: int = 1

    with pytest.raises(SchemaError, match=r"Reserved\.nematocyte"):
        nematocyte.Validator(Reserved, config_dir)


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


def test_drop_warning_for_a_primary_config_only(config_dir: Path) -> None:
    write_files(
        config_dir, {"child.yaml": "defaults:\n  _self_: good_run\n  model: fcn\n"}
    )
    with pytest.warns(nematocyte.DroppedValueWarning, match="model.activation"):
        assert nematocyte.Validator(Config, config_dir).validate("child.yaml").ok
    # To check a named config, the validator selects it on top of `good_run`;
    # nobody wrote that selection, so what it drops is not warned about. The
    # test fails on any warning of this class.
    validator = nematocyte.Validator(Config, config_dir, config_name="good_run")
    assert validator.validate("model/single_linear_layer.yaml").ok
    assert validator.validate("model/my_cnn.yaml").ok
