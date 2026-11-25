"""Applying overrides, and the flags of the entry point (`design/cli.md`)."""

from pathlib import Path

import pytest

import nematocyte
from nematocyte import ConfigError

from .helpers import (
    BatchNorm,
    Cnn,
    Config,
    Fcn,
    Opt,
    compose_settings,
    slurm_launcher,
    write_files,
)

# -- dict-typed fields ---------------------------------------------------------


def test_replace_in_a_dict_on_the_command_line_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(
            Config, overrides=["opt.optimizer_kwargs={_replace_:true,eps:1e-6}"]
        )
    message = str(excinfo.value)
    assert "'opt.optimizer_kwargs={_replace_:true,eps:1e-6}'" in message
    assert "'_replace_' is in the dict for 'opt.optimizer_kwargs'" in message
    assert "leave out the key" in message


def test_unknown_key_under_a_dataclass_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, overrides=["model.hiden_dim=10"])
    message = str(excinfo.value)
    assert "model.hiden_dim=10" in message
    assert "'hiden_dim'" in message
    assert "Fcn" in message
    assert "the option 'fcn' of 'model'" in message
    # The valid names and the closest match, without docs.
    assert "num_hidden, hidden_dim, activation, dropout_prob" in message
    assert "Did you mean 'hidden_dim'?" in message
    assert "Number of hidden layers" not in message


def test_key_of_another_option_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, overrides=["model=cnn", "model.hidden_dim=10"])
    message = str(excinfo.value)
    assert "Cnn" in message
    assert "the option 'cnn' of 'model'" in message
    assert "kernel_size, stride, activation, norm" in message


# -- group or value, from the type of the field ---------------------------------


def test_selection_of_a_named_config(config_dir: Path) -> None:
    config = nematocyte.compose(
        Config, config_dir, overrides=["model=single_linear_layer"]
    )
    assert config.model == Fcn(num_hidden=0, activation="relu")


def test_preset_needs_no_entry_in_defaults(config_dir: Path) -> None:
    # In Hydra this needs `- opt: null` in the primary config, or `+opt=...`.
    config = nematocyte.compose(
        Config, config_dir, overrides=["opt=weight_decay_adamw"]
    )
    assert config.opt == Opt(lr=0.001, weight_decay=0.01)


def test_selection_without_such_a_config(config_dir: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["wandb=foo"])
    message = str(excinfo.value)
    assert "'wandb' has no config called 'foo'" in message

    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, config_dir, overrides=["model=cnnn"])
    message = str(excinfo.value)
    assert "'model' has no option or config called 'cnnn'" in message
    assert "fcn, cnn" in message
    assert "single_linear_layer" in message
    assert "Did you mean 'cnn'?" in message


def test_selection_with_a_number_is_a_clear_error() -> None:
    # An unhandled traceback in Hydra.
    with pytest.raises(ConfigError) as excinfo:
        nematocyte.compose(Config, overrides=["model=123"])
    message = str(excinfo.value)
    assert "model=123" in message
    assert "fcn, cnn" in message


def test_order_of_arguments_has_no_meaning() -> None:
    one = nematocyte.compose(Config, overrides=["model.kernel_size=3", "model=cnn"])
    two = nematocyte.compose(Config, overrides=["model=cnn", "model.kernel_size=3"])
    assert one == two
    assert one.model == Cnn(kernel_size=3)


def test_selections_are_applied_shortest_path_first(config_dir: Path) -> None:
    config = nematocyte.compose(
        Config,
        config_dir,
        overrides=[
            "model.norm.momentum=0.9",
            "model.norm=my_batchnorm",
            "model=my_cnn",
        ],
    )
    assert config.model == Cnn(stride=1, norm=BatchNorm(momentum=0.9))


def test_key_given_twice_the_last_one_wins() -> None:
    assert nematocyte.compose(Config, overrides=["seed=1", "seed=2"]).seed == 2
    config = nematocyte.compose(
        Config, overrides=["model=fcn", "model.kernel_size=3", "model=cnn"]
    )
    assert config.model == Cnn(kernel_size=3)
    # An earlier value that would be an error does not matter.
    assert nematocyte.compose(Config, overrides=["seed=abc", "seed=3"]).seed == 3


def test_sweep_without_multirun_is_an_error() -> None:
    with pytest.raises(ConfigError, match="--multirun"):
        nematocyte.compose(Config, overrides=["seed=42,43"])
    with pytest.raises(ConfigError, match="--multirun"):
        nematocyte.compose(Config, overrides=["model=fcn,cnn"])


# -- nematocyte's own settings ---------------------------------------------------


def test_launcher_value(config_dir: Path) -> None:
    settings = compose_settings(
        config_dir,
        ["nematocyte.launcher=slurm_test", "nematocyte.launcher.partition=general"],
    )
    assert slurm_launcher(settings).partition == "general"


def test_unknown_launcher_is_an_error(config_dir: Path) -> None:
    # In Hydra, `hydra.launcher=nope` sets the node to a string, silently.
    with pytest.raises(ConfigError) as excinfo:
        compose_settings(config_dir, ["nematocyte.launcher=nope"])
    message = str(excinfo.value)
    assert "basic, submitit_slurm" in message
    assert "slurm_test" in message


def test_settings_in_a_primary_config(config_dir: Path) -> None:
    write_files(
        config_dir,
        {
            "cluster.yaml": """\
_defaults_:
  nematocyte:
    launcher: slurm_test

nematocyte:
  output_dir: runs/${seed}
  launcher:
    qos: high
""",
        },
    )
    settings = compose_settings(config_dir, config_name="cluster")
    assert slurm_launcher(settings).qos == "high"
    assert slurm_launcher(settings).partition == "test"
    assert settings.output_dir == Path("runs/42")


# -- the flags ---------------------------------------------------------------


def make_main(config_dir: Path, seen: list[Config]) -> nematocyte.EntryPoint:
    @nematocyte.main(Config, config_path=config_dir)
    def main(config: Config) -> None:
        seen.append(config)

    return main


def test_run(config_dir: Path) -> None:
    seen: list[Config] = []
    make_main(config_dir, seen)(["seed=3", "model=cnn"])
    assert seen == [Config(seed=3, model=Cnn())]


@pytest.mark.parametrize("flag", ["--config-name", "-cn"])
def test_config_name(config_dir: Path, flag: str) -> None:
    seen: list[Config] = []
    make_main(config_dir, seen)([flag, "good_run", "model.dropout_prob=0.5"])
    assert seen[0].seed == 1
    assert seen[0].model == Fcn(num_hidden=0, activation="selu", dropout_prob=0.5)


def test_arguments_come_from_sys_argv(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Config] = []
    monkeypatch.setattr("sys.argv", ["main.py", "seed=9"])
    make_main(config_dir, seen)()
    assert seen[0].seed == 9


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help(config_dir: Path, capsys: pytest.CaptureFixture[str], flag: str) -> None:
    seen: list[Config] = []
    make_main(config_dir, seen)(["model=cnn", flag])
    out = capsys.readouterr().out
    assert not seen
    assert "Train a model on a dataset." in out
    assert "model=cnn" in out
    assert "model.kernel_size=5" in out
    assert "model.num_hidden" not in out


def test_cfg_job(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[Config] = []
    make_main(config_dir, seen)(
        ["--cfg", "job", "model=cnn", "wandb.group=${now:%Y}-${seed}"]
    )
    out = capsys.readouterr().out
    assert not seen
    assert "kernel_size: 5" in out
    # The whole config: also the defaults of the dataclasses.
    assert "seed: 42" in out
    # Interpolations are printed as written.
    assert "group: ${now:%Y}-${seed}" in out
    assert "nematocyte:" not in out


def test_cfg_all(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_main(config_dir, [])(["-c", "all"])
    out = capsys.readouterr().out
    assert "seed: 42" in out
    assert "nematocyte:" in out
    assert "output_dir: null" in out


def test_cfg_with_an_added_config(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_main(config_dir, [])(["--cfg", "all", "+presets/debug"])
    out = capsys.readouterr().out
    assert "seed: 0" in out
    assert "log: debug" in out


def test_help_with_an_added_config(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_main(config_dir, [])(["+presets/debug_cnn", "--help"])
    out = capsys.readouterr().out
    assert "seed=5" in out
    assert "model.stride=1" in out
    assert "+NAME" in out


def test_cfg_needs_its_argument(config_dir: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        make_main(config_dir, [])(["--cfg"])
    assert excinfo.value.code == 2


def test_cfg_reports_errors_after_printing(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unresolved interpolation cannot be validated, so the tree is
    # resolved and built after it is printed.
    with pytest.raises(SystemExit) as excinfo:
        make_main(config_dir, [])(["--cfg", "job", "seed=${wandb.name}"])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "seed: ${wandb.name}" in captured.out
    assert "seed" in captured.err
    assert "integer" in captured.err


def test_config_error_is_printed_without_a_traceback(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []
    with pytest.raises(SystemExit) as excinfo:
        make_main(config_dir, seen)(["model=cnn", "model.hidden_dim=10"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert not seen
    assert err.startswith("Error in 'model.hidden_dim=10':")
    assert "Traceback" not in err


def test_plus_sets_a_key_in_a_dict(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []
    args = ["+opt.optimizer_kwargs.eps=1e-8", "opt.optimizer_kwargs={amsgrad:true}"]
    make_main(config_dir, seen)(args)
    assert seen[0].opt.optimizer_kwargs == {"amsgrad": True, "eps": 1e-8}
    make_main(config_dir, seen)(["--cfg", "job", *args])
    out = capsys.readouterr().out
    assert "  optimizer_kwargs:\n    amsgrad: true\n    eps: 1e-08\n" in out


def test_plus_on_a_declared_key_is_an_error(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []
    with pytest.raises(SystemExit) as excinfo:
        make_main(config_dir, seen)(["+opt.lr=0.1"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert not seen
    assert err.startswith("Error in '+opt.lr=0.1':")
    assert "Opt" in err
    assert "'+' is not needed: opt.lr=0.1" in err


def test_unknown_flag(config_dir: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        make_main(config_dir, [])(["--resolve"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "override",
    [
        "nematocyte.launcher=slurm_test",
        "nematocyte.launcher=basic",
        "nematocyte.sweeper=optuna",
        "nematocyte.sweeper.params={seed:'1,2'}",
    ],
)
def test_launcher_or_sweeper_without_multirun_is_an_error(
    config_dir: Path, capsys: pytest.CaptureFixture[str], override: str
) -> None:
    # Hydra ignores both silently, and the job runs on the login node.
    seen: list[Config] = []
    with pytest.raises(SystemExit) as excinfo:
        make_main(config_dir, seen)([override])
    assert excinfo.value.code == 1
    assert not seen
    assert "--multirun" in capsys.readouterr().err


def test_exception_of_the_function_is_not_caught(config_dir: Path) -> None:
    @nematocyte.main(Config, config_path=config_dir)
    def main(config: Config) -> None:
        raise RuntimeError(f"seed {config.seed}")

    with pytest.raises(RuntimeError, match="seed 42"):
        main([])


def test_drop_warning_on_the_command_line(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []
    with pytest.warns(nematocyte.DroppedValueWarning) as record:
        make_main(config_dir, seen)(["+good_run", "model=fcn"])
    assert seen[0].model == Fcn()
    message = str(record[0].message)
    assert message == (
        "The override 'model=fcn' resets the config values in 'model' to the "
        "values given by 'fcn'. This drops a value that was set for 'model' "
        "before:\n    'model.activation=selu', set in `configs/good_run.yaml`\n"
        "To keep the value, set it again after the selection; in the file that "
        "makes the selection, or on the command line."
    )
    assert record[0].filename == __file__
    # Also for --cfg, which composes the same tree.
    with pytest.warns(nematocyte.DroppedValueWarning):
        make_main(config_dir, seen)(["+good_run", "model=fcn", "--cfg", "job"])
    assert "activation: gelu" in capsys.readouterr().out
