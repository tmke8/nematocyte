"""`--multirun`, sweepers and launchers (`design/multirun.md`).

None of these tests needs a SLURM cluster: the SLURM launcher is run against
a stand-in for submitit's executor.
"""

from collections.abc import Callable, Iterable
import inspect
from pathlib import Path
import pickle
from typing import Any

import cloudpickle
import pytest
from ruamel.yaml import YAML
import submitit
from submitit.slurm import slurm

import nematocyte
from nematocyte._multirun import slurm_parameters
from nematocyte._settings import SubmititSlurmLauncher, TpeSampler

from .helpers import (
    Cnn,
    Config,
    Fcn,
    compose_settings,
    empty_config_dir,
    fcn,
    optuna_sweeper,
    write_files,
)


def make_main(
    config_dir: Path,
    seen: list[Config],
    result: Callable[[Config], Any] = lambda config: config.seed,
) -> nematocyte.EntryPoint:
    @nematocyte.main(Config, config_dir=config_dir)
    def main(config: Config) -> Any:
        seen.append(config)
        return result(config)

    return main


def run(main: nematocyte.EntryPoint, args: list[str]) -> int:
    """Run the entry point; returns the exit code."""
    try:
        main(args)
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        return exc.code
    return 0


# -- the basic sweeper and launcher ------------------------------------------


@pytest.mark.parametrize("flag", ["--multirun", "-m"])
def test_comma_separated_values(flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[Config] = []
    assert run(make_main(empty_config_dir(), seen), [flag, "seed=42,43"]) == 0
    assert [config.seed for config in seen] == [42, 43]
    err = capsys.readouterr().err
    assert "Launching 2 jobs locally" in err
    assert "#0 : seed=42" in err
    assert "#1 : seed=43" in err


def test_several_sweeps_give_the_product(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[Config] = []
    args = ["-m", "seed=42,43", "model.hidden_dim=10", "model.num_hidden=1,2,3"]
    assert run(make_main(empty_config_dir(), seen), args) == 0
    capsys.readouterr()
    assert [(c.seed, fcn(c).num_hidden) for c in seen] == [
        (42, 1),
        (42, 2),
        (42, 3),
        (43, 1),
        (43, 2),
        (43, 3),
    ]
    assert all(fcn(c).hidden_dim == 10 for c in seen)


def test_choice_and_range(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[Config] = []
    args = ["-m", "seed=range(1,3)", "opt.lr=choice(0.1,0.2)"]
    assert run(make_main(empty_config_dir(), seen), args) == 0
    capsys.readouterr()
    assert [(c.seed, c.opt.lr) for c in seen] == [
        (1, 0.1),
        (1, 0.2),
        (2, 0.1),
        (2, 0.2),
    ]


def test_selections_can_be_swept(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[Config] = []
    assert run(make_main(empty_config_dir(), seen), ["-m", "model=fcn,cnn"]) == 0
    capsys.readouterr()
    assert [type(config.model) for config in seen] == [Fcn, Cnn]


@pytest.mark.parametrize(
    ("args", "seeds"),
    [
        (["seed=1,2", "seed=3"], [3]),
        (["seed=3", "seed=1,2"], [1, 2]),
        (["seed=1,2", "seed=4,5,6"], [4, 5, 6]),
    ],
)
def test_key_given_twice_in_a_multirun_the_last_one_wins(
    capsys: pytest.CaptureFixture[str], args: list[str], seeds: list[int]
) -> None:
    seen: list[Config] = []
    assert run(make_main(empty_config_dir(), seen), ["-m", *args]) == 0
    capsys.readouterr()
    assert [config.seed for config in seen] == seeds


def test_multirun_without_a_sweep_is_one_job(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[Config] = []
    assert run(make_main(empty_config_dir(), seen), ["-m", "seed=5"]) == 0
    assert "Launching 1 job locally" in capsys.readouterr().err
    assert [config.seed for config in seen] == [5]


def test_interval_names_the_optuna_sweeper(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[Config] = []
    assert (
        run(make_main(empty_config_dir(), seen), ["-m", "opt.lr=interval(0.1,0.2)"])
        == 1
    )
    err = capsys.readouterr().err
    assert not seen
    assert "optuna" in err
    assert "opt.lr" in err


def test_every_job_is_checked_before_the_first_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # `model.num_hidden` does not exist for `cnn`, which is the second job.
    seen: list[Config] = []
    args = ["-m", "model=fcn,cnn", "model.num_hidden=2"]
    assert run(make_main(empty_config_dir(), seen), args) == 1
    err = capsys.readouterr().err
    assert not seen
    assert "job #1" in err
    assert "'num_hidden'" in err
    assert "Cnn" in err


def test_post_init_error_is_found_before_anything_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[Config] = []
    assert run(make_main(empty_config_dir(), seen), ["-m", "gpu=0,-7"]) == 1
    assert not seen
    assert "Invalid GPU index -7" in capsys.readouterr().err


def test_failed_job_does_not_stop_the_others(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[Config] = []

    def result(config: Config) -> float:
        if config.seed in (2, 4):
            raise RuntimeError(f"seed {config.seed} is unlucky")
        return 0.0

    assert run(make_main(empty_config_dir(), seen, result), ["-m", "seed=1,2,3,4"]) == 1
    assert [config.seed for config in seen] == [1, 2, 3, 4]
    err = capsys.readouterr().err
    # Every failure is listed, not only the first.
    assert "Job #1 (seed=2) failed" in err
    assert "Job #3 (seed=4) failed" in err
    assert "seed 2 is unlucky" in err
    assert "seed 4 is unlucky" in err
    assert "2 of 4 jobs failed" in err


def test_now_is_the_same_in_every_job(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[Config] = []
    # Hydra gives every job a different value for a pattern like this one.
    args = ["-m", "seed=1,2,3", "wandb.group=${now:%M-%S.%f}"]
    assert run(make_main(empty_config_dir(), seen), args) == 0
    capsys.readouterr()
    assert len({config.wandb.group for config in seen}) == 1


def test_job_gets_a_fresh_config_object(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[Config] = []
    assert run(make_main(empty_config_dir(), seen), ["-m", "seed=1,2"]) == 0
    capsys.readouterr()
    assert seen[0].opt is not seen[1].opt


def test_params_of_the_basic_sweeper(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_files(
        config_dir,
        {
            "nematocyte/sweeper/seeds.yaml": """\
defaults:
  _self_: basic

params:
  seed: 42,43
  model.num_hidden: range(1, 3)
  gpu: 0
""",
        },
    )
    seen: list[Config] = []
    args = ["-m", "nematocyte.sweeper=seeds"]
    assert run(make_main(config_dir, seen), args) == 0
    assert [(c.seed, fcn(c).num_hidden, c.gpu) for c in seen] == [
        (42, 1, 0),
        (42, 2, 0),
        (43, 1, 0),
        (43, 2, 0),
    ]
    # A key on the command line replaces the same key in `params`.
    seen.clear()
    assert (
        run(make_main(config_dir, seen), [*args, "seed=7", "model.num_hidden=5,6"]) == 0
    )
    capsys.readouterr()
    assert [(c.seed, fcn(c).num_hidden) for c in seen] == [(7, 5), (7, 6)]


def test_params_are_checked_at_the_start(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []
    args = ["-m", "nematocyte.sweeper.params={sed:'1,2'}"]
    assert run(make_main(config_dir, seen), args) == 1
    assert not seen
    assert "'sed'" in capsys.readouterr().err


def test_own_settings_cannot_be_swept(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["-m", "nematocyte.launcher=basic,submitit_slurm"]
    assert run(make_main(empty_config_dir(), []), args) == 1
    assert "cannot be swept" in capsys.readouterr().err


def test_help_with_multirun_leaves_the_sweeps_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        run(
            make_main(empty_config_dir(), []), ["-m", "seed=1,2", "model=cnn", "--help"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "seed=42" in out
    assert "model=cnn" in out


# -- the SLURM launcher -------------------------------------------------------


class FakeJob:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def result(self) -> Any:
        # What the compute node does: unpickle the submission and call it.
        function, args = pickle.loads(self.payload)  # noqa: S301
        return function(*args)


class FakeExecutor:
    """Stands in for `submitit.SlurmExecutor`, which needs `srun`."""

    instances: list[FakeExecutor] = []

    def __init__(self, folder: str, max_num_timeout: int, python: str | None) -> None:
        self.folder = folder
        self.max_num_timeout = max_num_timeout
        self.python = python
        self.parameters: dict[str, Any] = {}
        self.batches: list[int] = []
        FakeExecutor.instances.append(self)

    def update_parameters(self, **kwargs: Any) -> None:
        # The real executor checks the names against `_make_sbatch_string`.
        slurm._make_sbatch_string(command="x", folder=self.folder, **kwargs)  # noqa: SLF001
        self.parameters.update(kwargs)

    def map_array(
        self, function: Callable[..., Any], *iterables: Iterable[Any]
    ) -> list[FakeJob]:
        jobs = [
            FakeJob(cloudpickle.dumps((function, args)))
            for args in zip(*iterables, strict=True)
        ]
        self.batches.append(len(jobs))
        return jobs


@pytest.fixture
def fake_slurm(monkeypatch: pytest.MonkeyPatch) -> type[FakeExecutor]:
    FakeExecutor.instances = []
    monkeypatch.setattr(submitit, "SlurmExecutor", FakeExecutor)
    return FakeExecutor


def slurm_main(config_dir: Path) -> nematocyte.EntryPoint:
    # The jobs run from a pickle, so the results come back as return values.
    @nematocyte.main(Config, config_dir=config_dir)
    def main(config: Config) -> Any:
        if config.seed == 13:
            raise RuntimeError("unlucky")
        return config.seed, str(nematocyte.output_dir())

    return main


def test_slurm_launcher_submits_one_array(
    config_dir: Path,
    tmp_path: Path,
    fake_slurm: type[FakeExecutor],
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ["-m", "seed=1,2,3", "nematocyte.launcher=slurm_test"]
    assert run(slurm_main(config_dir), args) == 0
    err = capsys.readouterr().err
    assert "Submitting 3 jobs to SLURM" in err

    (executor,) = fake_slurm.instances
    assert executor.batches == [3]
    # `submitit_folder` is `.submitit/%j` in the directory of the multirun.
    (start,) = (tmp_path / "outputs").glob("*/*")
    assert executor.folder == str(start / ".submitit" / "%j")
    # The settings of the launcher file, under the names that SLURM uses.
    assert executor.parameters["partition"] == "test"
    assert executor.parameters["cpus_per_task"] == 4
    assert executor.parameters["mem"] == "8GB"
    assert executor.parameters["gres"] == "gpu:1"
    assert executor.parameters["setup"] == ["source .venv/bin/activate"]
    assert executor.parameters["time"] == 60
    # The jobs ran from their pickles, each in its own numbered directory.
    assert sorted(path.name for path in start.iterdir()) == ["0", "1", "2"]
    assert (start / "2" / ".nematocyte" / "config.json").is_file()


def test_slurm_job_name_defaults_to_the_script(
    config_dir: Path,
    fake_slurm: type[FakeExecutor],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["/some/where/train_model.py"])
    args = ["-m", "seed=1", "nematocyte.launcher=slurm_test"]
    assert run(slurm_main(config_dir), args) == 0
    capsys.readouterr()
    assert fake_slurm.instances[0].parameters["job_name"] == "train_model"


def test_slurm_settings_from_the_command_line(
    config_dir: Path,
    tmp_path: Path,
    fake_slurm: type[FakeExecutor],
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [
        "-m",
        "seed=1",
        "nematocyte.launcher=submitit_slurm",
        "nematocyte.launcher.name=my-job",
        "nematocyte.launcher.submitit_folder=logs/%j",
        "nematocyte.launcher.max_num_timeout=2",
        "nematocyte.launcher.additional_parameters={mail-user:me}",
        "nematocyte.output_dir=sweeps/a",
    ]
    assert run(slurm_main(config_dir), args) == 0
    capsys.readouterr()
    (executor,) = fake_slurm.instances
    assert executor.folder == "logs/%j"
    assert executor.max_num_timeout == 2
    assert executor.parameters["job_name"] == "my-job"
    assert executor.parameters["additional_parameters"] == {"mail-user": "me"}
    assert (tmp_path / "sweeps" / "a" / "0").is_dir()


def test_slurm_failures_are_reported(
    config_dir: Path,
    fake_slurm: type[FakeExecutor],
    capsys: pytest.CaptureFixture[str],
) -> None:
    del fake_slurm
    args = ["-m", "seed=1,13,2", "nematocyte.launcher=slurm_test"]
    assert run(slurm_main(config_dir), args) == 1
    err = capsys.readouterr().err
    assert "Job #1 (seed=13) failed" in err
    assert "1 of 3 jobs failed" in err


def test_slurm_job_does_not_read_the_config_files(
    config_dir: Path,
    fake_slurm: type[FakeExecutor],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # What is sent to the job is the resolved tree. An edit that is made
    # while the job waits in the queue does not affect it.
    results: list[Any] = []
    original = FakeJob.result

    def result_after_edit(self: FakeJob) -> Any:
        (config_dir / "good_run.yaml").write_text("seed: not-a-number\n")
        value = original(self)
        results.append(value)
        return value

    monkeypatch.setattr(FakeJob, "result", result_after_edit)
    args = ["-m", "-cn", "good_run", "nematocyte.launcher=slurm_test"]
    assert run(slurm_main(config_dir), args) == 0
    capsys.readouterr()
    assert [value[0] for value in results] == [1]
    del fake_slurm


def test_slurm_parameters_match_submitit() -> None:
    # The launcher is a hand-written copy of submitit's arguments.
    launcher = SubmititSlurmLauncher(mem_gb=1.5)
    parameters = slurm_parameters(launcher, "main")
    accepted = set(inspect.signature(slurm._make_sbatch_string).parameters)  # noqa: SLF001
    assert set(parameters) <= accepted
    assert parameters["mem"] == "1536MB"
    script = slurm._make_sbatch_string(command="run", folder="f", **parameters)  # noqa: SLF001
    assert "#SBATCH --job-name=main" in script
    assert "#SBATCH --time=60" in script


def test_slurm_launcher_has_all_of_hydras_settings() -> None:
    settings = compose_settings(overrides=["nematocyte.launcher=submitit_slurm"])
    assert isinstance(settings.launcher, SubmititSlurmLauncher)
    assert len(SubmititSlurmLauncher.__dataclass_fields__) == 28


def test_launcher_file_of_the_example() -> None:
    directory = Path(__file__).parent.parent / "example" / "configs"
    settings = compose_settings(directory, ["nematocyte.launcher=slurm_dipc"])
    assert isinstance(settings.launcher, SubmititSlurmLauncher)
    parameters = slurm_parameters(settings.launcher, "main")
    assert parameters["partition"] == "general"
    assert parameters["qos"] == "regular"
    assert parameters["mem"] == "96GB"
    assert parameters["constraint"] == "a100-pcie"
    assert len(parameters["setup"]) == 3
    slurm._make_sbatch_string(command="run", folder="f", **parameters)  # noqa: SLF001


# -- the Optuna sweeper -------------------------------------------------------


def test_optuna_sweeper(
    config_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []

    def objective(config: Config) -> float:
        assert isinstance(config.model, Fcn)
        return -abs(config.opt.lr - 1e-4) - config.model.num_hidden

    args = ["-m", "nematocyte.sweeper=search", "seed=7"]
    assert run(make_main(config_dir, seen, objective), args) == 0
    err = capsys.readouterr().err
    # `n_trials: 6` in batches of `n_jobs: 3`.
    assert len(seen) == 6
    assert err.count("Launching 3 jobs locally") == 2
    for config in seen:
        assert isinstance(config.model, Fcn)
        assert 1e-5 <= config.opt.lr <= 1e-3
        assert config.model.num_hidden in (1, 2, 3)
        assert config.model.dropout_prob in (0.0, 0.1)
        # An override that is not a sweep is a fixed value for every trial.
        assert config.seed == 7
    assert "Best parameters" in err
    assert "Best value" in err

    (start,) = (tmp_path / "outputs").glob("*/*")
    results = YAML(typ="safe", pure=True).load(start / "optimization_results.yaml")
    assert results["name"] == "optuna"
    assert set(results["best_params"]) == {
        "opt.lr",
        "model.num_hidden",
        "model.dropout_prob",
    }
    # `direction: [maximize]`
    best = max(-abs(c.opt.lr - 1e-4) - fcn(c).num_hidden for c in seen)
    assert results["best_value"] == pytest.approx(best)


def test_optuna_search_space_from_the_command_line(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []
    args = [
        "-m",
        "nematocyte.sweeper=optuna",
        "nematocyte.sweeper.n_trials=4",
        "nematocyte.sweeper.sampler=random",
        "nematocyte.sweeper.sampler.seed=1",
        "seed=interval(1, 12)",
        "model=choice(fcn,cnn)",
        "opt.weight_decay=range(0, 1, 0.25)",
    ]
    assert run(make_main(config_dir, seen), args) == 0
    capsys.readouterr()
    assert len(seen) == 4
    for config in seen:
        # An interval between integers is an integer distribution.
        assert isinstance(config.seed, int)
        assert 1 <= config.seed <= 12
        assert config.opt.weight_decay in (0.0, 0.25, 0.5, 0.75)


def test_optuna_keys_are_checked_at_the_start(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []
    # `num_hidden` exists for one option of `model`, but not the selected one.
    args = ["-m", "nematocyte.sweeper=search", "model=cnn"]
    assert run(make_main(config_dir, seen), args) == 1
    assert not seen
    err = capsys.readouterr().err
    assert "model.num_hidden" in err
    assert "Cnn" in err


def test_optuna_failures_and_max_failure_rate(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []

    def objective(config: Config) -> float:
        del config
        if len(seen) % 4 == 1:
            raise RuntimeError("failed trial")
        return 1.0

    base = [
        "-m",
        "nematocyte.sweeper=optuna",
        "nematocyte.sweeper.n_trials=8",
        "nematocyte.sweeper.n_jobs=4",
        "opt.lr=interval(0.1, 0.9)",
    ]
    # One of the four trials of a batch fails. The default tolerates none,
    # and the search stops after the batch.
    assert run(make_main(config_dir, seen, objective), base) == 1
    assert len(seen) == 4
    err = capsys.readouterr().err
    assert "max_failure_rate" in err
    assert "failed trial" in err

    seen.clear()
    tolerant = [*base, "nematocyte.sweeper.max_failure_rate=0.25"]
    assert run(make_main(config_dir, seen, objective), tolerant) == 0
    assert len(seen) == 8
    assert "Best value: 1.0" in capsys.readouterr().err


def test_optuna_without_a_completed_trial(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def objective(config: Config) -> float:
        raise RuntimeError(f"failed with {config.opt.lr}")

    args = [
        "-m",
        "nematocyte.sweeper=optuna",
        "nematocyte.sweeper.n_trials=2",
        "nematocyte.sweeper.max_failure_rate=1.0",
        "opt.lr=interval(0.1, 0.9)",
    ]
    assert run(make_main(config_dir, [], objective), args) == 0
    assert "No trial was completed" in capsys.readouterr().err


def test_optuna_with_several_directions(
    config_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Config] = []
    args = [
        "-m",
        "nematocyte.sweeper=optuna",
        "nematocyte.sweeper.n_trials=4",
        "nematocyte.sweeper.direction=[minimize,maximize]",
        "opt.lr=interval(0.1, 0.9)",
    ]
    main = make_main(
        config_dir, seen, lambda config: (config.opt.lr, 1 - config.opt.lr)
    )
    assert run(main, args) == 0
    capsys.readouterr()
    (start,) = (tmp_path / "outputs").glob("*/*")
    results = YAML(typ="safe", pure=True).load(start / "optimization_results.yaml")
    assert results["solutions"]
    assert set(results["solutions"][0]) == {"params", "values"}


def test_optuna_objective_has_to_be_a_number(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        "-m",
        "nematocyte.sweeper=optuna",
        "nematocyte.sweeper.n_trials=1",
        "opt.lr=interval(0.1, 0.9)",
    ]
    assert run(make_main(config_dir, [], lambda config: None), args) == 1  # noqa: ARG005
    assert "objective" in capsys.readouterr().err


def test_sampler_is_a_group_of_the_optuna_sweeper(config_dir: Path) -> None:
    settings = compose_settings(config_dir, ["nematocyte.sweeper=search"])
    assert optuna_sweeper(settings).sampler == TpeSampler(seed=0)
    assert optuna_sweeper(settings).direction == ["maximize"]
    assert optuna_sweeper(settings).n_trials == 6


def test_sweeper_file_of_the_example() -> None:
    directory = Path(__file__).parent.parent / "example" / "configs"
    settings = compose_settings(directory, ["nematocyte.sweeper=fcn_params"])
    sweeper = optuna_sweeper(settings)
    assert sweeper.sampler == TpeSampler(warn_independent_sampling=True)
    assert sweeper.study_name == "fcn_params"
    assert (sweeper.n_trials, sweeper.n_jobs) == (25, 3)
