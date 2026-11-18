"""Output directories (`design/outputs.md`)."""

from datetime import datetime
import json
from pathlib import Path

import pytest

import nematocyte
from nematocyte import ConfigError

from .helpers import Config, empty_config_dir


class FixedTime(datetime):
    @classmethod
    def now(cls, tz: object = None) -> FixedTime:  # noqa: ARG003
        return cls(2026, 11, 16, 16, 49, 10)


@pytest.fixture(autouse=True)
def fixed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nematocyte._cli.datetime", FixedTime)


def make_main(directories: list[Path], *, ask: bool = True) -> nematocyte.EntryPoint:
    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:
        if ask and config.flag:
            directories.append(nematocyte.output_dir())

    return main


def test_single_run_is_a_numbered_directory(tmp_path: Path) -> None:
    directories: list[Path] = []
    make_main(directories)(["flag=true"])
    expected = tmp_path / "outputs" / "2026-11-16" / "16-49-10" / "0"
    assert directories == [expected]
    assert expected.is_dir()


def test_directory_is_created_when_it_is_first_asked_for(tmp_path: Path) -> None:
    # A run that never asks for its directory leaves nothing behind.
    make_main([])(["flag=false"])
    assert not (tmp_path / "outputs").exists()


def test_asking_twice_gives_the_same_directory() -> None:
    directories: list[Path] = []

    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:  # noqa: ARG001
        directories.append(nematocyte.output_dir())
        (nematocyte.output_dir() / "checkpoints").mkdir()
        directories.append(nematocyte.output_dir())

    main([])
    assert directories[0] == directories[1]
    assert (directories[0] / "checkpoints").is_dir()


def test_what_nematocyte_writes(tmp_path: Path) -> None:
    make_main([])(["flag=true", "model=cnn", "wandb.group=${now:%Y}-${seed}"])
    internal = tmp_path / "outputs/2026-11-16/16-49-10/0/.nematocyte"
    assert sorted(path.name for path in internal.iterdir()) == [
        "config.json",
        "overrides.json",
    ]
    config = json.loads((internal / "config.json").read_text())
    # The resolved config, as `to_dict` gives it, with `_option_` for the groups.
    assert config == nematocyte.to_dict(
        nematocyte.compose(
            Config, overrides=["flag=true", "model=cnn", "wandb.group=2026-42"]
        )
    )
    assert config["model"]["_option_"] == "cnn"
    assert config["wandb"]["group"] == "2026-42"
    assert json.loads((internal / "overrides.json").read_text()) == [
        "flag=true",
        "model=cnn",
        "wandb.group=${now:%Y}-${seed}",
    ]
    # There is no log file.
    assert [path.name for path in internal.parent.iterdir()] == [".nematocyte"]


def test_saved_overrides_include_the_added_configs(
    tmp_path: Path, config_dir: Path
) -> None:
    @nematocyte.main(Config, config_path=config_dir)
    def main(config: Config) -> None:  # noqa: ARG001
        nematocyte.output_dir()

    main(["flag=true", "+presets/debug"])
    internal = tmp_path / "outputs/2026-11-16/16-49-10/0/.nematocyte"
    assert json.loads((internal / "overrides.json").read_text()) == [
        "+presets/debug",
        "flag=true",
    ]
    assert json.loads((internal / "config.json").read_text())["seed"] == 0


def test_output_dir_outside_a_run_is_an_error() -> None:
    nematocyte.compose(Config)
    with pytest.raises(ConfigError, match="outside a run"):
        nematocyte.output_dir()


def test_output_dir_is_gone_after_the_run() -> None:
    make_main([])(["flag=true"])
    with pytest.raises(ConfigError):
        nematocyte.output_dir()


def test_working_directory_is_not_changed(tmp_path: Path) -> None:
    make_main([])(["flag=true"])
    assert Path.cwd() == tmp_path


def test_setting_with_interpolations(tmp_path: Path) -> None:
    directories: list[Path] = []
    make_main(directories)(
        [
            "flag=true",
            "nematocyte.output_dir=runs/${wandb.group}/${now:%H-%M-%S}",
            "wandb.group=exp1",
        ]
    )
    assert directories == [tmp_path / "runs" / "exp1" / "16-49-10" / "0"]


def test_directory_is_never_shared(tmp_path: Path) -> None:
    # Two starts in the same second.
    directories: list[Path] = []
    for _ in range(3):
        make_main(directories)(["flag=true"])
    day = tmp_path / "outputs" / "2026-11-16"
    assert directories == [
        day / "16-49-10" / "0",
        day / "16-49-10-2" / "0",
        day / "16-49-10-3" / "0",
    ]


def test_help_and_cfg_create_no_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_main([])(["--help"])
    make_main([])(["--cfg", "job"])
    capsys.readouterr()
    assert not (tmp_path / "outputs").exists()


def test_multirun_has_one_directory_per_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directories: list[Path] = []

    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:
        # Only the jobs that ask get a directory.
        if config.seed != 2:
            directories.append(nematocyte.output_dir())

    main(["--multirun", "seed=1,2,3"])
    capsys.readouterr()
    start = tmp_path / "outputs" / "2026-11-16" / "16-49-10"
    assert directories == [start / "0", start / "2"]
    assert sorted(path.name for path in start.iterdir()) == ["0", "2"]
    overrides = json.loads((start / "2/.nematocyte/overrides.json").read_text())
    # The overrides of the job include its values of the sweep.
    assert overrides == ["seed=3"]
    config = json.loads((start / "2/.nematocyte/config.json").read_text())
    assert config["seed"] == 3


def test_multirun_directory_is_created_at_the_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_main([], ask=False)(["--multirun", "seed=1,2"])
    capsys.readouterr()
    start = tmp_path / "outputs" / "2026-11-16" / "16-49-10"
    assert start.is_dir()
    assert list(start.iterdir()) == []


def test_single_run_and_multirun_share_the_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directories: list[Path] = []
    make_main(directories)(["flag=true"])
    make_main(directories)(["--multirun", "flag=true", "seed=1,2"])
    capsys.readouterr()
    day = tmp_path / "outputs" / "2026-11-16"
    assert directories == [
        day / "16-49-10" / "0",
        day / "16-49-10-2" / "0",
        day / "16-49-10-2" / "1",
    ]
    # Every run is at the same depth.
    assert len(list(tmp_path.glob("outputs/*/*/*/.nematocyte"))) == 3


def test_job_num_of_a_single_run() -> None:
    numbers: list[int] = []

    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:  # noqa: ARG001
        numbers.append(nematocyte.job_num())

    main([])
    assert numbers == [0]


def test_job_num_is_the_name_of_the_directory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[tuple[int, int, str]] = []

    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:
        seen.append((config.seed, nematocyte.job_num(), nematocyte.output_dir().name))

    main(["--multirun", "seed=1,2,3"])
    capsys.readouterr()
    assert seen == [(1, 0, "0"), (2, 1, "1"), (3, 2, "2")]


def test_job_num_outside_a_run_is_an_error() -> None:
    with pytest.raises(ConfigError, match=r"nematocyte\.job_num"):
        nematocyte.job_num()
