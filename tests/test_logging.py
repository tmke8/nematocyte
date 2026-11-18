"""`nematocyte.log`, `log_level` and `configure_logging` (`design/logging.md`)."""

from collections.abc import Iterator
import logging
from pathlib import Path

import pytest

import nematocyte
from nematocyte import ConfigError

from .helpers import Config, empty_config_dir, write_files


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """`configure_logging` changes the root logger; undo that after each test."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def run(args: list[str], config_dir: Path | None = None) -> list[int]:
    """Run the entry point; return the level that each job saw."""
    levels: list[int] = []

    @nematocyte.main(Config, config_path=config_dir or empty_config_dir())
    def main(config: Config) -> None:
        del config
        levels.append(nematocyte.log_level())

    main(args)
    return levels


@pytest.mark.parametrize(
    ("args", "level"),
    [
        ([], logging.INFO),
        (["nematocyte.log=debug"], logging.DEBUG),
        (["nematocyte.log=info"], logging.INFO),
        (["nematocyte.log=warn"], logging.WARNING),
    ],
)
def test_log_level_follows_the_setting(args: list[str], level: int) -> None:
    assert run(args) == [level]


def test_log_level_from_a_config_file(tmp_path: Path) -> None:
    directory = write_files(
        tmp_path / "conf", {"quiet.yaml": "nematocyte:\n  log: warn\n"}
    )
    assert run(["--config-name", "quiet"], directory) == [logging.WARNING]


def test_log_level_in_every_job_of_a_multirun(
    capsys: pytest.CaptureFixture[str],
) -> None:
    levels = run(["-m", "nematocyte.log=debug", "seed=1,2"])
    capsys.readouterr()
    assert levels == [logging.DEBUG, logging.DEBUG]


def test_log_level_from_an_added_config_in_a_multirun(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # nematocyte's own settings of a multirun include the added configs.
    levels = run(["-m", "+presets/debug", "seed=1,2"], config_dir)
    capsys.readouterr()
    assert levels == [logging.DEBUG, logging.DEBUG]


def test_log_level_other_than_the_three_is_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run(["nematocyte.log=error"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "nematocyte.log" in err
    assert "'debug', 'info' or 'warn'" in err


@pytest.mark.parametrize("function", ["log_level", "configure_logging"])
def test_outside_a_run_is_an_error(function: str) -> None:
    with pytest.raises(ConfigError, match=f"nematocyte.{function}"):
        getattr(nematocyte, function)()


def test_help_shows_the_setting(capsys: pytest.CaptureFixture[str]) -> None:
    run(["--help"])
    out = capsys.readouterr().out
    assert "nematocyte.log=info" in out
    assert "Level of log messages" in out


def test_configure_logging_writes_to_stdout_and_to_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directories: list[Path] = []

    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:
        del config
        nematocyte.configure_logging()
        log = logging.getLogger("train")
        log.debug("hidden")
        log.info("epoch %d", 1)
        directories.append(nematocyte.output_dir())

    main([])
    captured = capsys.readouterr()
    assert "[train][INFO] - epoch 1" in captured.out
    assert "hidden" not in captured.out
    assert "epoch 1" not in captured.err
    (directory,) = directories
    assert directory.is_relative_to(tmp_path / "outputs")
    text = (directory / "run.log").read_text()
    assert "[train][INFO] - epoch 1" in text
    assert "hidden" not in text


def test_configure_logging_with_the_level_of_the_setting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:
        del config
        nematocyte.configure_logging(file=False)
        logging.getLogger("train").debug("details")
        logging.getLogger("train").info("progress")

    main(["nematocyte.log=debug"])
    assert "details" in capsys.readouterr().out
    main(["nematocyte.log=warn"])
    assert "progress" not in capsys.readouterr().out


def test_configure_logging_without_a_file_leaves_no_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:
        del config
        nematocyte.configure_logging(file=False)
        logging.getLogger("train").warning("careful")

    main([])
    assert "careful" in capsys.readouterr().out
    assert not (tmp_path / "outputs").exists()


def test_configure_logging_gives_each_job_its_own_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    directories: list[Path] = []

    @nematocyte.main(Config, config_path=empty_config_dir())
    def main(config: Config) -> None:
        nematocyte.configure_logging()
        logging.getLogger("train").info("seed %d", config.seed)
        directories.append(nematocyte.output_dir())

    main(["-m", "seed=1,2"])
    capsys.readouterr()
    first, second = (directory / "run.log" for directory in directories)
    assert "seed 1" in first.read_text()
    assert "seed 2" not in first.read_text()
    assert "seed 2" in second.read_text()
