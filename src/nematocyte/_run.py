"""Running a job: the config object, the function, and the output directory."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from ._build import build
from ._errors import ConfigError
from ._tree import node_from_plain

__all__ = [
    "ENTRY_ATTRIBUTE",
    "Entry",
    "Job",
    "JobResult",
    "StartDir",
    "output_dir",
    "run_job",
]

_INTERNAL_DIR = ".nematocyte"
ENTRY_ATTRIBUTE = "_nematocyte_entry"
"""The attribute of the decorated function that holds its `Entry`."""


@dataclass(frozen=True)
class Entry:
    """What the decorator of the entry point was given."""

    function: Callable[[Any], Any]
    config_cls: type
    config_dir: Path
    config_name: str | None


@dataclass(frozen=True)
class Job:
    """One job: a number, and the config that was composed for it.

    The config is the resolved tree as plain dicts, lists and scalars. The
    job builds the config object from it and does not read any config file.
    """

    number: int
    overrides: list[str]
    config: dict[str, Any]


@dataclass(frozen=True)
class JobResult:
    """The return value of a job, or the exception that it raised."""

    job: Job
    value: Any = None
    error: BaseException | None = None


class StartDir:
    """The directory of one start of the program, which holds a directory per job.

    It is created when it is first needed. If the path already exists then,
    a number is added to the name, so that a directory is never shared.
    """

    def __init__(self, requested: Path, *, created: bool = False) -> None:
        self.requested = requested.absolute()
        self._path: Path | None = self.requested if created else None

    def __getstate__(self) -> dict[str, Any]:
        # In another process, the directory is the one that this process made.
        return {"requested": self.requested, "_path": self.create()}

    def create(self) -> Path:
        if self._path is None:
            candidate = self.requested
            number = 1
            while True:
                try:
                    candidate.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    number += 1
                    candidate = self.requested.with_name(
                        f"{self.requested.name}-{number}"
                    )
                else:
                    break
            self._path = candidate
        return self._path


def default_start_dir(now: datetime) -> Path:
    return Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")


@dataclass
class _Context:
    start: StartDir
    job: Job

    def directory(self) -> Path:
        directory = self.start.create() / str(self.job.number)
        internal = directory / _INTERNAL_DIR
        if not internal.is_dir():
            internal.mkdir(parents=True, exist_ok=True)
            _write_json(internal / "config.json", self.job.config)
            _write_json(internal / "overrides.json", self.job.overrides)
        return directory


def _json_keys(value: Any) -> Any:
    """JSON only has string keys; other keys are written as their text."""
    if isinstance(value, dict):
        return {
            key if isinstance(key, str) else json.dumps(key, default=str): _json_keys(v)
            for key, v in value.items()
        }
    if isinstance(value, list):
        return [_json_keys(item) for item in value]
    return value


def _write_json(path: Path, content: Any) -> None:
    text = json.dumps(_json_keys(content), indent=2, default=str)
    path.write_text(text + "\n", encoding="utf-8")


_current: _Context | None = None


def output_dir() -> Path:
    """Return the output directory of the current job, creating it if needed.

    When the directory is created, the config and the overrides of the job
    are saved in it, under `.nematocyte/`. Outside a run, such as after
    `nematocyte.compose(...)`, this is an error.
    """
    if _current is None:
        raise ConfigError(
            "nematocyte.output_dir() was called outside a run. There is an "
            "output directory only while the function decorated with "
            "@nematocyte.main is running."
        )
    return _current.directory()


def run_job(entry: Entry, job: Job, start: StartDir) -> Any:
    """Build the config object of a job and call the function with it."""
    global _current  # noqa: PLW0603
    config = build(node_from_plain(entry.config_cls, job.config))
    previous = _current
    _current = _Context(start, job)
    try:
        return entry.function(config)
    finally:
        _current = previous
