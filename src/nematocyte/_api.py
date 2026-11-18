"""The public functions: the entry point, `compose` and `from_run_config`."""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import functools
import os
from pathlib import Path
import sys
from typing import Any, Protocol, cast
from typing_extensions import TypeForm

from ._build import build
from ._cli import run_cli
from ._compose import Composer
from ._interpolation import resolve
from ._overrides import parse_override
from ._run import ENTRY_ATTRIBUTE, Entry
from ._schema import RESERVED_KEY, config_class, validate_schema
from ._settings import Settings
from ._tree import node_from_plain
from ._types import JSONValue

__all__ = ["EntryPoint", "compose", "from_run_config", "main"]


class EntryPoint(Protocol):
    """The decorated function: it reads the command line and runs the program."""

    def __call__(self, args: Sequence[str] | None = None) -> None:
        """Run the program.

        `args` replaces the arguments of the command line, which is useful
        in tests.
        """
        ...


def _defining_dir(function: Callable[..., Any]) -> Path:
    """The directory of the file that defines `function`."""
    module = sys.modules.get(function.__module__)
    filename = getattr(module, "__file__", None)
    if filename is None:
        code = getattr(function, "__code__", None)
        filename = code.co_filename if code is not None else None
    if filename is None or not Path(filename).parent.is_dir():
        return Path.cwd()
    return Path(filename).resolve().parent


def main[C](
    config_cls: TypeForm[C],
    *,
    config_path: str | os.PathLike[str],
    config_name: str | None = None,
) -> Callable[[Callable[[C], object]], EntryPoint]:
    """Turn a function that takes the config into the entry point of a program.

    Args:
        config_cls: The root config class, a dataclass or a TypedDict.
            Every config is checked against it, and the function receives an
            instance of it.
        config_path: The directory with the config files, relative to the
            file that defines the function. It has to exist, even if it is empty.
        config_name: The primary config to use if `--config-name` is not
            given. Without it, the config comes from the root config class alone.

    The return value of the function is the objective for a sweeper, and is
    ignored otherwise.
    """

    cls = config_class(config_cls)

    def decorator(function: Callable[[C], object]) -> EntryPoint:
        directory = _defining_dir(function) / Path(config_path)
        entry = Entry(function, cls, directory, config_name)

        @functools.wraps(function)
        def wrapper(args: Sequence[str] | None = None) -> None:
            run_cli(entry, wrapper, args)

        vars(wrapper)[ENTRY_ATTRIBUTE] = entry
        return wrapper

    return decorator


def compose[C](
    config_cls: TypeForm[C],
    config_dir: str | os.PathLike[str] | None = None,
    *,
    config_name: str | None = None,
    overrides: Sequence[str] = (),
) -> C:
    """Compose a config without the command line, for tests and notebooks.

    Args:
        config_cls: The root config class, a dataclass or a TypedDict.
        config_dir: The directory with the config files. A relative path is
            relative to the working directory.
        config_name: The primary config, as `--config-name` takes it.
        overrides: Overrides as they are written on the command line.

    Returns:
        An instance of `config_cls`; a dict for a TypedDict.
    """
    directory = Path(config_dir) if config_dir is not None else None
    composer = Composer(config_class(config_cls), directory, Settings)
    root = composer.compose(config_name, [parse_override(text) for text in overrides])
    resolve(root, datetime.now())
    return cast("C", build(root, skip=RESERVED_KEY))


def from_run_config[C](config_cls: TypeForm[C], data: Mapping[str, JSONValue]) -> C:
    """Build the config object from plain dicts, such as a saved `config.json`.

    This is how a run's `.nematocyte/config.json` becomes the object that the
    function of the run received:

        config = nematocyte.from_run_config(Config, json.loads(path.read_text()))

    It reads what `config.json` holds, the resolved tree of the run. It also
    takes the output of `to_dict`, which leaves out the values of `InitVar`
    fields; those then take the default in the class.

    The mapping of a group names its option under the key ``_option_``. A
    key that is left out takes the default in the class. Every value is checked as a value from a
    config file is, but no interpolation is resolved and no config file is
    read.

    Args:
        config_cls: The root config class, a dataclass or a TypedDict.
        data: The config, as plain dicts, lists and scalars.

    Returns:
        An instance of `config_cls`; a dict for a TypedDict.
    """
    cls = config_class(config_cls)
    validate_schema(cls)
    root = node_from_plain(
        cls, data, source="the dict given to from_run_config()", checked=False
    )
    return cast("C", build(root))
