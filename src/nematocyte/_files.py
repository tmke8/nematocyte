"""Reading config files: YAML 1.2, with ruamel.yaml."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from ._errors import ConfigError
from ._schema import DEFAULTS_KEY

__all__ = [
    "SUFFIXES",
    "ConfigFile",
    "check_unique",
    "display_path",
    "find_file",
    "list_names",
    "load_file",
]

SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class ConfigFile:
    """The content of one config file."""

    path: Path
    label: str
    """The path of the file as it is shown in messages."""
    defaults: dict[str, str]
    values: dict[str, Any]


def display_path(path: Path) -> str:
    """The path as it is shown in messages: relative to the working directory."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_file(path: Path) -> ConfigFile:
    """Read a config file and split off its `defaults`."""
    label = display_path(path)
    yaml = YAML(typ="safe", pure=True)
    yaml.allow_duplicate_keys = False
    try:
        with path.open(encoding="utf-8") as stream:
            content = yaml.load(stream)
    except YAMLError as exc:
        raise ConfigError(f"{label} is not valid YAML:\n{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{label} cannot be read: {exc}") from exc
    if content is None:
        content = {}
    if not isinstance(content, dict):
        raise ConfigError(
            f"{label}: a config file has to be a mapping at the top level, "
            f"not a {type(content).__name__}."
        )
    _check_keys(content, label, ())

    defaults = content.pop(DEFAULTS_KEY, None)
    if defaults is None:
        defaults = {}
    if isinstance(defaults, list):
        raise ConfigError(
            f"{label}: 'defaults' is a list, as in Hydra. In nematocyte it is "
            "a mapping from a field to the name of the config to load there, "
            "such as\n    defaults:\n      model: cnn"
        )
    if not isinstance(defaults, dict):
        raise ConfigError(f"{label}: 'defaults' has to be a mapping.")
    for key, name in defaults.items():
        if not isinstance(name, str):
            hint = (
                " An entry with null is not needed in nematocyte; remove it."
                if name is None
                else ""
            )
            raise ConfigError(
                f"{label}: the entry {key!r} in 'defaults' has to name a config, "
                f"but is {name!r}.{hint}"
            )
    return ConfigFile(path=path, label=label, defaults=defaults, values=content)


def _check_keys(mapping: dict[Any, Any], label: str, path: tuple[str, ...]) -> None:
    """Keys are strings, down to where the schema could still have a dataclass."""
    for key in mapping:
        if not isinstance(key, str):
            where = f" under {'.'.join(path)!r}" if path else ""
            # Only reported at the top level; further down, a key that is
            # not a string can be a key of a dict-typed field.
            if not path:
                raise ConfigError(f"{label}: the key {key!r}{where} is not a string.")


def _clash(paths: list[Path]) -> ConfigError:
    names = " and ".join(display_path(path) for path in paths)
    return ConfigError(
        f"{names} are both there, so the name {paths[0].stem!r} is ambiguous. "
        "Remove or rename one of them."
    )


def find_file(directory: Path, name: str) -> Path | None:
    """The config file called `name` in `directory`, if there is one.

    A name with a file for each suffix is an error.
    """
    found = [
        candidate
        for suffix in SUFFIXES
        if (candidate := directory / f"{name}{suffix}").is_file()
    ]
    if len(found) > 1:
        raise _clash(found)
    return found[0] if found else None


def check_unique(paths: list[Path]) -> None:
    """Raise if two of the config files `paths` differ only in their suffix."""
    seen: dict[Path, Path] = {}
    for path in sorted(paths):
        stem = path.with_suffix("")
        if stem in seen:
            raise _clash([seen[stem], path])
        seen[stem] = path


def list_names(directory: Path) -> list[str]:
    """The names of the config files in `directory`, sorted."""
    if not directory.is_dir():
        return []
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in SUFFIXES
    ]
    check_unique(paths)
    return sorted(path.stem for path in paths)
