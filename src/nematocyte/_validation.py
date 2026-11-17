"""Checking every config file, without running anything."""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

from ._build import build
from ._compose import Composer
from ._errors import ConfigError
from ._files import SUFFIXES
from ._interpolation import resolve
from ._multirun import check_params, parse_params
from ._overrides import Override
from ._schema import RESERVED_KEY, FieldInfo, Kind, class_fields
from ._settings import Settings
from ._tree import Path as KeyPath

__all__ = ["ValidationResult", "validate_config", "validate_configs"]


@dataclass(frozen=True)
class ValidationResult:
    """The verdict on one config file."""

    path: Path
    """The path of the file, relative to the config directory."""
    error: ConfigError | None
    """What is wrong with the file, or `None`."""

    @property
    def ok(self) -> bool:
        return self.error is None


def _classify(composer: Composer, relative: Path) -> KeyPath | None:
    """The field that a file is a named config for; `None` for a primary config."""
    root_fields = dict(class_fields(composer.config_cls))
    root_fields[RESERVED_KEY] = FieldInfo(
        name=RESERVED_KEY,
        kind=Kind.CONFIG,
        annotation=composer.settings_cls,
        owner=None,
        config_cls=composer.settings_cls,
    )
    candidates = [root_fields]
    path: list[str] = []
    for index, part in enumerate(relative.parts[:-1]):
        infos = [fields[part] for fields in candidates if part in fields]
        here = "/".join(relative.parts[: index + 1])
        if not infos:
            if index == 0:
                return None
            parent = ".".join(path)
            raise ConfigError(
                f"{relative.as_posix()} belongs nowhere: the directory {here}/ "
                f"is inside the directory of the field {parent!r}, but "
                f"{part!r} is not a field of {parent!r} that is a subconfig or "
                "a group, so its files can be neither named configs nor "
                "primary configs."
            )
        nested = [info for info in infos if info.kind is not Kind.VALUE]
        if not nested:
            raise ConfigError(
                f"{relative.as_posix()} belongs nowhere: the directory {here}/ "
                f"is named after the field {'.'.join([*path, part])!r}, which "
                "is a value and not a subconfig or a group, so it has no "
                "named configs. A directory named after a field holds no "
                "primary configs either."
            )
        path.append(part)
        candidates = [
            class_fields(cls)
            for info in nested
            for cls in (info.options.values() if info.options else [info.config_cls])
            if cls is not None
        ]
    return tuple(path) if path else None


def _selection(
    composer: Composer, config_name: str | None, path: KeyPath, name: str
) -> list[Override]:
    """The overrides that select the config `name` for the field at `path`.

    Where the path goes through a group whose selected option does not have
    the next field, the first option that has it is selected.
    """
    overrides: list[Override] = []
    for depth in range(1, len(path)):
        root = composer.compose(config_name, overrides)
        node = root
        for segment in path[: depth - 1]:
            node = node.children[segment]
        info = node.fields[path[depth - 1]]
        child = node.children.get(info.name)
        following = path[depth]
        if child is not None and following in child.fields:
            continue
        for option_name, option_cls in info.options.items():
            found = class_fields(option_cls).get(following)
            if found is not None and found.kind is not Kind.VALUE:
                key = path[:depth]
                overrides.append(
                    Override(key, option_name, f"{'.'.join(key)}={option_name}")
                )
                break
    overrides.append(Override(path, name, f"{'.'.join(path)}={name}"))
    return overrides


def _validate(
    composer: Composer, relative: Path, config_name: str | None, *, construct: bool
) -> None:
    if composer.config_dir is None or not (composer.config_dir / relative).is_file():
        raise ConfigError(f"There is no config file {relative.as_posix()}.")
    if relative.suffix not in SUFFIXES:
        raise ConfigError(f"{relative.as_posix()} is not a YAML file.")
    path = _classify(composer, relative)
    if path is None:
        root = composer.compose(relative.with_suffix("").as_posix())
    else:
        overrides = _selection(composer, config_name, path, relative.stem)
        root = composer.compose(config_name, overrides)
    resolve(root, datetime.now())
    build(root, skip=RESERVED_KEY, construct=construct)
    # nematocyte's own settings are always built: their classes are its own.
    settings = build(root.children[RESERVED_KEY])
    assert isinstance(settings, Settings)
    if settings.sweeper.params is not None:
        where = relative.as_posix()
        check_params(
            parse_params(settings.sweeper.params, where), composer.config_cls, where
        )


def validate_config(
    config_cls: type,
    config_dir: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    config_name: str | None = None,
    construct: bool = True,
) -> ValidationResult:
    """Check one config file, in the same way as when the program runs.

    Args:
        config_cls: The root dataclass.
        config_dir: The directory with the config files.
        path: The file to check, relative to `config_dir`.
        config_name: A primary config that supplies the rest of the tree
            when a named config is checked. Only needed if some field of the
            root dataclass has no default.
        construct: With `False`, the dataclasses are not constructed, so no
            `__post_init__` runs. Every other check is still made.
    """
    directory = Path(config_dir)
    relative = Path(path)
    if relative.is_absolute():
        relative = relative.relative_to(directory.absolute())
    try:
        composer = Composer(config_cls, directory, Settings)
        _validate(composer, relative, config_name, construct=construct)
    except ConfigError as exc:
        return ValidationResult(relative, exc)
    return ValidationResult(relative, None)


def validate_configs(
    config_cls: type,
    config_dir: str | os.PathLike[str],
    *,
    config_name: str | None = None,
    construct: bool = True,
) -> list[ValidationResult]:
    """Check every YAML file in the config directory; one result per file.

    The arguments are those of :func:`validate_config`.
    """
    directory = Path(config_dir)
    if not directory.is_dir():
        raise ConfigError(f"The config directory {directory} does not exist.")
    files = sorted(
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in SUFFIXES
    )
    return [
        validate_config(
            config_cls, directory, path, config_name=config_name, construct=construct
        )
        for path in files
    ]
