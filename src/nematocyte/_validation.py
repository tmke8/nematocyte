"""Checking every config file, without running anything."""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any
from typing_extensions import TypeForm

from ._build import build
from ._compose import Composer
from ._errors import ConfigError
from ._files import SUFFIXES
from ._interpolation import resolve
from ._multirun import check_params, parse_params
from ._overrides import Override
from ._schema import RESERVED_KEY, FieldInfo, Kind, class_fields, config_class
from ._settings import Settings
from ._tree import Path as KeyPath

__all__ = ["ValidationResult", "Validator", "validate_configs"]


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


def _classify(composer: Composer, relative: Path) -> tuple[KeyPath, str] | None:
    """The field that a file is a named config for, and the name it has there.

    `None` for a primary config.
    """
    root_fields = dict(class_fields(composer.config_cls))
    root_fields[RESERVED_KEY] = FieldInfo(
        name=RESERVED_KEY,
        kind=Kind.CONFIG,
        annotation=composer.settings_cls,
        owner=None,
        config_cls=composer.settings_cls,
    )
    candidates = [root_fields]
    options: set[str] = set()
    path: list[str] = []
    directories = relative.parts[:-1]
    for index, part in enumerate(directories):
        infos = [fields[part] for fields in candidates if part in fields]
        here = "/".join(relative.parts[: index + 1])
        parent = ".".join(path)
        # A field keeps its directory when an option has the same name.
        if part in options and not infos:
            if index != len(directories) - 1:
                raise ConfigError(
                    f"{relative.as_posix()} belongs nowhere: the directory "
                    f"{here}/ is named after the option {part!r} of "
                    f"{parent!r}, and holds configs of {parent!r}, but no "
                    f"directories."
                )
            return tuple(path), f"{part}/{relative.stem}"
        if not infos:
            if index == 0:
                return None
            raise ConfigError(
                f"{relative.as_posix()} belongs nowhere: the directory {here}/ "
                f"is inside the directory of the field {parent!r}, but "
                f"{part!r} is neither a field of {parent!r} that is a "
                "subconfig or a group, nor an option of it, so its files can "
                "be neither named configs nor primary configs."
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
        options = {name for info in nested for name in info.options}
        candidates = [
            class_fields(cls)
            for info in nested
            for cls in (info.options.values() if info.options else [info.config_cls])
            if cls is not None
        ]
    return (tuple(path), relative.stem) if path else None


def _selections(
    composer: Composer,
    config_name: str | None,
    path: KeyPath,
    name: str,
    *,
    overrides: tuple[Override, ...] = (),
    depth: int = 1,
) -> list[list[Override]]:
    """The ways to select the config `name` for the field at `path`.

    Where the path goes through a group, there is one way for each option
    that has the next field: the option that is selected, if it has it, and
    every other one, selected with an override.
    """
    if depth == len(path):
        return [[*overrides, Override(path, name, f"{'.'.join(path)}={name}")]]
    root = composer.compose(config_name, overrides, warn=False)
    node = root
    for segment in path[: depth - 1]:
        node = node.children[segment]
    info = node.fields[path[depth - 1]]
    child = node.children.get(info.name)
    following = path[depth]
    choices: list[tuple[Override, ...]] = []
    if child is not None and following in child.fields:
        choices.append(overrides)
    for option_name, option_cls in info.options.items():
        if child is not None and option_cls is child.cls:
            continue
        found = class_fields(option_cls).get(following)
        if found is not None and found.kind is not Kind.VALUE:
            key = path[:depth]
            selection = Override(key, option_name, f"{'.'.join(key)}={option_name}")
            choices.append((*overrides, selection))
    return [
        selection
        for choice in choices
        for selection in _selections(
            composer, config_name, path, name, overrides=choice, depth=depth + 1
        )
    ]


def _check_tree(
    composer: Composer,
    config_name: str | None,
    overrides: list[Override],
    relative: Path,
) -> None:
    # The selections that the validator makes itself are in no file, so what
    # they drop is not warned about; a primary config is composed as in a run.
    root = composer.compose(config_name, overrides, warn=not overrides)
    resolve(root, datetime.now())
    build(root, skip=RESERVED_KEY)
    # nematocyte's own settings are always built: their classes are its own.
    settings = build(root.children[RESERVED_KEY])
    assert isinstance(settings, Settings)
    if settings.sweeper.params is not None:
        where = relative.as_posix()
        check_params(
            parse_params(settings.sweeper.params, where), composer.config_cls, where
        )


def _validate(composer: Composer, relative: Path, config_name: str | None) -> None:
    if composer.config_dir is None or not (composer.config_dir / relative).is_file():
        raise ConfigError(f"There is no config file {relative.as_posix()}.")
    if relative.suffix not in SUFFIXES:
        raise ConfigError(
            f"{relative.as_posix()} is not a config file: those end in "
            f"{', '.join(SUFFIXES)}."
        )
    found = _classify(composer, relative)
    if found is None:
        name = relative.with_suffix("").as_posix()
        _check_tree(composer, name, [], relative)
        return
    path, name = found
    selections = _selections(composer, config_name, path, name)
    if len(selections) == 1:
        _check_tree(composer, config_name, selections[0], relative)
        return
    # The file is checked with every option that has its field; it has to fit
    # all of them, since any of them can be selected when it is used.
    failures: list[str] = []
    for overrides in selections:
        try:
            _check_tree(composer, config_name, overrides, relative)
        except ConfigError as exc:
            shown = " ".join(override.text for override in overrides[:-1])
            failures.append(f"With {shown or 'the option that is selected'}:\n{exc}")
    if failures:
        raise ConfigError(
            f"{relative.as_posix()} is a config for {'.'.join(path)!r}, which "
            "more than one option has, and it has to fit each of them.\n"
            + "\n".join(failures)
        )


class Validator:
    """Checks config files one at a time, in the same way as when the program runs.

    The schema of the config class and the top of the config directory are
    checked once, when the validator is made; each call to :meth:`validate`
    then checks one file. This suits a parametrised test with one case per
    file.

    Args:
        config_cls: The root config class, a dataclass or a TypedDict.
        config_dir: The directory with the config files.
        config_name: A primary config that supplies the rest of the tree
            when a named config is checked. Only needed if some field of the
            root config class has no default.

    Raises:
        SchemaError: If the config class is declared in an unsupported way.
        ConfigError: If the config directory does not exist, or a file at
            its top is named after a subconfig or a group.
    """

    def __init__(
        self,
        config_cls: TypeForm[Any],
        config_dir: str | os.PathLike[str],
        *,
        config_name: str | None = None,
    ) -> None:
        self._directory = Path(config_dir)
        self._composer = Composer(config_class(config_cls), self._directory, Settings)
        self._config_name = config_name

    def validate(self, path: str | os.PathLike[str]) -> ValidationResult:
        """Check one config file.

        Args:
            path: The file to check, relative to the config directory, or an
                absolute path inside it.
        """
        relative = Path(path)
        if relative.is_absolute():
            relative = relative.relative_to(self._directory.absolute())
        try:
            _validate(self._composer, relative, self._config_name)
        except ConfigError as exc:
            return ValidationResult(relative, exc)
        return ValidationResult(relative, None)


def validate_configs(
    config_cls: TypeForm[Any],
    config_dir: str | os.PathLike[str],
    *,
    config_name: str | None = None,
) -> list[ValidationResult]:
    """Check every config file in the config directory; one result per file.

    The arguments are those of :class:`Validator`, which also says what is
    raised instead of being returned as a result.
    """
    directory = Path(config_dir)
    if not directory.is_dir():
        raise ConfigError(f"The config directory {directory} does not exist.")
    validator = Validator(config_cls, directory, config_name=config_name)
    files = sorted(
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in SUFFIXES
    )
    return [validator.validate(path) for path in files]
