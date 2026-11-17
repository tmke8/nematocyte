"""Finding config files and combining them, with the overrides, into one tree."""

from collections.abc import Sequence
import difflib
from pathlib import Path, PurePosixPath
import re
from typing import Any

from ._errors import ConfigError
from ._files import SUFFIXES, ConfigFile, find_file, list_names, load_file
from ._overrides import Override
from ._schema import (
    RESERVED_KEY,
    SELF_KEY,
    FieldInfo,
    Kind,
    class_fields,
    validate_schema,
)
from ._tree import Node, format_path, make_root, require_child, unknown_key_message
from ._tree import Path as KeyPath

__all__ = ["Composer"]

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _error(source: str, message: str) -> ConfigError:
    return ConfigError(f"Error in {source}:\n{message}")


def _did_you_mean(name: str, names: Sequence[str]) -> str:
    close = difflib.get_close_matches(name, names, n=1)
    return f"\nDid you mean {close[0]!r}?" if close else ""


class Composer:
    """Builds the config tree for one root dataclass and one config directory."""

    def __init__(
        self, config_cls: type, config_dir: Path | None, settings_cls: type
    ) -> None:
        validate_schema(config_cls)
        validate_schema(settings_cls, root=False)
        if config_dir is not None and not config_dir.is_dir():
            raise ConfigError(f"The config directory {config_dir} does not exist.")
        self.config_cls = config_cls
        self.config_dir = config_dir
        self.settings_cls = settings_cls

    # -- the public steps -------------------------------------------------

    def compose(
        self, config_name: str | None = None, overrides: Sequence[Override] = ()
    ) -> Node:
        """The tree for a primary config (or none) and a list of overrides."""
        root = make_root(self.config_cls, self.settings_cls)
        if config_name is not None and config_name != "":
            self._apply_primary(root, config_name, "--config-name", ())
        self.apply_overrides(root, overrides)
        return root

    def apply_overrides(self, root: Node, overrides: Sequence[Override]) -> None:
        """Apply command-line overrides: selections first, then values.

        The order of the overrides has no meaning. Selections are applied
        shortest path first, so that `model=cnn` is in place before
        `model.norm=batchnorm` is looked at.
        """
        seen: dict[KeyPath, Override] = {}
        for override in overrides:
            if override.key in seen:
                raise ConfigError(
                    f"The key {override.key_text!r} is given twice: "
                    f"{seen[override.key].text!r} and {override.text!r}. "
                    "The order of overrides has no meaning, so neither could win."
                )
            seen[override.key] = override

        values: list[Override] = []
        for override in sorted(overrides, key=lambda o: len(o.key)):
            source = repr(override.text)
            parent, info = self._locate(root, override.key, source)
            if info.kind is Kind.VALUE:
                values.append(override)
                continue
            name = self._selection_name(override, info)
            self._select(parent, info, name, source, ())
        for override in values:
            source = repr(override.text)
            parent, info = self._locate(root, override.key, source)
            if override.is_sweep:
                raise _error(
                    source,
                    "The value is a sweep over several values. Add --multirun "
                    "to run one job per value, or quote the value to pass it "
                    "as a string.",
                )
            parent.set_value(info.name, override.value, f"the override {source}")

    # -- where the files are ----------------------------------------------

    def directory(self, path: KeyPath) -> Path | None:
        """The directory that holds the named configs for the field at `path`."""
        if self.config_dir is None:
            return None
        return self.config_dir.joinpath(*path)

    def named_configs(self, path: KeyPath) -> list[str]:
        """The names of the config files for the field at `path`."""
        directory = self.directory(path)
        return list_names(directory) if directory is not None else []

    def root_field_names(self) -> set[str]:
        return {*class_fields(self.config_cls), RESERVED_KEY}

    def primary_file(self, name: str, source: str) -> Path:
        """The file of the primary config called `name`."""
        if self.config_dir is None:
            raise _error(
                source,
                f"There is no config directory in which to look for the "
                f"primary config {name!r}.",
            )
        clean = name
        for suffix in SUFFIXES:
            clean = clean.removesuffix(suffix)
        path = PurePosixPath(clean)
        if not path.parts or path.is_absolute() or ".." in path.parts:
            raise _error(source, f"{name!r} is not the name of a primary config.")
        first = path.parts[0]
        if len(path.parts) > 1 and first in self.root_field_names():
            raise _error(
                source,
                f"{name!r} is not a primary config: the directory "
                f"{self.config_dir.name}/{first}/ is named after the field "
                f"{first!r}, so its files are named configs for that field. "
                f"Select one with {first}={path.name}.",
            )
        found = find_file(self.config_dir / path.parent, path.name)
        if found is None:
            available = self.primary_configs()
            listing = (
                f"\nPrimary configs: {', '.join(available)}"
                if available
                else f"\nThere are no primary configs in {self.config_dir}."
            )
            raise _error(
                source,
                f"There is no primary config called {name!r} in "
                f"{self.config_dir}.{listing}{_did_you_mean(clean, available)}",
            )
        return found

    def primary_configs(self) -> list[str]:
        """The names of all primary configs, as `--config-name` takes them."""
        if self.config_dir is None:
            return []
        reserved = self.root_field_names()
        names: list[str] = []
        for path in sorted(self.config_dir.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            relative = path.relative_to(self.config_dir)
            if len(relative.parts) > 1 and relative.parts[0] in reserved:
                continue
            names.append(relative.with_suffix("").as_posix())
        return names

    # -- applying files ----------------------------------------------------

    def _apply_primary(
        self, root: Node, name: str, source: str, chain: tuple[Path, ...]
    ) -> None:
        path = self.primary_file(name, source)
        if path in chain:
            raise _error(
                source,
                f"The primary config {name!r} extends itself through '_self_': "
                + " -> ".join(p.name for p in (*chain, path)),
            )
        file = load_file(path)
        base = file.defaults.get(SELF_KEY)
        if base is not None:
            self._apply_primary(root, base, file.label, (*chain, path))
        root.touched = True
        self._apply_body(root, file)

    def _apply_body(self, node: Node, file: ConfigFile) -> None:
        """Apply the `defaults` of a file other than `_self_`, then its values."""
        entries = [
            (tuple(key.split(".")), key, name)
            for key, name in file.defaults.items()
            if key != SELF_KEY
        ]
        # Shortest path first: `model.norm` is inside `model` and is applied later.
        for path, key, name in sorted(entries, key=lambda entry: len(entry[0])):
            source = f"{file.label}, in the entry {key!r} of 'defaults'"
            if not all(_NAME.fullmatch(segment) for segment in path):
                raise _error(
                    source,
                    "A key in 'defaults' is a field, a dotted path to a nested "
                    "field, or '_self_'.",
                )
            parent, info = self._locate(node, path, source)
            if info.kind is Kind.VALUE:
                raise _error(
                    source,
                    f"{key!r} is a value of {parent.describe()}, not a subconfig "
                    "or a group, so there is nothing to select for it. Set it "
                    "as a value, outside 'defaults'.",
                )
            self._select(parent, info, name, source, ())
        self._apply_values(node, file.values, file.label)

    def _apply_values(self, node: Node, values: dict[Any, Any], label: str) -> None:
        for key, value in values.items():
            info = node.fields.get(key) if isinstance(key, str) else None
            if info is None:
                raise _error(label, unknown_key_message(node, str(key)))
            if info.kind is Kind.VALUE:
                node.set_value(info.name, value, label)
                continue
            path = format_path((*node.path, info.name))
            if value is None:
                continue
            if not isinstance(value, dict):
                hint = ""
                if isinstance(value, str):
                    hint = (
                        f"\nTo select the config called {value!r} for {path!r}, "
                        f"write it under 'defaults'."
                    )
                what = "a group" if info.kind is Kind.GROUP else "a subconfig"
                raise _error(
                    label,
                    f"{path!r} is {what}, so its value has to be a mapping of "
                    f"its fields, not {value!r}.{hint}",
                )
            try:
                child = require_child(node, info)
            except ConfigError as exc:
                raise _error(label, str(exc)) from None
            child.touched = True
            self._apply_values(child, value, label)

    # -- selections --------------------------------------------------------

    def _locate(
        self, start: Node, path: KeyPath, source: str
    ) -> tuple[Node, FieldInfo]:
        """Follow `path` from `start`; return the last node and the field in it."""
        node = start
        for index, segment in enumerate(path):
            info = node.fields.get(segment)
            if info is None:
                raise _error(source, unknown_key_message(node, segment))
            if index == len(path) - 1:
                return node, info
            if info.kind is Kind.VALUE:
                here = format_path((*node.path, segment))
                rest = format_path(path[index + 1 :])
                raise _error(
                    source,
                    f"{here!r} is a value, not a subconfig, so it has no key "
                    f"{rest!r}. A value such as a dict is set as a whole: "
                    f"{here}={{...}}.",
                )
            try:
                node = require_child(node, info)
            except ConfigError as exc:
                raise _error(source, str(exc)) from None
        raise AssertionError("an empty path")

    def _selection_name(self, override: Override, info: FieldInfo) -> str:
        source = repr(override.text)
        key = override.key_text
        what = "a config group" if info.kind is Kind.GROUP else "a subconfig"
        if override.is_sweep:
            raise _error(
                source,
                "The value is a sweep over several values. Add --multirun to "
                "run one job per value.",
            )
        if isinstance(override.value, dict):
            raise _error(
                source,
                f"{key!r} is {what} and cannot be given a dict value. Set its "
                f"fields one by one, as in {key}.field=value.",
            )
        if not isinstance(override.value, str):
            raise _error(
                source,
                f"{key!r} is {what}, so the value has to be the name of a "
                f"config for it, not {override.value!r}."
                f"{self._available(override.key, info)}",
            )
        return override.value

    def _available(self, path: KeyPath, info: FieldInfo) -> str:
        lines = ""
        if info.options:
            lines += f"\nOptions: {', '.join(info.options)}"
        files = self.named_configs(path)
        if files:
            label = "From files" if info.kind is Kind.GROUP else "Presets"
            lines += f"\n{label}: {', '.join(files)}"
        elif info.kind is Kind.CONFIG:
            directory = self.directory(path)
            where = f" (looked in {directory})" if directory is not None else ""
            lines += f"\nThere are no config files for {format_path(path)!r}{where}."
        return lines

    def _select(
        self,
        parent: Node,
        info: FieldInfo,
        name: str,
        source: str,
        chain: tuple[Path, ...],
    ) -> None:
        """Select the config called `name` for the field `info` of `parent`."""
        path = (*parent.path, info.name)
        if info.kind is Kind.GROUP and name in info.options:
            # Naming an option starts the node again from the class defaults.
            node = Node(info.options[name], path, option=name)
            node.touched = True
            parent.children[info.name] = node
            return

        directory = self.directory(path)
        file_path = find_file(directory, name) if directory is not None else None
        if file_path is None:
            candidates = [*info.options, *self.named_configs(path)]
            kind = "option or config" if info.kind is Kind.GROUP else "config"
            raise _error(
                source,
                f"{format_path(path)!r} has no {kind} called {name!r}."
                f"{self._available(path, info)}{_did_you_mean(name, candidates)}",
            )
        if file_path in chain:
            raise _error(
                source,
                f"The config {name!r} for {format_path(path)!r} extends itself "
                "through '_self_': " + " -> ".join(p.name for p in (*chain, file_path)),
            )
        file = load_file(file_path)
        base = file.defaults.get(SELF_KEY)
        if base is not None:
            self._select(
                parent,
                info,
                base,
                f"{file.label}, in the entry '_self_' of 'defaults'",
                (*chain, file_path),
            )
        elif info.kind is Kind.GROUP:
            raise _error(
                file.label,
                f"A config for the group {format_path(path)!r} has to say which "
                f"option it extends:\n    defaults:\n      _self_: <option>"
                f"{self._available(path, info)}",
            )
        node = parent.children[info.name]
        node.touched = True
        self._apply_body(node, file)
