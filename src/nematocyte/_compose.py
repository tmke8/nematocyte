"""Finding config files and combining them, with the overrides, into one tree."""

from collections.abc import Collection, Iterator, Sequence
from dataclasses import MISSING, dataclass
import difflib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any
import warnings

from pydantic import ValidationError

from ._errors import ConfigError, DroppedValueWarning
from ._files import (
    SUFFIXES,
    ConfigFile,
    check_unique,
    display_path,
    find_file,
    list_files,
    load_file,
)
from ._overrides import AddedConfig, Override, format_value, last_wins
from ._schema import (
    DEFAULTS_KEY,
    DOC_KEY,
    RESERVED_KEY,
    SELF_KEY,
    FieldInfo,
    Kind,
    class_fields,
    field_adapter,
    first_paragraph,
    is_typeddict_class,
    validate_schema,
)
from ._tree import (
    Node,
    Selection,
    format_path,
    make_root,
    next_step,
    require_child,
    switched_off_message,
    unknown_key_message,
)
from ._tree import Path as KeyPath

__all__ = ["Composer"]

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The name of the key in Hydra, which a config file may still use by mistake.
_HYDRA_DEFAULTS_KEY = "defaults"

# A warning points at the first frame outside nematocyte: the caller's code.
_PACKAGE_DIR = str(Path(__file__).parent) + os.sep


def _error(source: str, message: str) -> ConfigError:
    return ConfigError(f"Error in {source}:\n{message}")


def _hydra_defaults_hint(file: ConfigFile, fields: Collection[str]) -> str:
    """A hint if the file has Hydra's key `defaults`, and it is not a field."""
    if _HYDRA_DEFAULTS_KEY not in file.values or _HYDRA_DEFAULTS_KEY in fields:
        return ""
    return (
        f"\nThe file has a key {_HYDRA_DEFAULTS_KEY!r}, as in Hydra. In "
        f"nematocyte, the configs that a file builds on are listed under "
        f"{DEFAULTS_KEY!r}."
    )


def _option_fields(info: FieldInfo) -> set[str]:
    """The names of the fields of all options of a group."""
    return {name for cls in info.options.values() for name in class_fields(cls)}


def _option_directories(info: FieldInfo) -> list[str]:
    """The options of a group that have a directory, in the group's directory.

    An option that is also a field of an option has none: the directory with
    its name belongs to the field.
    """
    fields = _option_fields(info)
    return [option for option in info.options if option not in fields]


def _file_doc(path: Path) -> str:
    """The first paragraph of the `_doc_` of the config file at `path`."""
    return first_paragraph(load_file(path).doc)


def _is_file_name(name: str) -> bool:
    """Whether `name` is a plain file name on every platform.

    Not `.` or `..`, and without a directory or a drive, so that the file is
    looked for in the directory it is named in, on Linux, macOS and Windows.
    Any other character is allowed: a quoted override or a name in a file can
    be any string.
    """
    return (
        name not in ("", ".", "..")
        and PurePosixPath(name).name == name
        and PureWindowsPath(name).name == name
    )


def _did_you_mean(name: str, names: Sequence[str]) -> str:
    close = difflib.get_close_matches(name, names, n=1)
    return f"\nDid you mean {close[0]!r}?" if close else ""


class Composer:
    """Builds the config tree for one root config class and one config directory."""

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
        self._drops: list[_Drop] | None = None
        """While a tree is composed with warnings: what the selections dropped."""
        self._check_top_level_files()

    def _check_top_level_files(self) -> None:
        """A file at the top named after a subconfig or a group is an error.

        Its name says "the config of that field", but at the top it would be
        a primary config; the named configs of the field are in its directory.
        """
        if self.config_dir is None:
            return
        nested = {
            name
            for name, info in class_fields(self.config_cls).items()
            if info.kind is not Kind.VALUE
        }
        nested.add(RESERVED_KEY)
        for path in sorted(self.config_dir.iterdir()):
            if path.is_file() and path.suffix in SUFFIXES and path.stem in nested:
                directory = f"{self.config_dir.name}/{path.stem}/"
                raise _error(
                    display_path(path),
                    f"The file is named after the field {path.stem!r}, but "
                    f"a file at the top of the config directory is a primary "
                    f"config, for the whole tree. A config for {path.stem!r} "
                    f"goes in {directory}, and is selected with "
                    f"{path.stem}=NAME. A primary config needs another name.",
                )

    # -- the public steps -------------------------------------------------

    def compose(
        self,
        config_name: str | None = None,
        overrides: Sequence[Override] = (),
        added: Sequence[AddedConfig] = (),
        *,
        warn: bool = True,
    ) -> Node:
        """The tree for a primary config (or none), added configs and overrides.

        Each added config (`+NAME`) is a primary config that is applied on top
        of the tree so far, in the order given, as if it extended the primary
        config with `_self_`. The overrides come last.

        With `warn`, a selection that drops a value which a lower layer set,
        and which nothing sets again, gives a `DroppedValueWarning`.
        """
        self._drops = [] if warn else None
        try:
            root = make_root(self.config_cls, self.settings_cls)
            if config_name is not None and config_name != "":
                self._apply_primary(root, config_name, "--config-name", ())
            for config in added:
                self._apply_primary(root, config.name, repr(config.text), ())
            self.apply_overrides(root, overrides)
            drops = self._drops
        finally:
            self._drops = None
        for drop in drops or ():
            lost = [item for item in drop.dropped if _is_lost(root, drop.path, item)]
            if lost:
                warnings.warn(
                    drop.message(lost),
                    DroppedValueWarning,
                    skip_file_prefixes=(_PACKAGE_DIR,),
                )
        return root

    def apply_overrides(self, root: Node, overrides: Sequence[Override]) -> None:
        """Apply command-line overrides: selections first, then values.

        Apart from a key that is given more than once, where the last one
        wins, the order of the overrides has no meaning. Selections are
        applied shortest path first, so that `model=cnn` is in place before
        `model.norm=batchnorm` is looked at.
        """
        overrides = last_wins(overrides)
        values: list[Override] = []
        for override in sorted(overrides, key=lambda o: len(o.key)):
            source = repr(override.text)
            parent, info = self._locate(
                root, override.key, source, _plain_value(override)
            )
            if info.kind is Kind.VALUE:
                values.append(override)
                continue
            if override.unset:
                what = "a group" if info.kind is Kind.GROUP else "a subconfig"
                hint = (
                    f" To switch it off: {override.key_text}=null"
                    if info.optional
                    else ""
                )
                raise _error(
                    source,
                    f"{override.key_text!r} is {what} of {parent.describe()}, "
                    "so it cannot be left out. '~' leaves out a NotRequired "
                    f"key of a TypedDict.{hint}",
                )
            if info.kind is Kind.CONFIG and self._apply_subconfig_value(
                parent, info, override
            ):
                continue
            name = self._selection_name(override, info)
            self._select(parent, info, name, source, f"the override {source}")
        for override in values:
            source = repr(override.text)
            parent, info = self._locate(
                root, override.key, source, _plain_value(override)
            )
            if override.unset:
                _check_unset(override, parent, info)
                parent.unset_value(info.name, f"the override {source}")
                continue
            if override.is_sweep:
                raise _error(
                    source,
                    "The value is a sweep over several values. Add --multirun "
                    "to run one job per value, or quote the value to pass it "
                    "as a string.",
                )
            parent.set_value(info.name, override.value, f"the override {source}")

    def _apply_subconfig_value(
        self, parent: Node, info: FieldInfo, override: Override
    ) -> bool:
        """Apply `null` or a dict for a subconfig, as in a file.

        `null` switches an optional subconfig off; a dict is merged into the
        subconfig, and switches an optional one on. False if the value is
        neither, and so is the name of a preset.
        """
        source = repr(override.text)
        key = override.key_text
        if override.value is None:
            if not info.optional:
                assert info.config_cls is not None
                name = info.config_cls.__name__
                raise _error(
                    source,
                    f"{key!r} is a subconfig of {parent.describe()}, which "
                    f"cannot be switched off with null: its type is {name}, "
                    f"not {name} | None.",
                )
            parent.switch_off(info)
            return True
        if isinstance(override.value, dict):
            child = (
                parent.switch_on(info) if info.optional else require_child(parent, info)
            )
            child.touched = True
            self._apply_values(
                child, override.value, f"the override {source}", command_line=True
            )
            return True
        return False

    # -- where the files are ----------------------------------------------

    def directory(self, path: KeyPath) -> Path | None:
        """The directory that holds the named configs for the field at `path`."""
        if self.config_dir is None:
            return None
        return self.config_dir.joinpath(*path)

    def named_configs(self, path: KeyPath, info: FieldInfo) -> list[str]:
        """The names of the config files for the field `info` at `path`.

        For a group, the files in the directory of an option are included,
        as `option/name`.
        """
        return list(self._named_files(path, info))

    def named_config_docs(self, path: KeyPath, info: FieldInfo) -> dict[str, str]:
        """The named configs for the field `info` at `path`, with their docs.

        A doc is the first paragraph of the file's `_doc_`; empty if it has none.
        """
        return {
            name: _file_doc(file)
            for name, file in self._named_files(path, info).items()
        }

    def _named_files(self, path: KeyPath, info: FieldInfo) -> dict[str, Path]:
        """The config files for the field `info` at `path`, by name, sorted.

        The directory named after an option whose name is also a field of an
        option belongs to the field, and is left out.
        """
        directory = self.directory(path)
        if directory is None:
            return {}
        places = [
            (directory, ""),
            *(
                (directory / option, f"{option}/")
                for option in _option_directories(info)
            ),
        ]
        # A file that no name can select is left out; the validator reports it.
        return dict(
            sorted(
                (prefix + name, file)
                for place, prefix in places
                for name, file in list_files(place).items()
                if _is_file_name(name)
            )
        )

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
        # An absolute path starts with the part `/`, which is no file name.
        if not path.parts or not all(_is_file_name(part) for part in path.parts):
            raise _error(
                source,
                f"{name!r} is not the name of a primary config. It is the "
                f"path of the file in {self.config_dir.name}/, with slashes "
                f"between the directories, without '..', a backslash or a "
                f"drive.",
            )
        first = path.parts[0]
        if len(path.parts) > 1 and first in self.root_field_names():
            info = class_fields(self.config_cls).get(first)
            if info is not None and info.kind is Kind.VALUE:
                raise _error(
                    source,
                    f"{name!r} is not a primary config: the directory "
                    f"{self.config_dir.name}/{first}/ is named after the field "
                    f"{first!r}, which is a value and not a subconfig or a "
                    f"group, so it has no named configs, and a directory "
                    f"named after a field holds no primary configs either. "
                    f"Move the file to the top of {self.config_dir.name}/ "
                    f"or to a directory that matches no field.",
                )
            middle = path.parts[1:-1]
            if (
                info is not None
                and len(middle) == 1
                and middle[0] in _option_directories(info)
            ):
                # The directory of an option holds configs of the group.
                selection = f"{first}={middle[0]}/{path.name}"
            else:
                # A nested directory holds the configs of a nested field.
                selection = f"{format_path((first, *middle))}={path.name}"
            raise _error(
                source,
                f"{name!r} is not a primary config: the directory "
                f"{self.config_dir.name}/{first}/ is named after the field "
                f"{first!r}, so its files are named configs for that field. "
                f"Select one with {selection}.",
            )
        found = find_file(self.config_dir / path.parent, path.name)
        if found is None:
            available = self.primary_configs()
            listing = (
                f"\nPrimary configs: {', '.join(available)}"
                if available
                else f"\nThere are no primary configs in {self.config_dir}."
            )
            dotted = ""
            if "." in clean:
                slashed = clean.replace(".", "/")
                dotted = (
                    f"\nA primary config in a subdirectory is named with "
                    f"slashes, as in {slashed!r}."
                )
            raise _error(
                source,
                f"There is no primary config called {name!r} in "
                f"{self.config_dir}.{listing}{dotted}"
                f"{_did_you_mean(clean, available)}",
            )
        return found

    def primary_configs(self) -> list[str]:
        """The names of all primary configs, as `--config-name` takes them."""
        if self.config_dir is None:
            return []
        reserved = self.root_field_names()
        paths: list[Path] = []
        for path in sorted(self.config_dir.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            relative = path.relative_to(self.config_dir)
            if len(relative.parts) > 1 and relative.parts[0] in reserved:
                continue
            # A file that no name can select is left out; the validator
            # reports it.
            if not all(_is_file_name(part) for part in relative.with_suffix("").parts):
                continue
            paths.append(path)
        check_unique(paths)
        return [
            path.relative_to(self.config_dir).with_suffix("").as_posix()
            for path in paths
        ]

    def primary_config_docs(self) -> dict[str, str]:
        """The primary configs, as `--config-name` takes them, with their docs."""
        return {
            name: _file_doc(self.primary_file(name, "--help"))
            for name in self.primary_configs()
        }

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
        """Apply the `_defaults_` of a file other than `_self_`, then its values."""
        if hint := _hydra_defaults_hint(file, node.fields):
            raise _error(
                file.label,
                f"{_HYDRA_DEFAULTS_KEY!r} is not a field of {node.describe()}.{hint}",
            )
        entries = [
            (tuple(key.split(".")), key, name)
            for key, name in file.defaults.items()
            if key != SELF_KEY
        ]
        # Shortest path first: `model.norm` is inside `model` and is applied later.
        for path, key, name in sorted(entries, key=lambda entry: len(entry[0])):
            source = f"{file.label}, in the entry {key!r} of '_defaults_'"
            if not all(_NAME.fullmatch(segment) for segment in path):
                raise _error(
                    source,
                    "A key in '_defaults_' is a field, a dotted path to a nested "
                    "field, or '_self_'.",
                )
            parent, info = self._locate(node, path, source)
            if info.kind is Kind.VALUE:
                raise _error(
                    source,
                    f"{key!r} is a value of {parent.describe()}, not a subconfig "
                    "or a group, so there is nothing to select for it. Set it "
                    "as a value, outside '_defaults_'.",
                )
            self._select(parent, info, name, source, file.label)
        self._apply_values(node, file.values, file.label)

    def _apply_values(
        self,
        node: Node,
        values: dict[Any, Any],
        label: str,
        *,
        command_line: bool = False,
    ) -> None:
        """Apply the values of a file, or of a dict on the command line.

        A mapping goes into a subconfig or a group, and switches an optional
        subconfig on; null switches it off, and is ignored for any other
        subconfig or group.
        """
        for key, value in values.items():
            info = node.fields.get(key) if isinstance(key, str) else None
            if info is None and key == DOC_KEY and not command_line:
                raise _error(
                    label,
                    f"'{DOC_KEY}' is under {node.describe()}, but it describes "
                    f"the whole file, so it goes at the top level of the file.",
                )
            if info is None:
                raise _error(label, unknown_key_message(node, str(key)))
            if info.kind is Kind.VALUE:
                node.set_value(info.name, value, label)
                continue
            path = format_path((*node.path, info.name))
            if value is None:
                # Also a key with nothing under it, which YAML reads as null.
                if info.optional:
                    node.switch_off(info)
                continue
            if not isinstance(value, dict):
                hint = ""
                if isinstance(value, str) and command_line:
                    hint = (
                        f"\nTo select the config called {value!r} for {path!r}: "
                        f"{path}={format_value(value)}"
                    )
                elif isinstance(value, str):
                    hint = (
                        f"\nTo select the config called {value!r} for {path!r}, "
                        f"write it under '_defaults_'."
                    )
                what = "a group" if info.kind is Kind.GROUP else "a subconfig"
                raise _error(
                    label,
                    f"{path!r} is {what}, so its value has to be a mapping of "
                    f"its fields, not {value!r}.{hint}",
                )
            if info.optional:
                child = node.switch_on(info)
            else:
                try:
                    child = require_child(node, info)
                except ConfigError as exc:
                    raise _error(label, str(exc)) from None
            child.touched = True
            self._apply_values(child, value, label, command_line=command_line)

    # -- selections --------------------------------------------------------

    def _locate(
        self, start: Node, path: KeyPath, source: str, value: Any = MISSING
    ) -> tuple[Node, FieldInfo]:
        """Follow `path` from `start`; return the last node and the field in it.

        `value` is what is to be set at `path`, if anything, for the error
        when the path goes through an optional subconfig that is off.
        """
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
            if node.is_off(info):
                rest = path[index + 1 :]
                raise _error(source, switched_off_message(node, info, rest, value))
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
                f"{key!r} is {what} and cannot be given a dict value: the dict "
                f"names no option. Select one with {key}=NAME, and set its "
                f"fields one by one, as in {key}.field=value."
                f"{self._available(override.key, info)}",
            )
        if override.value is None:
            raise _error(
                source,
                f"{key!r} is {what}, which cannot be switched off with null: "
                f"only a subconfig of one class can be optional, as in X | None."
                f"{self._available(override.key, info)}",
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
        files = self.named_configs(path, info)
        if files:
            label = "From files" if info.kind is Kind.GROUP else "Presets"
            lines += f"\n{label}: {', '.join(files)}"
        elif info.kind is Kind.CONFIG:
            directory = self.directory(path)
            where = f" (looked in {directory})" if directory is not None else ""
            lines += f"\nThere are no config files for {format_path(path)!r}{where}."
        return lines

    def _select(
        self, parent: Node, info: FieldInfo, name: str, source: str, origin: str
    ) -> None:
        """Select the config called `name` for the field `info` of `parent`.

        `source` says where the selection is, for error messages, and `origin`
        is the file that makes it, or the override. What the selection drops
        is recorded, for the warning at the end.
        """
        # A group's node is replaced; a preset is applied to the node in place.
        old = parent.children.get(info.name) if info.kind is Kind.GROUP else None
        start = next_step()
        self._select_config(parent, info, name, source, ())
        node = parent.children[info.name]
        node.selection = Selection(name, origin, start, next_step())
        if old is not None and self._drops is not None:
            covering = (old.selection,) if old.selection is not None else ()
            dropped = tuple(_explicit(old, covering))
            if dropped:
                self._drops.append(_Drop(node.path, name, origin, dropped))

    def _select_config(
        self,
        parent: Node,
        info: FieldInfo,
        name: str,
        source: str,
        chain: tuple[Path, ...],
    ) -> None:
        """Apply the config called `name` for the field `info` of `parent`."""
        path = (*parent.path, info.name)
        if info.kind is Kind.GROUP and name in info.options:
            # Naming an option starts the node again from the class defaults.
            node = Node(info.options[name], path, option=name)
            node.touched = True
            parent.children[info.name] = node
            return

        option, stem = self._split_name(path, info, name, source)
        directory = self.directory(path if option is None else (*path, option))
        file_path = find_file(directory, stem) if directory is not None else None
        if file_path is None:
            candidates = [*info.options, *self.named_configs(path, info)]
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
            self._select_config(
                parent,
                info,
                base,
                f"{file.label}, in the entry '_self_' of '_defaults_'",
                (*chain, file_path),
            )
        elif info.kind is Kind.GROUP:
            raise _error(
                file.label,
                f"A config for the group {format_path(path)!r} has to say which "
                f"option it extends:\n    _defaults_:\n      _self_: <option>"
                f"{self._available(path, info)}"
                f"{_hydra_defaults_hint(file, _option_fields(info))}",
            )
        # A preset switches an optional subconfig on.
        node = parent.switch_on(info) if info.optional else parent.children[info.name]
        if option is not None and node.option != option:
            raise _error(
                file.label,
                f"The file is in the directory of the option {option!r} of "
                f"{format_path(path)!r}, so it has to extend {option!r}, but its "
                f"chain of '_self_' ends at the option {node.option!r}. Move "
                f"the file, or let its '_self_' name {option!r} or a config "
                f"that extends it.",
            )
        node.touched = True
        self._apply_body(node, file)

    def _split_name(
        self, path: KeyPath, info: FieldInfo, name: str, source: str
    ) -> tuple[str | None, str]:
        """The option whose directory a config `option/name` is in, and the name.

        A name without a slash is in the field's own directory: `None`.
        """
        where = format_path(path)
        if "/" not in name:
            if not _is_file_name(name):
                raise _error(
                    source,
                    f"{where!r} has no config called {name!r}. The name of a "
                    f"config is the name of its file, without '..', a "
                    f"backslash or a drive.{self._available(path, info)}",
                )
            return None, name
        option, _, stem = name.partition("/")
        if info.kind is not Kind.GROUP:
            raise _error(
                source,
                f"{where!r} is a subconfig, and has no config called {name!r}: "
                f"only a group has directories in its directory, one for each "
                f"option, and a name with a slash is a config in one of them."
                f"{self._available(path, info)}",
            )
        if option not in info.options or not _is_file_name(stem):
            raise _error(
                source,
                f"{where!r} has no config called {name!r}. A name with a slash "
                f"is a config in the directory of an option of the group, as "
                f"in OPTION/NAME, one level deep, where NAME is the name of "
                f"the file, without '..', a backslash or a drive."
                f"{self._available(path, info)}",
            )
        if option not in _option_directories(info):
            owners = [
                option_name
                for option_name, cls in info.options.items()
                if option in class_fields(cls)
            ]
            directory = self.directory((*path, option))
            raise _error(
                source,
                f"{where!r} has no config called {name!r}: {option!r} is an "
                f"option of {where!r}, but also a field of the option "
                f"{' and '.join(repr(o) for o in owners)}, so the directory "
                f"{directory} belongs to that field and holds no configs of "
                f"the option. Configs that extend the option {option!r} go "
                f"in the directory of {where!r} itself."
                f"{self._available(path, info)}",
            )
        return option, stem


# -- what a selection drops ----------------------------------------------


@dataclass(frozen=True)
class _Dropped:
    """A value, or a selection, that a lower layer set under a replaced node."""

    path: KeyPath
    value: Any
    """The value as it was set, or the name of what was selected."""
    origin: str
    is_selection: bool

    def describe(self) -> str:
        verb = "selected" if self.is_selection else "set"
        text = f"{format_path(self.path)}={format_value(self.value)}"
        return f"{text!r}, {verb} in {_quote_origin(self.origin)}"


@dataclass(frozen=True)
class _Drop:
    """A selection that replaced a node, and what it dropped there."""

    path: KeyPath
    name: str
    origin: str
    dropped: tuple[_Dropped, ...]

    def message(self, lost: Sequence[_Dropped]) -> str:
        where = format_path(self.path)
        who = _quote_origin(self.origin)
        who = who[0].upper() + who[1:]
        lines = "".join(f"\n    {item.describe()}" for item in lost)
        if len(lost) == 1:
            drops, keep = "a value that was", "the value, set it"
        else:
            drops, keep = "values that were", "the values, set them"
        return (
            f"{who} resets the config values in {where!r} to the values given "
            f"by {self.name!r}. This drops {drops} set for {where!r} before:"
            f"{lines}\nTo keep {keep} again after the selection; in the file "
            f"that makes the selection, or on the command line."
        )


def _quote_origin(origin: str) -> str:
    """An override as "the override '...'", and a file's path in backticks."""
    return origin if origin.startswith("the override ") else f"`{origin}`"


def _covered(step: int, selections: Sequence[Selection]) -> bool:
    return any(selection.covers(step) for selection in selections)


def _explicit(node: Node, selections: tuple[Selection, ...]) -> Iterator[_Dropped]:
    """What was set under `node` other than by the selections that made it.

    `selections` are those of the replaced node and of the nodes on the way
    down to `node`: what they brought goes along with the config.
    """
    for name, step in node.steps.items():
        if not _covered(step, selections):
            yield _Dropped(
                (*node.path, name),
                node.values[name],
                node.sources[name],
                is_selection=False,
            )
    for child in node.children.values():
        inner = selections
        if child.selection is not None:
            if not _covered(child.selection.start, selections):
                selection = child.selection
                yield _Dropped(
                    child.path, selection.name, selection.origin, is_selection=True
                )
            inner = (*selections, child.selection)
        yield from _explicit(child, inner)


def _plain_value(override: Override) -> Any:
    """The value of `override`, or ``MISSING`` for `~key` or a sweep."""
    if override.unset or override.is_sweep:
        return MISSING
    return override.value


def _check_unset(override: Override, parent: Node, info: FieldInfo) -> None:
    """`~key` leaves out only a `NotRequired` key of a `TypedDict`."""
    source = repr(override.text)
    key = override.key_text
    if not is_typeddict_class(parent.cls):
        hint = ""
        if _takes_none(info):
            hint = f" To set it to None: {key}=null"
        raise _error(
            source,
            f"{key!r} is a field of the dataclass {parent.describe()}, so it "
            "cannot be left out. '~' leaves out a NotRequired key of a "
            f"TypedDict.{hint}",
        )
    if info.required:
        raise _error(
            source,
            f"{key!r} is a required key of {parent.describe()}, so it cannot "
            "be left out. '~' leaves out a NotRequired key of a TypedDict.",
        )


def _takes_none(info: FieldInfo) -> bool:
    """Whether the field accepts `None`, as `key=null` would give it."""
    adapter = field_adapter(info)
    if adapter is None:
        return False
    try:
        adapter.validate_json("null", strict=True)
    except ValidationError:
        return False
    return True


def _is_lost(root: Node, replaced: KeyPath, item: _Dropped) -> bool:
    """Whether the composed tree lacks what was dropped, and could have it.

    Not lost: a field that the new config does not have, one that a layer
    above the selection set again, and one that ends up with the same value.
    """
    found = _parent_in_tree(root, replaced, item.path)
    if found is None:
        return False
    node, selections = found
    name = item.path[-1]
    info = node.fields.get(name)
    if info is None:
        return False
    if item.is_selection:
        child = node.children.get(name)
        if child is None or child.selection is None:
            return True
        set_again = not _covered(child.selection.start, selections)
        return not set_again and child.selection.name != item.value
    if info.kind is not Kind.VALUE:
        return True
    step = node.steps.get(name)
    set_again = step is not None and not _covered(step, selections)
    return not set_again and bool(node.value(info) != item.value)


def _parent_in_tree(
    root: Node, replaced: KeyPath, path: KeyPath
) -> tuple[Node, list[Selection]] | None:
    """The node that holds `path` now, and the selections that brought it.

    The selections are those of the replaced node and of the nodes below it
    on the way to `path`. `None` if the path is not in the tree any more.
    """
    node = root
    selections: list[Selection] = []
    for index, segment in enumerate(path[:-1]):
        child = node.children.get(segment)
        if child is None:
            return None
        node = child
        if index >= len(replaced) - 1 and node.selection is not None:
            selections.append(node.selection)
    return node, selections
