"""Finding config files and combining them, with the overrides, into one tree."""

from collections.abc import Collection, Iterator, Mapping, Sequence
import dataclasses
from dataclasses import MISSING, dataclass
import difflib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, cast
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
    REPLACE_KEY,
    RESERVED_KEY,
    SELF_KEY,
    SOME,
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
    nest,
    next_step,
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
        self._files: dict[Path, ConfigFile] = {}
        """The config files read so far, by path. Each is read once, so that
        every tree composed here sees the file as it was then; nothing
        changes what is in them."""
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
        """Apply command-line overrides: selections and dicts first, then values.

        Apart from a key that is given more than once, where the last one
        wins, the order of the overrides has no meaning. Selections, `null`
        for an optional subconfig or group, `_some_` for an optional
        subconfig, and the dicts for a TypedDict are applied shortest path
        first, so that `model=cnn` is in place before
        `model.norm=batchnorm` is looked at. The values come after.

        A dict replaces what was set before at its key, for a TypedDict as for
        a value; it is not merged as a mapping for a TypedDict in a file is.
        """
        overrides = last_wins(overrides)
        # A value's node stays in the tree: what comes after it in this order
        # replaces only nodes at least as deep as the value itself.
        values: list[tuple[Override, Node, FieldInfo]] = []
        for override in sorted(overrides, key=lambda o: len(o.key)):
            source = repr(override.text)
            _check_no_replace(override)
            parent, info = self._locate(
                root, override.key, source, _plain_value(override)
            )
            if info.kind is Kind.VALUE:
                values.append((override, parent, info))
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
            if override.value is None:
                if not info.optional:
                    raise _error(source, self._null_message(parent, info))
                parent.switch_off(info)
                continue
            if override.value == SOME:
                self._switch_on(parent, info, source)
                continue
            if isinstance(override.value, dict):
                self._apply_mapping(parent, info, override)
                continue
            name = self._selection_name(override, info)
            self._select(parent, info, name, source, f"the override {source}")
        for override, parent, info in values:
            source = repr(override.text)
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
            label = f"the override {source}"
            parent.set_value(info.name, override.value, label)

    def _apply_mapping(self, parent: Node, info: FieldInfo, override: Override) -> None:
        """Replace a TypedDict with a dict from the command line.

        The TypedDict is a subconfig, which is switched on if it is optional,
        or the selected option of a group, which stays selected. What was set
        or selected under it goes, its default included, and the dict is
        applied as a mapping in a file would be. A dataclass is not given as
        a dict.
        """
        source = repr(override.text)
        label = f"the override {source}"
        if not _holds_typeddicts(info):
            raise _error(
                source, self._dataclass_dict_message(parent, info, override.value)
            )
        child = self._mapping_child(parent, info, source)
        step = next_step()
        child.replace(step)
        # What the dict replaces is not warned about: it says so.
        self._forget_drops(child.path)
        self._apply_values(child, override.value, label, since=step, command_line=True)

    def _dataclass_dict_message(
        self, parent: Node, info: FieldInfo, value: dict[Any, Any]
    ) -> str:
        """The error for a dict on the command line for a dataclass."""
        path = (*parent.path, info.name)
        key = format_path(path)
        if info.kind is Kind.GROUP:
            what = f"a group of {parent.describe()}, whose options are dataclasses"
        else:
            assert info.config_cls is not None
            what = (
                f"a subconfig of {parent.describe()}, and "
                f"{info.config_cls.__name__} is a dataclass"
            )
        child = parent.children.get(info.name)
        if child is None and info.config_cls is not None:
            # Off: the fields are those of the class, which `_some_` switches on.
            child = Node(info.config_cls, path)
        dotted = " ".join(
            f"{format_path(keys)}={format_value(inner)}"
            for keys, inner in _dotted(child, value, path)
        )
        lines = [
            f"{key!r} is {what}. On the command line, a dict replaces what was "
            f"set before, which is for a TypedDict or a dict value; a "
            f"dataclass is not given as a dict."
        ]
        if parent.is_off(info) and info.kind is Kind.GROUP:
            lines.append(
                f"{key!r} is switched off (null). Selecting an option switches "
                f"it on: {key}=NAME; its fields can then be set one by one."
            )
        elif parent.is_off(info):
            example = f"{key}={SOME} {dotted}".rstrip()
            lines.append(
                f"{key!r} is switched off (null). {key}={SOME} switches it on "
                f"with the defaults of its class, and a named config for it, "
                f"{key}=NAME, switches it on too; its fields can then be set "
                f"one by one: {example}"
            )
        else:
            if dotted:
                lines.append(f"Set its fields one by one: {dotted}")
            lines.append(f"A named config for it is selected with {key}=NAME.")
        return "\n".join(lines) + self._available(path, info)

    def _defaults_null_message(self, node: Node, info: FieldInfo) -> str:
        """The error for null under `_defaults_`, for a field that is not optional."""
        return (
            f"{self._null_message(node, info)}\nAn entry with null, as in "
            f"Hydra's '- {info.name}: null', is not needed in nematocyte; "
            f"remove it."
        )

    def _null_message(self, node: Node, info: FieldInfo) -> str:
        """The error for null on a subconfig or a group that is not optional."""
        path = (*node.path, info.name)
        key = format_path(path)
        if info.kind is Kind.GROUP:
            return (
                f"{key!r} is a group of {node.describe()}, and cannot be null: "
                f"its type does not allow None. An optional group has None "
                f"outside the union of the group, as in Annotated[..., "
                f"Discriminator(...)] | None.{self._available(path, info)}"
            )
        assert info.config_cls is not None
        name = info.config_cls.__name__
        return (
            f"{key!r} is a subconfig of {node.describe()}, and cannot be null: "
            f"its type is {name}, not {name} | None."
        )

    def _replace(
        self, node: Node, values: dict[Any, Any], label: str, since: int
    ) -> dict[Any, Any]:
        """Apply `_replace_: true` in the mapping `values` for `node`, if it has it.

        The rest of the mapping is returned, without the key.
        """
        if REPLACE_KEY not in values:
            return values
        _check_replace(node, values[REPLACE_KEY], label)
        node.replace(since)
        self._forget_drops(node.path)
        return {key: value for key, value in values.items() if key != REPLACE_KEY}

    def _forget_drops(self, path: KeyPath) -> None:
        """Forget the values under `path` that a selection dropped.

        `_replace_` drops them anyway, so they are not warned about.
        """
        if self._drops is None:
            return
        kept: list[_Drop] = []
        for drop in self._drops:
            items = tuple(
                item for item in drop.dropped if item.path[: len(path)] != path
            )
            if items:
                kept.append(dataclasses.replace(drop, dropped=items))
        self._drops = kept

    # -- where the files are ----------------------------------------------

    def _load(self, path: Path) -> ConfigFile:
        file = self._files.get(path)
        if file is None:
            file = self._files[path] = load_file(path)
        return file

    def _file_doc(self, path: Path) -> str:
        """The first paragraph of the `_doc_` of the config file at `path`."""
        return first_paragraph(self._load(path).doc)

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
            name: self._file_doc(file)
            for name, file in self._named_files(path, info).items()
        }

    def _named_files(self, path: KeyPath, info: FieldInfo) -> dict[str, Path]:
        """The config files for the field `info` at `path`, by name, sorted."""
        directory = self.directory(path)
        if directory is None:
            return {}
        places = [
            (directory, ""),
            *((directory / option, f"{option}/") for option in info.options),
        ]
        # A file that no name can select is left out; the validator reports it.
        # `_some_` is no config's name, but `option/_some_` is.
        return dict(
            sorted(
                (prefix + name, file)
                for place, prefix in places
                for name, file in list_files(place).items()
                if _is_file_name(name) and (prefix or name != SOME)
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
            if info is not None and len(middle) == 1 and middle[0] in info.options:
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
        return list(self._primary_files())

    def primary_config_docs(self) -> dict[str, str]:
        """The primary configs, as `--config-name` takes them, with their docs."""
        return {
            name: self._file_doc(path) for name, path in self._primary_files().items()
        }

    def _primary_files(self) -> dict[str, Path]:
        """The files of all primary configs, by the names that select them."""
        if self.config_dir is None:
            return {}
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
        return {
            path.relative_to(self.config_dir).with_suffix("").as_posix(): path
            for path in paths
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
        file = self._load(path)
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
        # A `_replace_` in the file drops what was set before this point.
        since = next_step()
        entries = [
            (tuple(key.split(".")), key, name)
            for key, name in file.defaults.items()
            if key != SELF_KEY
        ]
        # What the file switches off here, it cannot set among its values.
        switched_off: dict[KeyPath, str] = {}
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
            if name is None:
                if not info.optional:
                    raise _error(source, self._defaults_null_message(parent, info))
                parent.switch_off(info)
                switched_off[path] = key
                continue
            if name == SOME:
                self._switch_on(parent, info, source)
                continue
            self._select(parent, info, name, source, file.label)
        self._apply_values(
            node, file.values, file.label, since=since, switched_off=switched_off
        )

    def _apply_values(
        self,
        node: Node,
        values: dict[Any, Any],
        label: str,
        *,
        since: int,
        command_line: bool = False,
        switched_off: dict[KeyPath, str] | None = None,
        prefix: KeyPath = (),
    ) -> None:
        """Apply the values of a file, or of a dict on the command line.

        A mapping goes into a subconfig or a group. In a file, an optional
        subconfig or group that is off takes no mapping and no null: they are
        switched on and off under `_defaults_`. `switched_off` maps what the
        file switched off there, by its path below the node of the file, to
        the entry; the file cannot give it a value as well. On the command
        line, a dict switches an optional subconfig on, null switches an
        optional subconfig or group off, and null is an error for any other.
        A dict on the command line holds no mapping for a dataclass. `prefix`
        is the path of `node` below the node of the file.

        A mapping for a subconfig or a group is merged into what was set
        before: its keys are declared. A value replaces what was set before,
        mappings in it included, since their keys are not declared. `since` is
        when the layer started: a `_replace_` for a TypedDict drops what was
        set before it.
        """
        values = self._replace(node, values, label, since)
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
                if not command_line:
                    _check_null_for_a_dict(node, info, value, label)
                _check_value_replaces(node, info, value, label)
                node.set_value(info.name, _without_replace(value), label)
                continue
            path = format_path((*node.path, info.name))
            if switched_off is not None:
                _check_not_switched_off(
                    switched_off, (*prefix, info.name), path, value, label
                )
            if value is None:
                # On the command line, null switches an optional subconfig or
                # group off. In a file, it is an error: they are switched off
                # under `_defaults_`, and a key with nothing under it, which
                # YAML reads as null, may have been meant to leave them alone.
                if not command_line:
                    message = self._file_null_message(node, info, prefix)
                    raise _error(label, message)
                if not info.optional:
                    raise _error(label, self._null_message(node, info))
                node.switch_off(info)
                continue
            if not isinstance(value, dict):
                raise _error(
                    label,
                    _not_a_mapping_message(
                        path, info, value, command_line=command_line
                    ),
                )
            if command_line and not _holds_typeddicts(info):
                raise _error(label, self._dataclass_dict_message(node, info, value))
            if node.is_off(info) and info.kind is Kind.CONFIG and not command_line:
                raise _error(
                    label, self._file_mapping_off_message(node, info, value, prefix)
                )
            self._apply_values(
                self._mapping_child(node, info, label),
                value,
                label,
                since=since,
                command_line=command_line,
                switched_off=switched_off,
                prefix=(*prefix, info.name),
            )

    def _file_null_message(self, node: Node, info: FieldInfo, prefix: KeyPath) -> str:
        """The error for null among the values of a file, for a subconfig or group."""
        path = (*node.path, info.name)
        key = format_path(path)
        what = "group" if info.kind is Kind.GROUP else "subconfig"
        if info.optional:
            lines = [
                f"{key!r} is an optional {what} of {node.describe()}, and cannot "
                f"be null among the values of a config file, nor a key with "
                f"nothing under it, which YAML reads as null. (In Hydra, this "
                f"switches it off.)",
                f"To switch it off, write the entry under '_defaults_':\n"
                f"    _defaults_:\n      {format_path((*prefix, info.name))}: null",
            ]
        else:
            lines = [
                f"{self._null_message(node, info)} A key with nothing under it "
                f"is null too, to YAML."
            ]
        lines.append("To leave it as it is, leave out the key.")
        if _holds_typeddicts(info):
            lines.append(
                f"To drop what was set in it before: {info.name}: "
                f"{{{REPLACE_KEY}: true}}"
            )
        return "\n".join(lines)

    def _file_mapping_off_message(
        self, node: Node, info: FieldInfo, value: dict[Any, Any], prefix: KeyPath
    ) -> str:
        """The error for a mapping in a file, for an optional subconfig that is off."""
        assert info.config_cls is not None
        name = info.config_cls.__name__
        key = format_path((*node.path, info.name))
        entry = format_path((*prefix, info.name))
        return (
            f"{key!r} is an optional subconfig of {node.describe()} "
            f"({name} | None), and it is switched off (null), so a mapping "
            f"cannot set its fields ({format_value(value)}). (In Hydra, the "
            f"mapping switches it on.)\n"
            f"To switch it on, with the defaults of {name}, write the entry "
            f"under '_defaults_' of the same file, and the mapping applies "
            f"on top:\n    _defaults_:\n      {entry}: {SOME}"
        )

    def _switch_on(self, parent: Node, info: FieldInfo, source: str) -> None:
        """Switch on the optional subconfig `info`, for `_some_`.

        If it is off, it starts from the defaults of its class; if it is on,
        it stays as it is.
        """
        key = format_path((*parent.path, info.name))
        if info.kind is Kind.GROUP:
            first = next(iter(info.options))
            raise _error(
                source,
                f"{key!r} is a group of {parent.describe()}, and {SOME} "
                f"switches on an optional subconfig, which has one class. An "
                f"option of a group is selected by its name, which also "
                f"switches on an optional group: {key}={first} on the command "
                f"line, or {key}: {first} under '_defaults_' in a config "
                f"file.{self._available((*parent.path, info.name), info)}",
            )
        if not info.optional:
            assert info.config_cls is not None
            name = info.config_cls.__name__
            raise _error(
                source,
                f"{key!r} is a subconfig of {parent.describe()}, and its type "
                f"is {name}, not {name} | None, so it is always on. {SOME} "
                f"switches on an optional subconfig; remove it.",
            )
        parent.switch_on(info)

    # -- selections --------------------------------------------------------

    def _locate(
        self, start: Node, path: KeyPath, source: str, value: Any = MISSING
    ) -> tuple[Node, FieldInfo]:
        """Follow `path` from `start`; return the last node and the field in it.

        `value` is what is to be set at `path`, if anything, for the error
        when the path goes through an optional subconfig or group that is off.
        """
        node = start
        for index, segment in enumerate(path):
            info = node.fields.get(segment)
            if info is None and segment == REPLACE_KEY:
                parent = format_path(node.path)
                example = f": {parent}={{...}}" if parent else ""
                raise _error(
                    source,
                    f"{REPLACE_KEY!r} is not a field of {node.describe()}. In a "
                    f"config file, {REPLACE_KEY}: true goes into the mapping for a "
                    f"TypedDict, which then replaces what was set before instead "
                    f"of merging. On "
                    f"the command line, a dict replaces anyway{example}",
                )
            if info is None:
                raise _error(source, unknown_key_message(node, segment))
            if index == len(path) - 1:
                return node, info
            if info.kind is Kind.VALUE:
                here = format_path((*node.path, segment))
                rest = path[index + 1 :]
                example = "{...}"
                if value is not MISSING:
                    example = format_value(nest(rest, value))
                raise _error(
                    source,
                    f"{here!r} is a value, not a subconfig, so it has no key "
                    f"{format_path(rest)!r}. On the command line, a dict value is "
                    f"given as a whole, and replaces the dict that it holds: "
                    f"{here}={example}",
                )
            if node.is_off(info):
                rest = path[index + 1 :]
                raise _error(source, switched_off_message(node, info, rest, value))
            node = self._child(node, info, source)
        raise AssertionError("an empty path")

    def _child(self, parent: Node, info: FieldInfo, source: str) -> Node:
        """The node of a subconfig or group field.

        An error if no option is selected, or if an optional subconfig or
        group is off.
        """
        child = parent.children.get(info.name)
        if child is not None:
            return child
        if parent.is_off(info):
            raise _error(source, switched_off_message(parent, info))
        path = format_path((*parent.path, info.name))
        hint = f" To switch it off: {path}=null." if info.optional else ""
        raise _error(
            source,
            f"No option is selected for the group {path!r}. "
            f"Options: {', '.join(info.options)}.{hint}",
        )

    def _mapping_child(self, parent: Node, info: FieldInfo, source: str) -> Node:
        """The node that a mapping, or a preset, for the field `info` goes into.

        It switches an optional subconfig on; an optional group is switched
        on only by selecting an option.
        """
        if info.optional and info.kind is Kind.CONFIG:
            return parent.switch_on(info)
        child = self._child(parent, info, source)
        child.touched = True
        return child

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
        # The node is replaced, for an option of a group and for a preset alike.
        old = parent.children.get(info.name)
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
        if name == SOME:
            # A direct entry or override is handled before; this is `_self_`.
            raise _error(
                source,
                f"{SOME} is not the name of a config, so '_self_' cannot name "
                f"it. It switches an optional subconfig on, as an entry under "
                f"'_defaults_' of the file that holds the subconfig, as in "
                f"{info.name}: {SOME}, or on the command line, as in "
                f"{format_path(path)}={SOME}. A config for "
                f"{format_path(path)!r} switches it on anyway.",
            )
        if info.kind is Kind.GROUP and name in info.options:
            # Naming an option starts the node again from the class defaults.
            node = Node(info.options[name], path, option=name)
            node.touched = True
            parent.children[info.name] = node
            parent.off.pop(info.name, None)
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
        file = self._load(file_path)
        # A file in the directory of an option extends the option, unless its
        # `_self_` names another config.
        base = file.defaults.get(SELF_KEY, option)
        if base is not None:
            self._select_config(
                parent,
                info,
                base,
                f"{file.label}, in the entry '_self_' of '_defaults_'",
                (*chain, file_path),
            )
        elif info.kind is Kind.CONFIG:
            # A chain of presets ends at the class defaults, as a chain of
            # named configs for a group ends at an option.
            assert info.config_cls is not None
            parent.children[info.name] = Node(info.config_cls, path)
            parent.off.pop(info.name, None)
        else:
            raise _error(
                file.label,
                f"A config for the group {format_path(path)!r} has to say which "
                f"option it extends:\n    _defaults_:\n      _self_: <option>\n"
                f"A file in the directory of an option, such as "
                f"{'/'.join(path)}/<option>/, extends that option without it."
                f"{self._available(path, info)}"
                f"{_hydra_defaults_hint(file, _option_fields(info))}",
            )
        node = self._mapping_child(parent, info, source)
        if option is not None and node.option != option:
            raise _error(
                file.label,
                f"The file is in the directory of the option {option!r} of "
                f"{format_path(path)!r}, so it has to extend {option!r}, but its "
                f"chain of '_self_' ends at the option {node.option!r}. Move "
                f"the file, or let its '_self_' name {option!r} or a config "
                f"that extends it.",
            )
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


def _not_a_mapping_message(
    path: str, info: FieldInfo, value: Any, *, command_line: bool
) -> str:
    """The error for a value of a subconfig or a group that is not a mapping."""
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
    return (
        f"{path!r} is {what}, so its value has to be a mapping of its fields, "
        f"not {value!r}.{hint}"
    )


def _check_null_for_a_dict(node: Node, info: FieldInfo, value: Any, label: str) -> None:
    """In a file, null for a value that is a dict, and cannot be `None`, is an error.

    A key with nothing under it is null to YAML, and may have been meant as
    an empty dict, which is written `{}`.
    """
    if value is not None or _takes_none(info) or not _takes_a_mapping(info):
        return
    key = format_path((*node.path, info.name))
    raise _error(
        label,
        f"{key!r} of {node.describe()} holds a dict, and its type does not "
        f"allow None, so it cannot be null, nor a key with nothing under it, "
        f"which YAML reads as null. An empty dict is written "
        f"{info.name}: {{}}, which replaces what was set in it before; to "
        f"leave it as it is, leave out the key.",
    )


def _check_not_switched_off(
    switched_off: dict[KeyPath, str],
    here: KeyPath,
    path: str,
    value: Any,
    label: str,
) -> None:
    """A file cannot give a value to what it switches off under `_defaults_`."""
    entry = switched_off.get(here)
    if entry is not None:
        shown = "nothing under it" if value is None else format_value(value)
        raise _error(
            label,
            f"{path!r} is switched off in the entry {entry!r} of '_defaults_', "
            f"and the same file gives it a value among the values ({shown}). "
            f"Remove one of the two.",
        )


def _holds_typeddicts(info: FieldInfo) -> bool:
    """Whether the subconfig or group `info` is a TypedDict, or a group of them."""
    if info.kind is Kind.CONFIG:
        assert info.config_cls is not None
        return is_typeddict_class(info.config_cls)
    return all(is_typeddict_class(cls) for cls in info.options.values())


def _dotted(
    node: Node | None, value: dict[Any, Any], path: KeyPath
) -> Iterator[tuple[KeyPath, Any]]:
    """The dotted keys that a dict for `node` at `path` stands for.

    They go down to the first field that holds a value, or that is not known.
    """
    for key, inner in value.items():
        here = (*path, str(key))
        info = (
            node.fields.get(key) if node is not None and isinstance(key, str) else None
        )
        if info is None or info.kind is Kind.VALUE or not isinstance(inner, dict):
            yield here, inner
            continue
        child = node.children.get(info.name) if node is not None else None
        yield from _dotted(child, cast("dict[Any, Any]", inner), here)


def _check_no_replace(override: Override) -> None:
    """A dict on the command line replaces anyway, so it has no `_replace_`."""
    for path, _ in _replace_paths(override.value):
        where = format_path((*override.key, *path))
        raise _error(
            repr(override.text),
            f"{REPLACE_KEY!r} is in the dict for {where!r}. On the command "
            f"line, a dict replaces what was set before anyway, so leave out "
            f"the key. In a config file, {REPLACE_KEY}: true makes the mapping "
            f"for a TypedDict replace instead of merging.",
        )


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


def _check_replace(node: Node, value: Any, label: str) -> None:
    """`_replace_: true` replaces only a `TypedDict` below the root."""
    if not node.path:
        raise _error(
            label,
            f"{REPLACE_KEY!r} is at the top level of the config, where it would "
            f"replace the whole of {node.describe()}. It goes into the mapping "
            "for a TypedDict subconfig or group.",
        )
    where = f"{REPLACE_KEY!r} in the mapping for {format_path(node.path)!r}"
    if not is_typeddict_class(node.cls):
        raise _error(
            label,
            f"{where}: {node.describe()} is a dataclass, which cannot be "
            f"replaced. {REPLACE_KEY!r} replaces a TypedDict; a field of a "
            "dataclass is set back to its default by setting it again.",
        )
    if value is not True:
        raise _error(
            label,
            f"{where} is {format_value(value)}. It takes true, which replaces "
            f"what was set before under {node.describe()}; to merge the "
            "mapping into it, leave out the key.",
        )


def _replace_paths(value: Any, path: KeyPath = ()) -> Iterator[tuple[KeyPath, Any]]:
    """The places of the mappings in `value` that have `_replace_`, with its value."""
    if not isinstance(value, Mapping):
        return
    mapping = cast("Mapping[Any, Any]", value)
    if REPLACE_KEY in mapping:
        yield path, mapping[REPLACE_KEY]
    for key, inner in mapping.items():
        if key != REPLACE_KEY:
            yield from _replace_paths(inner, (*path, str(key)))


def _without_replace(value: Any) -> Any:
    """`value` without the key `_replace_` in its mappings."""
    if not isinstance(value, Mapping):
        return value
    mapping = cast("Mapping[Any, Any]", value)
    return {
        key: _without_replace(inner)
        for key, inner in mapping.items()
        if key != REPLACE_KEY
    }


def _check_value_replaces(node: Node, info: FieldInfo, value: Any, label: str) -> None:
    """`_replace_` in a mapping inside a value takes true, and changes nothing.

    A value replaces what was set before anyway, since the keys of a mapping
    in it are not declared.
    """
    for path, flag in _replace_paths(value):
        if flag is not True:
            where = format_path((*node.path, info.name, *path))
            raise _error(
                label,
                f"{REPLACE_KEY!r} in the mapping for {where!r} is "
                f"{format_value(flag)}. It takes only true, and is not needed "
                f"here: a dict replaces what was set before anyway, so leave "
                "out the key.",
            )


def _takes_a_mapping(info: FieldInfo) -> bool:
    """Whether the field accepts an empty dict, and so holds a mapping."""
    adapter = field_adapter(info)
    if adapter is None:
        return False
    try:
        adapter.validate_python({})
    except ValidationError:
        return False
    return True


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
