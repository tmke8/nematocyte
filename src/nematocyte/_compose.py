"""Finding config files and combining them, with the overrides, into one tree."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import difflib
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
import warnings

from ._errors import ConfigError, DroppedValueWarning
from ._files import (
    SUFFIXES,
    ConfigFile,
    check_unique,
    display_path,
    find_file,
    list_names,
    load_file,
)
from ._overrides import AddedConfig, Override, format_value, last_wins
from ._schema import (
    RESERVED_KEY,
    SELF_KEY,
    FieldInfo,
    Kind,
    class_fields,
    validate_schema,
)
from ._tree import (
    Node,
    Selection,
    format_path,
    make_root,
    next_step,
    require_child,
    unknown_key_message,
)
from ._tree import Path as KeyPath

__all__ = ["Composer"]

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# A warning points at the first frame outside nematocyte: the caller's code.
_PACKAGE_DIR = str(Path(__file__).parent) + os.sep


def _error(source: str, message: str) -> ConfigError:
    return ConfigError(f"Error in {source}:\n{message}")


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
            parent, info = self._locate(root, override.key, source)
            if info.kind is Kind.VALUE:
                values.append(override)
                continue
            name = self._selection_name(override, info)
            self._select(parent, info, name, source, f"the override {source}")
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
            paths.append(path)
        check_unique(paths)
        return [
            path.relative_to(self.config_dir).with_suffix("").as_posix()
            for path in paths
        ]

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
            self._select(parent, info, name, source, file.label)
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
            self._select_config(
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
