"""The composed config, as a tree of nodes that mirrors the config classes."""

from collections.abc import Mapping
from dataclasses import MISSING, dataclass
import difflib
import itertools
from typing import Any, cast

from ._errors import ConfigError, SchemaError
from ._json_form import to_json_form
from ._merge import Part, merge_parts
from ._overrides import format_value
from ._schema import (
    OPTION_KEY,
    RESERVED_KEY,
    FieldInfo,
    Kind,
    class_fields,
    is_dict_or_frozendict,
    is_typeddict_class,
)

__all__ = [
    "Node",
    "Path",
    "Selection",
    "format_path",
    "make_root",
    "next_step",
    "node_from_json_tree",
    "require_child",
    "root_fields",
    "switched_off_message",
    "unknown_key_message",
]

type Path = tuple[str, ...]


def format_path(path: Path) -> str:
    return ".".join(path)


_steps = itertools.count()


def next_step() -> int:
    """A number that is larger than every one before it, to order what was set."""
    return next(_steps)


@dataclass(frozen=True)
class Selection:
    """The selection that made a node, or the last preset applied to it."""

    name: str
    """The option or the named config that was selected."""
    origin: str
    """The file that made the selection, or the override."""
    start: int
    end: int
    """The steps between which the selected config was applied."""

    def covers(self, step: int) -> bool:
        """Whether what was set at `step` came with the selected config."""
        return self.start <= step <= self.end


class Node:
    """One config class (a dataclass or a `TypedDict`) in the composed config.

    A node knows its class, the values that config files and overrides set
    on it, and where every other value comes from: the default instance of
    the field, if there is one, and the defaults of the class otherwise.
    """

    def __init__(
        self,
        cls: type,
        path: Path,
        *,
        default: Any = MISSING,
        option: str | None = None,
        fields: dict[str, FieldInfo] | None = None,
        expand: bool = True,
    ) -> None:
        self.cls = cls
        self.path = path
        self.default = default
        """The default instance that missing values are taken from, if any."""
        self.option = option
        """For a node of a group: the name of the option that its class is."""
        self.fields = class_fields(cls) if fields is None else fields
        self.children: dict[str, Node] = {}
        self.off: dict[str, int] = {}
        """The optional subconfigs and groups that are switched off (null).

        Each with the step when it was switched off; -1 for a default. A field
        is off only while it has no node in `children`.
        """
        self.values: dict[str, Any] = {}
        """The values that were set, merged from their `parts`.

        ``MISSING`` for a key of a `TypedDict` that `~key` left out.
        """
        self.parts: dict[str, list[Part]] = {}
        """What the files and overrides gave for each value, in order."""
        self.sources: dict[str, str] = {}
        """Where each of the values was set, for error messages."""
        self.steps: dict[str, int] = {}
        """When each of the values was set, from `next_step`."""
        self.selection: Selection | None = None
        """The selection that made the node; `None` for a default."""
        self.touched = False
        """Whether a config file or an override selected or set something here."""
        if expand:
            for info in self.fields.values():
                if info.kind is not Kind.VALUE and info.name not in self.children:
                    self.expand_field(info)

    def expand_field(self, info: FieldInfo) -> None:
        default = self.default_value(info)
        where = f"{self.class_name}.{info.name}"
        path = (*self.path, info.name)
        if default is None and info.optional:
            # An optional subconfig or group that is off by default has no node.
            self.off[info.name] = -1
            return
        if info.kind is Kind.CONFIG:
            assert info.config_cls is not None
            if is_typeddict_class(info.config_cls):
                fits = is_dict_or_frozendict(default)
            else:
                fits = type(default) is info.config_cls
            if default is not MISSING and not fits:
                raise SchemaError(
                    f"{where}: the default is an instance of "
                    f"{type(default).__name__}, but the field is declared as "
                    f"{info.config_cls.__name__}."
                )
            self.children[info.name] = Node(info.config_cls, path, default=default)
        elif default is not MISSING:
            name = self._default_option(info, default, where)
            cls = (
                info.options[name] if is_dict_or_frozendict(default) else type(default)
            )
            self.children[info.name] = Node(cls, path, default=default, option=name)

    @staticmethod
    def _default_option(info: FieldInfo, default: Any, where: str) -> str:
        """The option that the default of a group field is."""
        typeddicts = any(is_typeddict_class(cls) for cls in info.options.values())
        if typeddicts and is_dict_or_frozendict(default):
            # A TypedDict's dict names its option in its key `_option_`.
            name = default.get(OPTION_KEY)
            if not isinstance(name, str) or name not in info.options:
                raise SchemaError(
                    f"{where}: the default has to name one of the options in its "
                    f"key {OPTION_KEY!r}, not {name!r}. Options: "
                    f"{', '.join(info.options)}."
                )
            return name
        name = info.option_of(type(default))
        if name is None:
            raise SchemaError(
                f"{where}: the default is an instance of "
                f"{type(default).__name__}, which is not an option of the "
                f"group. Options: {', '.join(info.options)}."
            )
        return name

    @property
    def class_name(self) -> str:
        return self.cls.__name__

    def describe(self) -> str:
        """The class of the node and where it is, for error messages."""
        if not self.path:
            return self.class_name
        where = format_path(self.path)
        if self.option is not None:
            return f"{self.class_name} (the option {self.option!r} of {where!r})"
        return f"{self.class_name} (at {where!r})"

    def switch_off(self, info: FieldInfo) -> None:
        """Switch off the optional subconfig or group `info`.

        What was set or selected under it is gone.
        """
        assert info.optional
        self.children.pop(info.name, None)
        self.off[info.name] = next_step()
        self.touched = True

    def switch_on(self, info: FieldInfo) -> Node:
        """The node of the optional subconfig `info`, switched on if it was off.

        A subconfig that is switched on starts from the defaults of its class.
        """
        assert info.kind is Kind.CONFIG
        child = self.children.get(info.name)
        if child is None:
            assert info.optional
            assert info.config_cls is not None
            child = Node(info.config_cls, (*self.path, info.name))
            self.children[info.name] = child
        self.off.pop(info.name, None)
        child.touched = True
        return child

    def is_off(self, info: FieldInfo) -> bool:
        """Whether `info` is an optional subconfig or group that is switched off."""
        return (
            info.optional and info.name in self.off and info.name not in self.children
        )

    def default_value(self, info: FieldInfo) -> Any:
        """The value of a field that nothing has set, or ``MISSING``."""
        if is_dict_or_frozendict(self.default):
            # The default of a `TypedDict` node; a key it leaves out is not set.
            return self.default.get(info.name, MISSING)
        if self.default is not MISSING and not info.initvar:
            return getattr(self.default, info.name)
        return info.class_default()

    def value(self, info: FieldInfo) -> Any:
        """The value of a field, or ``MISSING``.

        A value that was set is as its source gave it; a default is in JSON
        form.
        """
        if info.name in self.values:
            return self.values[info.name]
        default = self.default_value(info)
        return default if default is MISSING else to_json_form(default)

    def set_value(
        self,
        name: str,
        value: Any,
        source: str,
        *,
        since: int | None = None,
        whole: bool = False,
    ) -> None:
        """Set a value on top of what was set before.

        A mapping is merged into a mapping. `since` is when the layer that
        sets the value started, which a `_replace_` in the value goes back
        to; without it, the layer starts with the value. With `whole`, the
        value replaces what was there.
        """
        step = next_step()
        part = Part(value, step, step if since is None else since, whole)
        self.parts.setdefault(name, []).append(part)
        self._merge(name)
        self.sources[name] = source
        self.steps[name] = step
        self.touched = True

    def _merge(self, name: str) -> None:
        info = self.fields[name]
        default = self.default_value(info)
        base = default if default is MISSING else to_json_form(default)
        self.values[name] = merge_parts(base, self.parts[name])

    def unset_value(self, name: str, source: str, *, since: int | None = None) -> None:
        """Leave the key out, whatever set it before, the default included."""
        self.set_value(name, MISSING, source, since=since, whole=True)

    def changed_since(self, step: int) -> bool:
        """Whether anything under the node was set or selected at `step` or later."""
        return (
            any(value >= step for value in self.steps.values())
            or any(value >= step for value in self.off.values())
            or (self.selection is not None and self.selection.start >= step)
            or any(child.changed_since(step) for child in self.children.values())
        )

    def replace(self, since: int) -> None:
        """Drop what was set or selected under the node before `since`.

        That is what `_replace_` does, from the start of its layer. The
        default of the field goes too, and what is not kept is as if the node
        had no default. A group that was selected at `since` or later is kept
        as it is: its node is new. The option of the node itself stays.
        """
        self.default = MISSING
        for name, parts in list(self.parts.items()):
            kept = [part for part in parts if part.step >= since]
            if kept:
                self.parts[name] = kept
                self._merge(name)
                continue
            del self.parts[name], self.values[name]
            del self.sources[name], self.steps[name]
        self.touched = True
        for info in self.fields.values():
            if info.kind is Kind.VALUE:
                continue
            child = self.children.get(info.name)
            if child is None and self.off.get(info.name, -1) >= since:
                continue
            if child is not None:
                selected = (
                    child.selection is not None and child.selection.start >= since
                )
                if info.kind is Kind.GROUP and selected:
                    continue
                if child.changed_since(since):
                    child.replace(since)
                    continue
            self.children.pop(info.name, None)
            self.off.pop(info.name, None)
            self.expand_field(info)

    def to_json_tree(self) -> dict[str, Any]:
        """The node in JSON form, with ``_option_`` for the groups.

        A value that is missing is left out. A value that JSON has no form
        for is kept as it is.
        """
        result: dict[str, Any] = {}
        if self.option is not None:
            result[OPTION_KEY] = self.option
        for info in self.fields.values():
            if info.kind is Kind.VALUE:
                value = self.value(info)
                if value is not MISSING:
                    result[info.name] = to_json_form(value)
            elif info.name in self.children:
                result[info.name] = self.children[info.name].to_json_tree()
            elif self.is_off(info):
                result[info.name] = None
        return result

    def any_touched(self) -> bool:
        return self.touched or any(c.any_touched() for c in self.children.values())


def _from_json_tree(
    cls: type,
    tree: Mapping[Any, Any],
    path: Path,
    option: str | None,
    rebuild: _Rebuild,
) -> Node:
    node = Node(cls, path, option=option, expand=False)
    for key, value in tree.items():
        if key == OPTION_KEY and option is not None:
            # The option of a group node, which was read by the parent.
            continue
        info = node.fields.get(key) if isinstance(key, str) else None
        if info is None:
            where = format_path((*path, str(key)))
            message = unknown_key_message(node, str(key)).replace("\n", "\n    ")
            rebuild.errors.append(f"{where}: {message}")
            continue
        child_path = (*path, info.name)
        where = format_path(child_path)
        if info.kind is Kind.VALUE:
            node.set_value(info.name, value, rebuild.source, whole=True)
            continue
        if value is None and info.optional:
            # An optional subconfig or group that is off: it has no node.
            node.off[info.name] = -1
            continue
        what = "a group" if info.kind is Kind.GROUP else "a subconfig"
        if not isinstance(value, Mapping):
            rebuild.errors.append(
                f"{where}: this is {what} of {node.describe()}, so its value has "
                f"to be a mapping of its fields, not {value!r}."
            )
            continue
        value = cast("Mapping[Any, Any]", value)
        if info.kind is Kind.CONFIG:
            assert info.config_cls is not None
            node.children[info.name] = _from_json_tree(
                info.config_cls, value, child_path, None, rebuild
            )
            continue
        options = ", ".join(info.options)
        name = value.get(OPTION_KEY)
        if name is None:
            rebuild.errors.append(
                f"{where}: this is a group of {node.describe()}, so its mapping "
                f"has to name the option under the key {OPTION_KEY!r}. "
                f"Options: {options}."
            )
        elif not isinstance(name, str) or name not in info.options:
            rebuild.errors.append(
                f"{where}: {name!r} is not an option of this group of "
                f"{node.describe()}. Options: {options}."
            )
        else:
            node.children[info.name] = _from_json_tree(
                info.options[name], value, child_path, name, rebuild
            )
    for info in node.fields.values():
        if info.kind is Kind.VALUE or info.name in tree:
            continue
        # A subconfig or a group that the mapping leaves out takes the
        # default of the field, as it would without any config file.
        node.expand_field(info)
        if info.name not in node.children:
            rebuild.errors.append(
                f"{format_path((*path, info.name))}: the mapping has no key for "
                f"this group of {node.describe()}, and the field has no default. "
                f"Options: {', '.join(info.options)}."
            )
    return node


@dataclass(frozen=True)
class _Rebuild:
    source: str
    errors: list[str]


def node_from_json_tree(cls: type, tree: Mapping[Any, Any], *, source: str) -> Node:
    """Rebuild the tree from plain dicts, such as `Node.to_json_tree` gives.

    The mapping of a group names its option under ``_option_``. A key that the
    mapping leaves out takes the default in the class. `source` says where the
    values were set, for error messages.

    An unknown key, a missing or unknown option, and a subconfig that is not
    a mapping are errors, which are collected and raised together.
    """
    rebuild = _Rebuild(source, [])
    node = _from_json_tree(cls, tree, (), None, rebuild)
    if rebuild.errors:
        raise ConfigError("Error in the configuration:\n" + "\n".join(rebuild.errors))
    return node


def root_fields(config_cls: type, settings_cls: type) -> dict[str, FieldInfo]:
    """The fields of the root: the user's config, and nematocyte's own settings."""
    fields = dict(class_fields(config_cls))
    fields[RESERVED_KEY] = FieldInfo(
        name=RESERVED_KEY,
        kind=Kind.CONFIG,
        annotation=settings_cls,
        owner=None,
        config_cls=settings_cls,
    )
    return fields


def make_root(config_cls: type, settings_cls: type) -> Node:
    """The root of the tree: the user's config, and nematocyte's own settings."""
    return Node(config_cls, (), fields=root_fields(config_cls, settings_cls))


def switched_off_message(
    node: Node, info: FieldInfo, rest: Path = (), value: Any = MISSING
) -> str:
    """The error for a key under an optional subconfig or group that is off.

    `rest` is the path below the subconfig that was to be set, and `value`
    the value for it, if they are known; for a TypedDict, the error then
    shows the dict that switches it on with that value. A dataclass is
    switched on by a named config, and a group by selecting an option, which
    the error lists.
    """
    path = format_path((*node.path, info.name))
    what = (
        f"{format_path((*node.path, info.name, *rest))!r} cannot be set"
        if rest
        else "its fields cannot be set"
    )
    if info.kind is Kind.GROUP:
        first = next(iter(info.options))
        return (
            f"{path!r} is an optional group of {node.describe()}, and it is "
            f"switched off (null), so {what}. Selecting an option switches "
            f"it on: {path}={first} on the command line, or {path}: {first} "
            f"under '_defaults_' in a config file. "
            f"Options: {', '.join(info.options)}."
        )
    assert info.config_cls is not None
    name = info.config_cls.__name__
    start = (
        f"{path!r} is an optional subconfig of {node.describe()} "
        f"({name} | None), and it is switched off (null), so {what}."
    )
    if not is_typeddict_class(info.config_cls):
        return (
            f"{start} A named config for it switches it on: {path}=NAME on the "
            f"command line, or {path}: NAME under '_defaults_' in a config "
            f"file; so does a mapping in a config file, {path}: {{...}}. On "
            f"the command line, {name} is a dataclass, which is not given as "
            f"a dict."
        )
    example = f"{path}={{}}"
    if rest and value is not MISSING:
        nested = value
        for segment in reversed(rest):
            nested = {segment: nested}
        example = f"{path}={format_value(nested)}"
    return (
        f"{start} A mapping switches it on: {example} on the command line, "
        f"or {path}: {{...}} in a config file."
    )


def unknown_key_message(node: Node, name: str) -> str:
    """The error for a key that is not a field of the node's class."""
    names = [name for name in node.fields if name != RESERVED_KEY]
    message = f"{name!r} is not a field of {node.describe()}."
    if names:
        message += f"\nAvailable fields: {', '.join(names)}"
        close = difflib.get_close_matches(name, names, n=1)
        if close:
            message += f"\nDid you mean {close[0]!r}?"
    else:
        message += f"\n{node.class_name} has no fields."
    return message


def require_child(node: Node, info: FieldInfo) -> Node:
    """The node of a subconfig or group field.

    An error if no option is selected, or if an optional subconfig or group
    is off.
    """
    child = node.children.get(info.name)
    if child is None:
        path = format_path((*node.path, info.name))
        if node.is_off(info):
            raise ConfigError(switched_off_message(node, info))
        hint = f" To switch it off: {path}=null." if info.optional else ""
        raise ConfigError(
            f"No option is selected for the group {path!r}. "
            f"Options: {', '.join(info.options)}.{hint}"
        )
    return child
