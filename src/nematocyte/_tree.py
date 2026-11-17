"""The composed config, as a tree of nodes that mirrors the dataclasses."""

from dataclasses import MISSING
import difflib
from typing import Any

from ._errors import ConfigError, SchemaError
from ._plain import to_plain
from ._schema import OPTION_KEY, RESERVED_KEY, FieldInfo, Kind, class_fields

__all__ = ["Node", "Path", "format_path", "make_root", "unknown_key_message"]

type Path = tuple[str, ...]


def format_path(path: Path) -> str:
    return ".".join(path)


class Node:
    """One dataclass in the composed config.

    A node knows its class, the values that config files and overrides set
    on it, and where every other value comes from: the default instance of
    the field, if there is one, and the defaults of the class otherwise.
    """

    def __init__(
        self,
        cls: type | None,
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
        if fields is None:
            assert cls is not None
            fields = class_fields(cls)
        self.fields = fields
        self.children: dict[str, Node] = {}
        self.values: dict[str, Any] = {}
        """The values that were set, as they came from a file or an override."""
        self.sources: dict[str, str] = {}
        """Where each of the values was set, for error messages."""
        self.touched = False
        """Whether a config file or an override selected or set something here."""
        if expand:
            for info in self.fields.values():
                if info.kind is not Kind.VALUE and info.name not in self.children:
                    self.expand_field(info)

    def expand_field(self, info: FieldInfo) -> None:
        if self.default is not MISSING and not info.initvar:
            default = getattr(self.default, info.name)
        else:
            default = info.class_default()
        where = f"{self.class_name}.{info.name}"
        path = (*self.path, info.name)
        if info.kind is Kind.CONFIG:
            assert info.config_cls is not None
            if default is not MISSING and type(default) is not info.config_cls:
                raise SchemaError(
                    f"{where}: the default is an instance of "
                    f"{type(default).__name__}, but the field is declared as "
                    f"{info.config_cls.__name__}."
                )
            self.children[info.name] = Node(info.config_cls, path, default=default)
        elif default is not MISSING:
            name = info.option_of(type(default))
            if name is None:
                raise SchemaError(
                    f"{where}: the default is an instance of "
                    f"{type(default).__name__}, which is not an option of the "
                    f"group. Options: {', '.join(info.options)}."
                )
            self.children[info.name] = Node(
                type(default), path, default=default, option=name
            )

    @property
    def class_name(self) -> str:
        return self.cls.__name__ if self.cls is not None else "the config"

    def describe(self) -> str:
        """The class of the node and where it is, for error messages."""
        if not self.path:
            return self.class_name
        where = format_path(self.path)
        if self.option is not None:
            return f"{self.class_name} (the option {self.option!r} of {where!r})"
        return f"{self.class_name} (at {where!r})"

    def default_value(self, info: FieldInfo) -> Any:
        """The value of a field that nothing has set, or ``MISSING``."""
        if self.default is not MISSING and not info.initvar:
            return getattr(self.default, info.name)
        return info.class_default()

    def value(self, info: FieldInfo) -> Any:
        """The value of a field in plain form, or ``MISSING``."""
        if info.name in self.values:
            return self.values[info.name]
        default = self.default_value(info)
        return default if default is MISSING else to_plain(default)

    def set_value(self, name: str, value: Any, source: str) -> None:
        self.values[name] = value
        self.sources[name] = source
        self.touched = True

    def to_plain_tree(self) -> dict[str, Any]:
        """The node as plain dicts, with ``_option_`` for the groups.

        A value that is missing is left out.
        """
        result: dict[str, Any] = {}
        if self.option is not None:
            result[OPTION_KEY] = self.option
        for info in self.fields.values():
            if info.kind is Kind.VALUE:
                value = self.value(info)
                if value is not MISSING:
                    result[info.name] = value
            elif info.name in self.children:
                result[info.name] = self.children[info.name].to_plain_tree()
        return result

    def any_touched(self) -> bool:
        return self.touched or any(c.any_touched() for c in self.children.values())


def node_from_plain(
    cls: type, plain: dict[str, Any], path: Path = (), option: str | None = None
) -> Node:
    """Rebuild a node from the output of :meth:`Node.to_plain_tree`."""
    node = Node(cls, path, option=option, expand=False)
    for info in node.fields.values():
        if info.name not in plain:
            continue
        value = plain[info.name]
        child_path = (*path, info.name)
        if info.kind is Kind.VALUE:
            node.set_value(info.name, value, "the config of the job")
        elif info.kind is Kind.CONFIG:
            assert info.config_cls is not None
            node.children[info.name] = node_from_plain(
                info.config_cls, value, child_path
            )
        else:
            name = value[OPTION_KEY]
            node.children[info.name] = node_from_plain(
                info.options[name], value, child_path, name
            )
    return node


def make_root(config_cls: type, settings_cls: type) -> Node:
    """The root of the tree: the user's config, and nematocyte's own settings."""
    fields = dict(class_fields(config_cls))
    fields[RESERVED_KEY] = FieldInfo(
        name=RESERVED_KEY,
        kind=Kind.CONFIG,
        annotation=settings_cls,
        owner=None,
        config_cls=settings_cls,
    )
    root = Node(config_cls, (), fields=fields, expand=False)
    for info in fields.values():
        if info.name == RESERVED_KEY:
            root.children[RESERVED_KEY] = Node(settings_cls, (RESERVED_KEY,))
        elif info.kind is not Kind.VALUE:
            root.expand_field(info)
    return root


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
    """The node of a subconfig or group field; an error if no option is selected."""
    child = node.children.get(info.name)
    if child is None:
        path = format_path((*node.path, info.name))
        raise ConfigError(
            f"No option is selected for the group {path!r}. "
            f"Options: {', '.join(info.options)}."
        )
    return child
