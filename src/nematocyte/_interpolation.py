"""Resolving `${...}` in the values of the composed tree."""

from dataclasses import MISSING
from datetime import datetime
import re
from typing import Any

from ._errors import ConfigError
from ._schema import Kind
from ._tree import Node, format_path

__all__ = ["resolve"]

_REFERENCE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*")
_RESOLVERS = ("now",)


class _Reference:
    __slots__ = ("body",)

    def __init__(self, body: str) -> None:
        self.body = body


def _split(text: str) -> list[str | _Reference]:
    """Split a string into literal text and the bodies of its interpolations."""
    parts: list[str | _Reference] = []
    pos = 0
    while True:
        start = text.find("${", pos)
        if start == -1:
            if pos < len(text):
                parts.append(text[pos:])
            return parts
        if start > pos:
            parts.append(text[pos:start])
        end = text.find("}", start)
        if end == -1:
            raise ConfigError(f"The interpolation in {text!r} is not closed with '}}'.")
        body = text[start + 2 : end]
        if "${" in body:
            raise ConfigError(
                f"{text!r} has an interpolation inside an interpolation, which "
                "is not supported."
            )
        parts.append(_Reference(body))
        pos = end + 1


class _Resolver:
    def __init__(self, root: Node, now: datetime) -> None:
        self.root = root
        self.now = now
        self.done: set[tuple[int, str]] = set()
        self.active: list[tuple[int, str, str]] = []

    def resolve_all(self, node: Node) -> None:
        for name in list(node.values):
            self.resolved_value(node, name)
        for child in node.children.values():
            self.resolve_all(child)

    def resolved_value(self, node: Node, name: str) -> Any:
        """Resolve the value that was set for `name`, in place."""
        key = (id(node), name)
        if key in self.done:
            return node.values[name]
        where = format_path((*node.path, name))
        if any(entry[:2] == key for entry in self.active):
            cycle = " -> ".join([*(entry[2] for entry in self.active), where])
            raise ConfigError(
                f"The interpolations refer to each other in a cycle: {cycle}"
            )
        self.active.append((*key, where))
        try:
            node.values[name] = self._resolve(node.values[name])
        except ConfigError as exc:
            if len(self.active) > 1 or str(exc).startswith("Error in"):
                raise
            raise ConfigError(
                f"Error in the value of {where!r}, set in {node.sources[name]}:\n{exc}"
            ) from None
        finally:
            self.active.pop()
        self.done.add(key)
        return node.values[name]

    def _resolve(self, value: Any) -> Any:
        if isinstance(value, str):
            if "${" not in value:
                return value
            parts = _split(value)
            if len(parts) == 1 and isinstance(parts[0], _Reference):
                # Exactly one interpolation: the value keeps the type of its target.
                return self._evaluate(parts[0].body)
            return "".join(
                part if isinstance(part, str) else str(self._evaluate(part.body))
                for part in parts
            )
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve(item) for key, item in value.items()}
        return value

    def _evaluate(self, body: str) -> Any:
        expression = "${" + body + "}"
        if ":" in body:
            name, _, argument = body.partition(":")
            name = name.strip()
            if name != "now":
                raise ConfigError(
                    f"{expression}: there is no resolver called {name!r}. "
                    f"The only resolver is 'now', as in ${{now:%Y-%m-%d}}."
                )
            return self.now.strftime(argument)
        target = body.strip()
        if target.startswith("."):
            raise ConfigError(
                f"{expression}: relative references are not supported. Write "
                "the full path from the root of the config, such as ${dm.root}."
            )
        if _REFERENCE.fullmatch(target) is None:
            raise ConfigError(
                f"{expression} is not a valid interpolation. A reference is a "
                "dotted path to a value, such as ${dm.root}, and the only "
                "resolver is ${now:FORMAT}."
            )
        return self._lookup(tuple(target.split(".")), expression)

    def _lookup(self, path: tuple[str, ...], expression: str) -> Any:
        node = self.root
        for index, segment in enumerate(path):
            info = node.fields.get(segment)
            if info is None:
                raise ConfigError(
                    f"{expression} refers to a key that does not exist: "
                    f"{segment!r} is not a field of {node.describe()}.\n"
                    f"Available fields: {', '.join(node.fields)}"
                )
            if info.kind is not Kind.VALUE:
                child = node.children.get(segment)
                here = format_path(path[: index + 1])
                if child is None:
                    raise ConfigError(
                        f"{expression}: no option is selected for the group {here!r}."
                    )
                if index == len(path) - 1:
                    raise ConfigError(
                        f"{expression} refers to {here!r}, which is a subconfig "
                        "and not a value."
                    )
                node = child
                continue
            if segment in node.values:
                value = self.resolved_value(node, segment)
            else:
                value = node.value(info)
                if value is MISSING:
                    raise ConfigError(
                        f"{expression} refers to "
                        f"{format_path(path[: index + 1])!r}, which has no value."
                    )
            return self._descend(value, path[index + 1 :], expression)
        raise ConfigError(f"{expression} is empty.")

    @staticmethod
    def _descend(value: Any, rest: tuple[str, ...], expression: str) -> Any:
        """Follow the rest of a path into the keys of a dict value."""
        for segment in rest:
            if isinstance(value, dict):
                if segment in value:
                    value = value[segment]
                    continue
                if segment.isdigit() and int(segment) in value:
                    value = value[int(segment)]
                    continue
            raise ConfigError(
                f"{expression}: there is no key {segment!r} in the value that "
                "the path leads to."
            )
        return value


def resolve(root: Node, now: datetime, *, only: Node | None = None) -> None:
    """Resolve every interpolation in the tree, in place.

    This runs after all files are merged and all overrides are applied, and
    before any value is validated. With `only`, just the values under that
    node are resolved, and the values that they refer to.
    """
    _Resolver(root, now).resolve_all(root if only is None else only)
