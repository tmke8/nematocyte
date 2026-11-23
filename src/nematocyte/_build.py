"""Turning the composed tree into config instances, with one call to pydantic."""

from collections.abc import Iterator
from dataclasses import MISSING
import json
from typing import Any

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from ._errors import ConfigError
from ._json_form import first_difference, is_json_form, to_json_form, without_json_form
from ._overrides import format_value
from ._schema import RESERVED_KEY, FieldInfo, Kind, class_adapter, field_adapter
from ._settings import Settings
from ._tree import Node, format_path, unknown_key_message
from ._types import type_text

__all__ = ["build", "build_settings", "checked_json"]


def _value_fields(node: Node, skip: str | None) -> Iterator[tuple[Node, FieldInfo]]:
    """Every field that holds a value in the tree under `node`, with its node.

    `skip` names a field of `node` to leave out.
    """
    for info in node.fields.values():
        if info.name == skip:
            continue
        if info.kind is Kind.VALUE:
            yield node, info
        else:
            child = node.children.get(info.name)
            if child is not None:
                yield from _value_fields(child, None)


def _json_problems(root: Node, skip: str | None) -> list[str]:
    """The errors for every value in the tree that JSON has no form for."""
    errors: list[str] = []
    for node, info in _value_fields(root, skip):
        value = node.value(info)
        if value is MISSING:
            continue
        found = without_json_form(to_json_form(value))
        if found is None:
            continue
        inner, part = found
        location = f" (at {inner})" if inner else ""
        if info.name in node.sources:
            hint = "Write it as a string, a number, a bool, a list or a mapping."
        else:
            hint = (
                "A value that a config file cannot hold belongs in a field "
                "with init=False, set in __post_init__, or in a property."
            )
        path = format_path((*node.path, info.name))
        errors.append(
            f"{path}: JSON has no form for {part!r} "
            f"({type(part).__name__}){location}, so it cannot be checked. "
            f"{hint}\n    Value: {value!r} ({_origin(node, info)})"
            f"{_field_line(node, info)}"
        )
    return errors


type _Loc = tuple[int | str, ...]


def _locate(node: Node, loc: _Loc) -> tuple[Node, FieldInfo | None, _Loc]:
    """The node and the field that an error of pydantic is about.

    The rest of the location is inside the value of the field; with no
    field, it is inside the node. The option of a group, which pydantic
    puts into the location if the union has a `Discriminator`, is skipped.
    """
    current = node
    index = 0
    while index < len(loc):
        key = loc[index]
        info = current.fields.get(key) if isinstance(key, str) else None
        if info is None:
            return current, None, loc[index:]
        child = current.children.get(info.name)
        if info.kind is Kind.VALUE or child is None:
            return current, info, loc[index + 1 :]
        index += 2 if info.tagged else 1
        current = child
    return current, None, ()


def _origin(node: Node, info: FieldInfo) -> str:
    if info.name in node.sources:
        return f"set in {node.sources[info.name]}"
    return f"the default of {node.class_name}.{info.name}"


def _field_line(node: Node, info: FieldInfo) -> str:
    described = f"{node.class_name}.{info.name} ({type_text(info.annotation)})"
    return f"\n    Field: {described}{'. ' + info.doc if info.doc else ''}"


def _quote_hint(node: Node, info: FieldInfo, value: Any) -> str:
    source = node.sources.get(info.name, "")
    if not source.startswith("the override"):
        return ""
    text = format_value(value)
    key = format_path((*node.path, info.name))
    return (
        f"\n    To pass a string that looks like something else, quote it, and "
        f"protect the quotes from the shell: \"{key}='{text}'\""
    )


def _value_error(node: Node, info: FieldInfo, details: list[ErrorDetails]) -> str:
    """The error for a value that pydantic rejected."""
    path = format_path((*node.path, info.name))
    value = node.value(info)
    if value is MISSING:
        return (
            f"{path}: a value is required. The field {info.name!r} of "
            f"{node.describe()} has no default."
        )
    lines: list[str] = []
    hint = ""
    for error in details:
        inner = ".".join(str(part) for part in error["loc"])
        location = f" (at {inner})" if inner else ""
        lines.append(f"{error['msg']}{location}")
        scalar = isinstance(value, (bool, int, float))
        if error["type"] == "string_type" and not inner and scalar:
            hint = _quote_hint(node, info, value)
    if len(lines) == 1:
        message = f"{path}: {lines[0]}."
    else:
        message = f"{path}: the value fits none of the allowed types:" + "".join(
            f"\n    - {line}" for line in lines
        )
    message += f"\n    Value: {value!r} ({_origin(node, info)})"
    return message + _field_line(node, info) + hint


def _node_error(node: Node, rest: _Loc, error: ErrorDetails) -> str:
    """The error for a node: an unknown key, or one raised by its constructor."""
    if rest and error["type"] in ("unexpected_keyword_argument", "extra_forbidden"):
        where = format_path((*node.path, str(rest[0])))
        message = unknown_key_message(node, str(rest[0])).replace("\n", "\n    ")
        return f"{where}: {message}"
    cause = error.get("ctx", {}).get("error")
    text = str(cause) if isinstance(cause, Exception) else error["msg"]
    where = f"{format_path(node.path)} ({node.class_name}): " if node.path else ""
    return f"{where}{text}"


type _FieldKey = tuple[int, str]


def _translate(node: Node, exc: ValidationError) -> tuple[list[str], set[_FieldKey]]:
    """pydantic's errors, as nematocyte reports them, and the values they are about."""
    by_field: dict[_FieldKey, tuple[Node, FieldInfo, list[ErrorDetails]]] = {}
    messages: list[str | _FieldKey] = []
    for error in exc.errors():
        found, info, rest = _locate(node, error["loc"])
        if info is None:
            messages.append(_node_error(found, rest, error))
        elif info.kind is not Kind.VALUE:
            path = format_path((*found.path, info.name))
            off = f", or switch it off with {path}=null" if info.optional else ""
            messages.append(
                f"{path}: no option is selected for this group. Select one "
                f"with {path}=NAME, or in the '_defaults_' of a config "
                f"file{off}. Options: {', '.join(info.options)}"
            )
        else:
            key = (id(found), info.name)
            if key not in by_field:
                by_field[key] = (found, info, [])
                messages.append(key)
            by_field[key][2].append({**error, "loc": rest})
    texts = [
        message if isinstance(message, str) else _value_error(*by_field[message])
        for message in messages
    ]
    return texts, set(by_field)


def _default_problems(
    root: Node, skip: str | None, failed: set[_FieldKey]
) -> list[str]:
    """The check that pydantic does not make, for every default it accepted."""
    errors: list[str] = []
    for node, info in _value_fields(root, skip):
        if info.name in node.values or (id(node), info.name) in failed:
            continue
        problem = _default_problem(node, info)
        if problem is not None:
            path = format_path((*node.path, info.name))
            errors.append(f"{path}: {problem}{_field_line(node, info)}")
    return errors


def _default_problem(node: Node, info: FieldInfo) -> str | None:
    """The error if a default that nothing set loses its type in JSON form.

    What pydantic makes of the JSON form of the default is compared with
    what it makes of the default itself, for this field on its own, before
    a constructor can change it. A conversion that pydantic does anyway,
    such as `1` to `1.0` for a float, does not count. Only a default that
    `to_json_form` changes in more than the kinds of its containers is checked:
    only then can the two forms give different results.
    """
    default = node.default_value(info)
    if default is MISSING or is_json_form(default):
        return None
    adapter = field_adapter(info)
    if adapter is None:
        # A type that only the class's own pydantic config lets through.
        return None
    try:
        # The JSON form is made from the same object that it is compared
        # with: a `default_factory` makes a new one each time.
        text = json.dumps(to_json_form(default))
        result = adapter.validate_json(text, strict=True)
    except TypeError, ValueError:
        # No JSON form, which the build reports, or a value that pydantic
        # rejects (a `ValidationError` is a `ValueError`).
        return None
    try:
        direct = adapter.validate_python(default)
    except ValidationError:
        # Only the JSON form is valid, as with `name: str = Path("x")`.
        return None
    lost = first_difference(direct, result)
    if lost is None:
        return None
    inner, before, after = lost
    location = f" (at {inner})" if inner else ""
    return (
        f"the default loses its type{location}: as a config file would hold "
        f"it, {before!r} becomes {after!r}. Give the field a type that builds "
        f"{type(before).__name__} from that, or a default that a config file "
        f"can hold.\n    Value: {default!r} ({_origin(node, info)})"
    )


def build(node: Node, *, skip: str | None = None) -> object:
    """Build the instance for `node`, and for everything under it.

    The whole tree goes to pydantic in one call, as JSON and in strict mode,
    which checks every value and runs the constructors, and with them every
    `__post_init__`. A value that JSON has no form for is an error. A
    `TypedDict` node becomes a plain dict. All the errors in the tree are
    collected and reported together. `skip` names a field of `node` to
    leave out.
    """
    return _build(node, skip)[0]


def build_settings(root: Node) -> Settings:
    """Build nematocyte's own settings, from the root of a composed tree."""
    settings = build(root.children[RESERVED_KEY])
    assert isinstance(settings, Settings)
    return settings


def checked_json(node: Node, *, skip: str | None = None) -> str:
    """Build the instance for `node` as `build` does, and return the JSON text.

    The text is what pydantic validated, so validating it again with the
    adapter of the node's class gives the same instance, without the tree.
    """
    return _build(node, skip)[1]


def _build(node: Node, skip: str | None) -> tuple[object, str]:
    adapter = class_adapter(node.cls)
    # The node of a group names its option under `_option_`, from which
    # `nematocyte.pick` reads it. A value or a subconfig that is missing is
    # left out, and pydantic reports it.
    data = node.to_json_tree()
    if skip is not None:
        data.pop(skip, None)
    try:
        text = json.dumps(data)
    except (TypeError, ValueError) as exc:
        problems = _json_problems(node, skip) or [f"The config has no JSON form: {exc}"]
        raise ConfigError(
            "Error in the configuration:\n" + "\n".join(problems)
        ) from None
    errors: list[str] = []
    failed: set[_FieldKey] = set()
    instance: object = None
    try:
        instance = adapter.validate_json(text, strict=True)
    except ValidationError as exc:
        errors, failed = _translate(node, exc)
    except Exception as exc:
        # A constructor raised something other than a `ValueError` or an
        # `AssertionError`; pydantic passes it on, without its location.
        errors = [f"{type(exc).__name__}: {exc}"]
    errors += _default_problems(node, skip, failed)
    if errors:
        raise ConfigError("Error in the configuration:\n" + "\n".join(errors))
    return instance, text
