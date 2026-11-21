"""Turning the composed tree into config instances, with one call to pydantic."""

from collections.abc import Collection, Mapping
import dataclasses
from dataclasses import MISSING
import math
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from ._errors import ConfigError
from ._overrides import format_value
from ._plain import to_plain
from ._schema import (
    OPTION_KEY,
    FieldInfo,
    Kind,
    Selected,
    class_adapter,
    field_adapter,
    is_dict_or_frozendict,
    is_exact_dict_or_frozendict,
    is_typeddict_class,
)
from ._tree import Node, format_path, unknown_key_message
from ._types import type_text

__all__ = ["build"]


def _input(node: Node, skip: str | None) -> dict[str, Any]:
    """The node as the input for pydantic.

    The node of a group of dataclasses is a `Selected`, from which
    `nematocyte.pick` reads the option. A value or a subconfig that is
    missing is left out, and pydantic reports it.
    """
    data: dict[str, Any] = {}
    if node.option is not None and is_typeddict_class(node.cls):
        data[OPTION_KEY] = node.option
    for info in node.fields.values():
        if info.name == skip:
            continue
        if info.kind is Kind.VALUE:
            value = node.value(info)
            if value is not MISSING:
                data[info.name] = value
            continue
        child = node.children.get(info.name)
        if child is None:
            continue
        fields = _input(child, None)
        if child.option is not None and not is_typeddict_class(child.cls):
            data[info.name] = Selected(child.option, fields)
        else:
            data[info.name] = fields
    return data


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
            messages.append(
                f"{path}: no option is selected for this group. Select one "
                f"with {path}=NAME, or in the '_defaults_' of a config file. "
                f"Options: {', '.join(info.options)}"
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


def _check_values(
    node: Node, errors: list[str], skip: str | None, failed: set[_FieldKey]
) -> None:
    """The checks that pydantic does not make, for every value it accepted."""
    for info in node.fields.values():
        if info.name == skip:
            continue
        if info.kind is not Kind.VALUE:
            child = node.children.get(info.name)
            if child is not None:
                _check_values(child, errors, None, failed)
            continue
        if (id(node), info.name) in failed:
            continue
        problem = _value_problem(node, info)
        if problem is not None:
            path = format_path((*node.path, info.name))
            errors.append(f"{path}: {problem}{_field_line(node, info)}")


def _value_problem(node: Node, info: FieldInfo) -> str | None:
    """A bool that pydantic turned into a number, or a default that loses its type.

    Both compare with what pydantic makes of the value of this field on its
    own, before a constructor can change it. The value is only validated
    again when it holds a bool, or is a default that is not plain.
    """
    if info.name in node.values:
        value, default = node.values[info.name], MISSING
    else:
        # The plain form is made from the same object that it is compared
        # with: a `default_factory` makes a new one each time.
        default = node.default_value(info)
        if default is MISSING:
            return None
        value = to_plain(default)
    unset_default = default is not MISSING and not _is_plain(default)
    if not unset_default and not _has_bool(value):
        return None
    adapter = field_adapter(info)
    if adapter is None:
        # A type that only the class's own pydantic config lets through.
        return None
    try:
        result = adapter.validate_python(value, extra="forbid")
    except ValidationError:
        return None
    origin = _origin(node, info)
    problem = _bool_problem(value, result, origin)
    if problem is None and unset_default:
        problem = _default_problem(default, adapter, result, origin)
    return problem


def _has_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_bool(item) for item in value)
    if isinstance(value, dict):
        return any(_has_bool(item) for item in value.values())
    return False


def _bool_problem(value: Any, result: Any, origin: str) -> str | None:
    """The error if pydantic turned a bool into a number."""
    inner = _bool_to_number(value, result)
    if inner is None:
        return None
    location = f" (at {inner})" if inner else ""
    return (
        f"a bool is not a number{location}; write 1 or 0 instead."
        f"\n    Value: {value!r} ({origin})"
    )


def _default_problem(
    default: Any, adapter: TypeAdapter[Any], result: Any, origin: str
) -> str | None:
    """The error if a default that nothing set loses its type in plain form.

    `result` is what pydantic made of the plain form of the default. It is
    compared with what pydantic makes of the default itself, so that a
    conversion that pydantic does anyway, such as `1` to `1.0` for a float,
    does not count.
    """
    try:
        direct = adapter.validate_python(default, extra="forbid")
    except ValidationError:
        # Only the plain form is valid, as with `name: str = Path("x")`.
        return None
    lost = _difference(direct, result)
    if lost is None:
        return None
    inner, before, after = lost
    location = f" (at {inner})" if inner else ""
    return (
        f"the default loses its type{location}: as a config file would hold "
        f"it, {before!r} becomes {after!r}. Give the field a type that builds "
        f"{type(before).__name__} from that, or a default that a config file "
        f"can hold.\n    Value: {default!r} ({origin})"
    )


_PLAIN_SCALARS = frozenset({str, int, float, bool, type(None)})
_PLAIN_COLLECTIONS = frozenset({list, tuple, set, frozenset})


def _is_plain(value: Any) -> bool:
    """Whether `value` holds only plain scalars, in built-in containers.

    Its plain form then differs from it only in the kinds of its containers
    and the order of a set, which a config file cannot express, so the
    check of a default can skip the second validation. Subclasses, such as a `NamedTuple` or an
    `IntEnum`, are not plain.
    """
    kind = type(value)
    if kind in _PLAIN_SCALARS:
        return True
    if kind in _PLAIN_COLLECTIONS:
        return all(_is_plain(item) for item in value)
    if is_exact_dict_or_frozendict(value):
        return all(_is_plain(key) and _is_plain(item) for key, item in value.items())
    return False


type _Pairs = list[tuple[str, Any, Any]]


def _difference(expected: Any, actual: Any) -> tuple[str, Any, Any] | None:
    """The first place where `actual` differs from `expected`, with both values.

    A list, tuple, set and frozenset count as the same, as do a dict and a
    frozendict: a config file cannot tell them apart. The location is `""`
    for the value itself, as for `_bool_to_number`.
    """
    if expected is actual:
        return None
    pairs = _parts(expected, actual)
    if pairs is None:
        return "", expected, actual
    for location, item, out in pairs:
        found = _difference(item, out)
        if found is not None:
            inner, before, after = found
            joined = f"{location}.{inner}" if location and inner else location or inner
            return joined, before, after
    return None


def _parts(expected: Any, actual: Any) -> _Pairs | None:
    """The parts of two values that are compared; `None` if they differ already."""
    if is_dict_or_frozendict(expected) and is_dict_or_frozendict(actual):
        return _mapping_parts(expected, actual)
    if isinstance(expected, _COLLECTIONS) and isinstance(actual, _COLLECTIONS):
        return _collection_parts(expected, actual)
    if type(expected) is not type(actual):
        return None
    if dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        return [
            (field.name, getattr(expected, field.name), getattr(actual, field.name))
            for field in dataclasses.fields(expected)
            if field.init
        ]
    return [] if _equal(expected, actual) else None


_COLLECTIONS = (list, tuple, set, frozenset)


def _mapping_parts(
    expected: Mapping[Any, Any], actual: Mapping[Any, Any]
) -> _Pairs | None:
    if len(expected) != len(actual):
        return None
    keys = {key: key for key in actual}
    pairs: _Pairs = []
    for key, item in expected.items():
        if key not in keys:
            return None
        pairs.append(("", key, keys[key]))
        pairs.append((str(key), item, actual[key]))
    return pairs


def _collection_parts(
    expected: Collection[Any], actual: Collection[Any]
) -> _Pairs | None:
    if len(expected) != len(actual):
        return None
    if isinstance(expected, (set, frozenset)) or isinstance(actual, (set, frozenset)):
        # Without an order, each item is matched with an equal one.
        remaining = list(actual)
        for item in expected:
            index = next(
                (
                    i
                    for i, out in enumerate(remaining)
                    if _difference(item, out) is None
                ),
                None,
            )
            if index is None:
                return None
            del remaining[index]
        return []
    return [
        (str(index), item, out)
        for index, (item, out) in enumerate(zip(expected, actual, strict=True))
    ]


def _equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) and math.isnan(expected):
        return math.isnan(actual)
    return bool(expected == actual)


def _bool_to_number(value: Any, result: Any) -> str | None:
    """Where pydantic turned a bool into a number; `None` if it did not.

    pydantic's lax mode accepts `true` for an `int` or a `float`, as `1`.
    The bools are found by comparing the value with the result, also inside
    lists, tuples and dicts. The location is `""` for the value itself.
    """
    if isinstance(value, bool):
        return None if isinstance(result, bool) else ""
    pairs: list[tuple[str, Any, Any]] = []
    sequences = isinstance(value, (list, tuple)) and isinstance(result, (list, tuple))
    mappings = isinstance(value, dict) and isinstance(result, dict)
    if sequences and len(value) == len(result):
        pairs = [
            (str(index), item, out)
            for index, (item, out) in enumerate(zip(value, result, strict=True))
        ]
    elif mappings and len(value) == len(result):
        pairs = [
            (str(key), item, out)
            for (key, item), out in zip(value.items(), result.values(), strict=True)
        ]
    for location, item, out in pairs:
        inner = _bool_to_number(item, out)
        if inner is not None:
            return f"{location}.{inner}" if inner else location
    return None


def build(node: Node, *, skip: str | None = None) -> object:
    """Build the instance for `node`, and for everything under it.

    The whole tree goes to pydantic in one call, which checks every value
    and runs the constructors, and with them every `__post_init__`. A
    `TypedDict` node becomes a plain dict. All the errors in the tree are
    collected and reported together. `skip` names a field of `node` to
    leave out.
    """
    assert node.cls is not None
    adapter = class_adapter(node.cls)
    errors: list[str] = []
    failed: set[_FieldKey] = set()
    instance: object = None
    try:
        instance = adapter.validate_python(_input(node, skip), extra="forbid")
    except ValidationError as exc:
        errors, failed = _translate(node, exc)
    except Exception as exc:
        # A constructor raised something other than a `ValueError` or an
        # `AssertionError`; pydantic passes it on, without its location.
        errors = [f"{type(exc).__name__}: {exc}"]
    _check_values(node, errors, skip, failed)
    if errors:
        raise ConfigError("Error in the configuration:\n" + "\n".join(errors))
    return instance
