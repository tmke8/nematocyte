"""Turning the composed tree into config instances, with pydantic per value."""

from dataclasses import MISSING
from typing import Any
from typing_extensions import sentinel

from pydantic import (
    ConfigDict,
    PydanticSchemaGenerationError,
    PydanticUserError,
    TypeAdapter,
    ValidationError,
)

from ._errors import ConfigError, SchemaError
from ._overrides import format_value
from ._schema import OPTION_KEY, FieldInfo, Kind, is_typeddict_class
from ._tree import Node, format_path
from ._types import type_text

__all__ = ["build"]

_FAILED = sentinel("_FAILED")
_UNSET = sentinel("_UNSET")
_ADAPTERS: dict[tuple[type | None, str], tuple[TypeAdapter[Any], bool]] = {}


def _adapter(info: FieldInfo) -> tuple[TypeAdapter[Any], bool]:
    """The adapter for a field, and whether pydantic can read its type."""
    key = (info.owner, info.name)
    cached = _ADAPTERS.get(key)
    if cached is not None:
        return cached
    try:
        cached = (TypeAdapter(info.annotation), True)
    except (PydanticSchemaGenerationError, PydanticUserError) as first:
        # A type that pydantic does not know. A value of it cannot be read
        # from a config file, but a default of that type is still accepted.
        # pydantic checks the default with `isinstance` if the type is a
        # class, and not at all if it is not one, such as a sentinel.
        try:
            adapter = TypeAdapter(
                info.annotation, config=ConfigDict(arbitrary_types_allowed=True)
            )
        except Exception:
            owner = info.owner.__name__ if info.owner is not None else "?"
            raise SchemaError(
                f"{owner}.{info.name}: pydantic cannot handle the type "
                f"{info.annotation!r}:\n{first}"
            ) from first
        cached = (adapter, False)
    _ADAPTERS[key] = cached
    return cached


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


def _check_value(
    node: Node, info: FieldInfo, errors: list[str]
) -> object | _FAILED | _UNSET:
    path = format_path((*node.path, info.name))
    value = node.value(info)
    if value is MISSING and not info.required:
        # A `NotRequired` key of a `TypedDict` that nothing set is left out.
        return _UNSET
    if value is MISSING:
        errors.append(
            f"{path}: a value is required. The field {info.name!r} of "
            f"{node.describe()} has no default."
        )
        return _FAILED
    if info.name in node.sources:
        origin = f"set in {node.sources[info.name]}"
    else:
        origin = f"the default of {node.class_name}.{info.name}"
    described = f"{node.class_name}.{info.name} ({type_text(info.annotation)})"
    field_line = f"\n    Field: {described}{'. ' + info.doc if info.doc else ''}"
    adapter, readable = _adapter(info)
    if not readable and info.name in node.sources and not node.checked:
        errors.append(
            f"{path}: this field cannot be set in a config file, on the command "
            f"line or with from_run_config(), because pydantic cannot read its type. "
            f"Only the default in the class can be used."
            f"\n    Value: {value!r} ({origin}){field_line}"
        )
        return _FAILED
    try:
        result = adapter.validate_python(value, extra="forbid")
    except ValidationError as exc:
        lines: list[str] = []
        hint = ""
        for error in exc.errors():
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
        message += f"\n    Value: {value!r} ({origin}){field_line}"
        errors.append(message + hint)
        return _FAILED
    inner = _bool_to_number(value, result)
    if inner is not None:
        location = f" (at {inner})" if inner else ""
        errors.append(
            f"{path}: a bool is not a number{location}; write 1 or 0 instead."
            f"\n    Value: {value!r} ({origin}){field_line}"
        )
        return _FAILED
    return result


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


def _build(
    node: Node, errors: list[str], *, construct: bool, skip: str | None
) -> object | _FAILED:
    kwargs: dict[str, Any] = {}
    failed = False
    for info in node.fields.values():
        if info.name == skip:
            continue
        if info.kind is Kind.VALUE:
            value = _check_value(node, info, errors)
        else:
            child = node.children.get(info.name)
            if child is None:
                path = format_path((*node.path, info.name))
                errors.append(
                    f"{path}: no option is selected for this group. Select one "
                    f"with {path}=NAME, or in the '_defaults_' of a config file. "
                    f"Options: {', '.join(info.options)}"
                )
                value = _FAILED
            else:
                value = _build(child, errors, construct=construct, skip=None)
        if value is _FAILED:
            failed = True
        elif value is not _UNSET:
            kwargs[info.name] = value
    if failed or not construct:
        return _FAILED
    assert node.cls is not None
    if node.option is not None and is_typeddict_class(node.cls):
        # A TypedDict that is an option of a group says which one it is.
        kwargs = {OPTION_KEY: node.option, **kwargs}
    try:
        return node.cls(**kwargs)
    except Exception as exc:
        where = f"{format_path(node.path)} ({node.class_name}): " if node.path else ""
        errors.append(f"{where}{exc}")
        return _FAILED


def build(node: Node, *, construct: bool = True, skip: str | None = None) -> object:
    """Build the instance for `node`, and for everything under it.

    Every value is checked by pydantic against the type of its field, and
    then the constructors run, which runs every `__post_init__`. A
    `TypedDict` node becomes a plain dict. All the
    errors in the tree are collected and reported together.

    With `construct=False` the constructors are skipped, and the result is
    `None`. `skip` names a field of `node` to leave out.
    """
    errors: list[str] = []
    result = _build(node, errors, construct=construct, skip=skip)
    if errors:
        raise ConfigError("Error in the configuration:\n" + "\n".join(errors))
    return None if result is _FAILED else result
