"""Turning values into what a JSON file would hold."""

from collections.abc import Collection, Iterable, Mapping
import dataclasses
import datetime
import enum
import math
from pathlib import PurePath
from typing import Any, cast

from ._errors import ConfigError
from ._schema import (
    OPTION_KEY,
    Kind,
    class_fields,
    is_config_class,
    is_dict_or_frozendict,
    is_exact_dict_or_frozendict,
    is_typeddict_class,
    reachable_classes,
)

__all__ = [
    "first_difference",
    "is_json_form",
    "to_dict",
    "to_json_form",
    "without_json_form",
]


def _json_set(value: set[Any] | frozenset[Any]) -> list[Any]:
    items = [to_json_form(item) for item in value]
    try:
        return sorted(items)
    except TypeError:
        return items


def _json_key(key: Any) -> Any:
    """A key in JSON form; one that would become a list, such as a tuple, is kept."""
    result = to_json_form(key)
    return key if isinstance(result, (list, dict)) else result


def to_json_form(value: Any) -> Any:
    """Convert a value to the types that JSON has.

    A path becomes a string, a date, a datetime or a time becomes its ISO
    string, a tuple or a set becomes a list, a `frozendict` becomes a dict,
    and an enum member becomes its value. Everything else is left as it is;
    a value that JSON has no form for stays as it is too, and is reported
    when the tree is built.
    """
    result = value
    if isinstance(value, enum.Enum):
        result = to_json_form(value.value)
    elif isinstance(value, PurePath):
        result = str(value)
    elif isinstance(value, _DATES):
        result = value.isoformat()
    elif is_dict_or_frozendict(value):
        result = {_json_key(key): to_json_form(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [to_json_form(item) for item in value]
    elif isinstance(value, (set, frozenset)):
        result = _json_set(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        # A dataclass in the position of a value, such as an item of a list.
        result = {
            field.name: to_json_form(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.init
        }
    return result


_CONTAINERS = (list, tuple, set, frozenset)
_DATES = (datetime.date, datetime.time)


def is_json_form(value: Any) -> bool:
    """Whether `to_json_form` changes at most the kinds of the containers in `value`.

    It mirrors the branches of `to_json_form`: an enum member, a path, a date,
    a datetime, a time, a dataclass instance and a subclass of a container,
    such as a `NamedTuple`, are converted; any other value is kept as it is,
    inside a list, tuple, set, frozenset, dict or frozendict. A set may come
    out in another order.
    """
    if is_exact_dict_or_frozendict(value):
        return all(
            is_json_form(key) and is_json_form(item) for key, item in value.items()
        )
    if type(value) in _CONTAINERS:
        return all(is_json_form(item) for item in value)
    converted = isinstance(value, (enum.Enum, PurePath, *_DATES, *_CONTAINERS))
    instance = dataclasses.is_dataclass(value) and not isinstance(value, type)
    return not (converted or instance or is_dict_or_frozendict(value))


_JSON_SCALARS = (str, int, float, bool, type(None))

type _Part = tuple[str, Any]


def without_json_form(value: Any, seen: frozenset[int] = frozenset()) -> _Part | None:
    """The first part of a value in JSON form that JSON has no form for.

    The location is `""` for the value itself. A container that holds
    itself has no JSON form either.
    """
    if isinstance(value, _JSON_SCALARS):
        return None
    if id(value) in seen or not isinstance(value, (dict, list)):
        return "", value
    inner = seen | {id(value)}
    items: Iterable[tuple[Any, Any]] = (
        cast("dict[Any, Any]", value).items()
        if isinstance(value, dict)
        else enumerate(cast("list[Any]", value))
    )
    for key, item in items:
        if not isinstance(key, _JSON_SCALARS):
            return "", key
        found = without_json_form(item, inner)
        if found is not None:
            location, part = found
            return (f"{key}.{location}" if location else str(key)), part
    return None


type _Pairs = list[tuple[str, Any, Any]]


def first_difference(expected: Any, actual: Any) -> tuple[str, Any, Any] | None:
    """The first place where `actual` differs from `expected`, with both values.

    A list, tuple, set and frozenset count as the same, as do a dict and a
    frozendict: a config file cannot tell them apart. The location is `""`
    for the value itself.
    """
    if expected is actual:
        return None
    pairs = _parts(expected, actual)
    if pairs is None:
        return "", expected, actual
    for location, item, out in pairs:
        found = first_difference(item, out)
        if found is not None:
            inner, before, after = found
            joined = f"{location}.{inner}" if location and inner else location or inner
            return joined, before, after
    return None


def _parts(expected: Any, actual: Any) -> _Pairs | None:
    """The parts of two values that are compared; `None` if they differ already."""
    if is_dict_or_frozendict(expected) and is_dict_or_frozendict(actual):
        return _mapping_parts(expected, actual)
    if isinstance(expected, _CONTAINERS) and isinstance(actual, _CONTAINERS):
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
                    if first_difference(item, out) is None
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


def _check_flat_typeddicts(cls: type) -> None:
    """Refuse a TypedDict that holds a subconfig or a group.

    Checked on the schema, before anything is converted.
    """
    for current in reachable_classes(cls):
        if not is_typeddict_class(current):
            continue
        for info in class_fields(current).values():
            if info.kind is not Kind.VALUE:
                raise TypeError(
                    f"to_dict() needs every TypedDict in the config to hold only "
                    f"values, but {current.__name__}.{info.name} is a "
                    f"{'group' if info.kind is Kind.GROUP else 'subconfig'}."
                )


def to_dict(config: object) -> dict[str, Any]:
    """Return the config as plain dicts, lists and scalars, for logging.

    The dict for a group field records the option under the key
    ``_option_``. Paths become strings, dates and times become ISO strings,
    tuples and sets become lists, and enum members become their values, so
    that the result can be dumped as YAML or JSON.

    The root has to be a dataclass. A TypedDict below it can be a subconfig
    or an option of a group, but may only hold values: it is kept as it is,
    apart from the conversion of its values. An option already holds its
    ``_option_``.
    """
    cls = type(config)
    if isinstance(config, dict):
        raise TypeError(
            "to_dict() does not support a TypedDict as the root of the config; "
            "it expects an instance of a config dataclass."
        )
    if isinstance(config, type) or not is_config_class(cls):
        raise TypeError(
            f"to_dict() expects an instance of a config dataclass, not {config!r}."
        )
    _check_flat_typeddicts(cls)
    return _to_dict(cls, config)


def _to_dict(cls: type, config: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for info in class_fields(cls).values():
        if info.initvar:
            # The value of an `InitVar` is not kept on the instance.
            continue
        value = getattr(config, info.name)
        if info.kind is Kind.VALUE or isinstance(value, dict) or value is None:
            # A value, a TypedDict, which holds only values, or an optional
            # subconfig that is off.
            result[info.name] = to_json_form(value)
        elif info.kind is Kind.CONFIG:
            result[info.name] = _to_dict(type(value), value)
        else:
            name = info.option_of(type(value))
            if name is None:
                name = next(
                    (
                        option_name
                        for option_name, option_cls in info.options.items()
                        if isinstance(value, option_cls)
                    ),
                    None,
                )
            if name is None:
                raise ConfigError(
                    f"{cls.__name__}.{info.name} holds an instance of "
                    f"{type(value).__name__}, which is not an option of the group. "
                    f"Options: {', '.join(info.options)}."
                )
            result[info.name] = {OPTION_KEY: name, **_to_dict(type(value), value)}
    return result
