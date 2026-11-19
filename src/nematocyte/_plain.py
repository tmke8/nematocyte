"""Turning values into what a config file would hold."""

import dataclasses
import enum
from pathlib import PurePath
from typing import Any

from ._errors import ConfigError
from ._schema import (
    OPTION_KEY,
    Kind,
    class_fields,
    is_config_class,
    is_dict_or_frozendict,
    is_typeddict_class,
)

__all__ = ["to_dict", "to_plain"]


def _plain_set(value: set[Any] | frozenset[Any]) -> list[Any]:
    items = [to_plain(item) for item in value]
    try:
        return sorted(items)
    except TypeError:
        return items


def to_plain(value: Any) -> Any:
    """Convert a value to plain YAML types.

    A path becomes a string, a tuple or a set becomes a list, a `frozendict`
    becomes a dict, and an enum member becomes its value. Everything else is
    left as it is.
    """
    result = value
    if isinstance(value, enum.Enum):
        result = to_plain(value.value)
    elif isinstance(value, PurePath):
        result = str(value)
    elif is_dict_or_frozendict(value):
        result = {to_plain(key): to_plain(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [to_plain(item) for item in value]
    elif isinstance(value, (set, frozenset)):
        result = _plain_set(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        # A dataclass in the position of a value, such as an item of a list.
        result = {
            field.name: to_plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.init
        }
    return result


def _check_flat_typeddicts(cls: type) -> None:
    """Refuse a TypedDict that holds a subconfig or a group.

    Checked on the schema, before anything is converted.
    """
    seen: set[type] = set()
    todo = [cls]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        for info in class_fields(current).values():
            if is_typeddict_class(current) and info.kind is not Kind.VALUE:
                raise TypeError(
                    f"to_dict() needs every TypedDict in the config to hold only "
                    f"values, but {current.__name__}.{info.name} is a "
                    f"{'group' if info.kind is Kind.GROUP else 'subconfig'}."
                )
            if info.config_cls is not None:
                todo.append(info.config_cls)
            todo.extend(info.options.values())


def to_dict(config: object) -> dict[str, Any]:
    """Return the config as plain dicts, lists and scalars, for logging.

    The dict for a group field records the option under the key
    ``_option_``. Paths become strings, tuples and sets become lists, and
    enum members become their values, so that the result can be dumped as
    YAML or JSON.

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
        if info.kind is Kind.VALUE or isinstance(value, dict):
            # A value, or a TypedDict, which holds only values.
            result[info.name] = to_plain(value)
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
