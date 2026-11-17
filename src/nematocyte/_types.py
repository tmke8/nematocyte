"""How the type of a value is shown, in `--help` and in error messages."""

import enum
import pathlib
import types
from typing import Annotated, Any, Literal, TypeAliasType, Union, get_args, get_origin

from ._overrides import format_value

__all__ = ["type_text"]


def _class_text(tp: type) -> str:
    if issubclass(tp, enum.Enum):
        return "|".join(format_value(member.value) for member in tp)
    if issubclass(tp, pathlib.PurePath):
        return "path"
    return tp.__name__


def type_text(tp: Any) -> str:
    """The type of a value, the way `--help` shows it."""
    while True:
        if isinstance(tp, TypeAliasType):
            tp = tp.__value__
        elif get_origin(tp) is Annotated:
            tp = tp.__origin__
        else:
            break
    origin = get_origin(tp)
    if tp is None or tp is type(None):
        text = "null"
    elif tp is Any:
        text = "any"
    elif origin is Literal:
        text = "|".join(format_value(value) for value in get_args(tp))
    elif origin in (Union, types.UnionType):
        text = " | ".join(type_text(arg) for arg in get_args(tp))
    elif origin is not None:
        name = str(getattr(origin, "__name__", origin)).lower()
        args = ", ".join(
            "..." if arg is ... else type_text(arg) for arg in get_args(tp)
        )
        text = f"{name}[{args}]" if args else name
    elif isinstance(tp, type):
        text = _class_text(tp)
    else:
        text = str(tp)
    return text
