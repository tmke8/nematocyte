"""Reading the schema off the dataclasses: fields, groups and docs."""

from collections.abc import Iterator, Mapping
import dataclasses
from dataclasses import MISSING, dataclass
import enum
import inspect
import re
import types
import typing
from typing import (
    Annotated,
    Any,
    ClassVar,
    TypeAliasType,
    Union,
    cast,
    get_args,
    get_origin,
)

from ._errors import SchemaError

__all__ = [
    "RESERVED_KEY",
    "FieldInfo",
    "Kind",
    "Option",
    "class_doc",
    "class_fields",
    "first_paragraph",
    "option",
    "validate_schema",
]

RESERVED_KEY = "nematocyte"
"""The top-level key under which nematocyte's own settings live."""

SELF_KEY = "_self_"
DEFAULTS_KEY = "defaults"
OPTION_KEY = "_option_"

_OPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")


@dataclass(frozen=True, slots=True)
class Option:
    """The name of a member of a group. Created with :func:`option`."""

    name: str


def option(name: str) -> Option:
    """Name a member of a config group.

    A group is a field whose type is a union of dataclasses, each of which is
    annotated with its name::

        model: Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")]
    """
    if not isinstance(name, str) or _OPTION_NAME.fullmatch(name) is None:
        raise SchemaError(
            f"{name!r} is not a valid name for an option. A name starts with a "
            "letter or an underscore, followed by letters, digits, '_' or '-'."
        )
    return Option(name)


class Kind(enum.Enum):
    """What a field of a config dataclass holds."""

    VALUE = "value"
    """A value, which is checked by pydantic."""
    CONFIG = "config"
    """A subconfig: one dataclass."""
    GROUP = "group"
    """A group: one of several dataclasses, selected by name."""


@dataclass(frozen=True)
class FieldInfo:
    """One config key of a dataclass."""

    name: str
    kind: Kind
    annotation: Any
    """The type of the field, as written; for a value, this is given to pydantic."""
    owner: type | None
    """The dataclass that the field belongs to."""
    options: Mapping[str, type] = dataclasses.field(default_factory=dict)
    """For a group: the classes of the options, by name."""
    config_cls: type | None = None
    """For a subconfig: its class."""
    field: dataclasses.Field[Any] | None = None
    initvar: bool = False

    def class_default(self) -> Any:
        """The default from the class definition, or ``MISSING``."""
        if self.field is None:
            return MISSING
        if self.field.default is not MISSING:
            return self.field.default
        if self.field.default_factory is not MISSING:
            return self.field.default_factory()
        return MISSING

    @property
    def has_class_default(self) -> bool:
        return self.field is not None and (
            self.field.default is not MISSING
            or self.field.default_factory is not MISSING
        )

    @property
    def doc(self) -> str:
        """The first paragraph of the `doc` of the field; empty if it has none.

        The doc is looked up along the MRO, so that it is not lost when a
        subclass declares the field again.
        """
        classes = self.owner.__mro__ if self.owner is not None else ()
        for klass in classes:
            fields = klass.__dict__.get("__dataclass_fields__")
            found = fields.get(self.name) if fields is not None else None
            doc = found.doc if found is not None else None
            if isinstance(doc, str) and doc.strip():
                return first_paragraph(doc)
        return ""

    def option_of(self, cls: type) -> str | None:
        """The name under which `cls` is an option of this group."""
        for name, option_cls in self.options.items():
            if option_cls is cls:
                return name
        return None


def is_config_class(tp: object) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _unwrap(tp: Any) -> tuple[Any, list[Any]]:
    """Strip PEP 695 aliases and `Annotated`, collecting the metadata."""
    metadata: list[Any] = []
    while True:
        if isinstance(tp, TypeAliasType):
            tp = tp.__value__
        elif get_origin(tp) is Annotated:
            metadata = [*tp.__metadata__, *metadata]
            tp = tp.__origin__
        else:
            return tp, metadata


def _members(tp: Any) -> Iterator[tuple[Any, list[Any]]]:
    """The members of a union, each with its metadata; a non-union is one member."""
    inner, metadata = _unwrap(tp)
    if get_origin(inner) in (Union, types.UnionType):
        for arg in get_args(inner):
            yield from _members(arg)
    else:
        yield inner, metadata


def _type_name(tp: Any) -> str:
    return tp.__name__ if isinstance(tp, type) else str(tp)


def _analyse(annotation: Any, where: str) -> tuple[Kind, dict[str, type], type | None]:
    members = list(_members(annotation))
    if not any(is_config_class(member) for member, _ in members):
        return Kind.VALUE, {}, None

    names = [[m for m in metadata if isinstance(m, Option)] for _, metadata in members]
    if len(members) == 1 and not names[0]:
        config_cls = members[0][0]
        assert isinstance(config_cls, type)
        return Kind.CONFIG, {}, config_cls

    options: dict[str, type] = {}
    for (member, _), member_names in zip(members, names, strict=True):
        if member is type(None):
            raise SchemaError(
                f"{where}: None is not allowed as a member of a group, and a "
                "dataclass field cannot be optional."
            )
        if not is_config_class(member):
            raise SchemaError(
                f"{where}: a union with a dataclass in it is a config group, and "
                f"all of its members have to be dataclasses; "
                f"{_type_name(member)} is not."
            )
        if len(member_names) != 1:
            problem = "has no name" if not member_names else "has more than one name"
            raise SchemaError(
                f"{where}: the member {member.__name__} of the group {problem}. "
                f"Every member needs one name: "
                f'Annotated[{member.__name__}, nematocyte.option("...")].'
            )
        name = member_names[0].name
        if name in options:
            raise SchemaError(
                f"{where}: the option name {name!r} is used for both "
                f"{options[name].__name__} and {member.__name__}."
            )
        if member in options.values():
            raise SchemaError(
                f"{where}: the class {member.__name__} is listed twice in the group."
            )
        if OPTION_KEY in class_fields(member):
            raise SchemaError(
                f"{member.__name__} is an option of a group ({where}) and so "
                f"cannot have a field called {OPTION_KEY!r}."
            )
        options[name] = member
    return Kind.GROUP, options, None


_FIELDS_CACHE: dict[type, dict[str, FieldInfo]] = {}


def class_fields(cls: type) -> dict[str, FieldInfo]:
    """The config keys of a dataclass, in the order of its fields.

    These are the arguments of its constructor: the fields with `init=True`,
    and the `InitVar` pseudo-fields.
    """
    cached = _FIELDS_CACHE.get(cls)
    if cached is not None:
        return cached
    if not is_config_class(cls):
        raise SchemaError(f"{_type_name(cls)} is not a dataclass.")
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception as exc:
        raise SchemaError(
            f"The type annotations of {cls.__name__} cannot be resolved: {exc}"
        ) from exc

    result: dict[str, FieldInfo] = {}
    # Registered before the fields are analysed, so that a class which refers
    # to itself does not recurse forever.
    _FIELDS_CACHE[cls] = result
    try:
        dataclass_fields: dict[str, dataclasses.Field[Any]] = cast(
            "Any", cls
        ).__dataclass_fields__
        for name, field in dataclass_fields.items():
            hint = hints.get(name, field.type)
            if hint is ClassVar or get_origin(hint) is ClassVar:
                continue
            initvar = isinstance(hint, dataclasses.InitVar)
            if initvar:
                hint = hint.type
            elif not field.init:
                continue
            where = f"{cls.__name__}.{name}"
            if name in (SELF_KEY, DEFAULTS_KEY):
                raise SchemaError(
                    f"{where}: {name!r} is reserved and cannot be the name of a field."
                )
            kind, options, config_cls = _analyse(hint, where)
            result[name] = FieldInfo(
                name=name,
                kind=kind,
                annotation=hint,
                owner=cls,
                options=options,
                config_cls=config_cls,
                field=field,
                initvar=initvar,
            )
    except BaseException:
        del _FIELDS_CACHE[cls]
        raise
    return result


def validate_schema(cls: type, *, root: bool = True) -> None:
    """Check `cls` and every config class that can be reached from it."""
    if root and RESERVED_KEY in class_fields(cls):
        raise SchemaError(
            f"{cls.__name__}.{RESERVED_KEY}: {RESERVED_KEY!r} is reserved for "
            "nematocyte's own settings and cannot be the name of a field of the "
            "root dataclass."
        )
    seen: set[type] = set()
    todo = [cls]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        for info in class_fields(current).values():
            if info.config_cls is not None:
                todo.append(info.config_cls)
            todo.extend(info.options.values())


def first_paragraph(text: str) -> str:
    """The first paragraph of a doc, on one line."""
    paragraph = inspect.cleandoc(text).split("\n\n", 1)[0]
    return " ".join(paragraph.split())


def class_doc(cls: type) -> str:
    """The first paragraph of the docstring of `cls`; empty if it wrote none.

    A dataclass without a docstring gets one generated from its signature,
    such as ``Name(x: int = 1)``. Those are ignored.
    """
    doc = cls.__dict__.get("__doc__")
    # Python 3.15 generates the docstring lazily, through an object that is
    # not a string; before that, it is a string that starts with the name.
    if not isinstance(doc, str) or doc.startswith(f"{cls.__name__}("):
        return ""
    return first_paragraph(doc)
