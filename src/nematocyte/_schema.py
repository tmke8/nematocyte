"""Reading the schema off the config classes: fields, groups and docs.

A config class is a dataclass or a `TypedDict`.
"""

from collections.abc import Iterator, Mapping
import dataclasses
from dataclasses import MISSING, dataclass
import enum
import inspect
import re
import sys
import types
import typing
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    NotRequired,
    ReadOnly,
    Required,
    TypeAliasType,
    TypeIs,
    Union,
    cast,
    get_args,
    get_origin,
)

# Unlike `typing.is_typeddict`, this also recognises `typing_extensions.TypedDict`,
# which is a class of its own, for `closed=True` and other newer features.
from typing_extensions import TypeForm, is_typeddict

from ._errors import SchemaError

__all__ = [
    "RESERVED_KEY",
    "FieldInfo",
    "Kind",
    "Option",
    "class_doc",
    "class_fields",
    "config_class",
    "first_paragraph",
    "is_config_class",
    "is_dict_or_frozendict",
    "is_typeddict_class",
    "option",
    "typeddict_option",
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
    """What a field of a config class holds."""

    VALUE = "value"
    """A value, which is checked by pydantic."""
    CONFIG = "config"
    """A subconfig: one config class."""
    GROUP = "group"
    """A group: one of several config classes, selected by name."""


@dataclass(frozen=True)
class FieldInfo:
    """One config key of a config class."""

    name: str
    kind: Kind
    annotation: Any
    """The type of the field, as written; for a value, this is given to pydantic.

    For a key of a `TypedDict`, `Required`, `NotRequired` and `ReadOnly` are
    stripped.
    """
    owner: type | None
    """The config class that the field belongs to."""
    options: Mapping[str, type] = dataclasses.field(default_factory=dict)
    """For a group: the classes of the options, by name."""
    config_cls: type | None = None
    """For a subconfig: its class."""
    field: dataclasses.Field[Any] | None = None
    initvar: bool = False
    required: bool = True
    """False for a `NotRequired` key of a `TypedDict`, which may be left out."""

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


def is_typeddict_class(tp: object) -> bool:
    return isinstance(tp, type) and is_typeddict(tp)


# A `frozendict` (Python 3.15) is hashable, so a dataclass takes it as the
# default of a TypedDict field without `default_factory`; that is experimental.
if sys.version_info >= (3, 15):
    from builtins import frozendict

    _DICT_TYPES: tuple[type, ...] = (dict, frozendict)
else:
    _DICT_TYPES: tuple[type, ...] = (dict,)


def is_dict_or_frozendict(value: object) -> TypeIs[Mapping[Any, Any]]:
    """Whether `value` is a dict or a `frozendict`.

    Either can be the default of a TypedDict node.
    """
    return isinstance(value, _DICT_TYPES)


def is_config_class(tp: object) -> bool:
    """Whether `tp` is a dataclass or a `TypedDict`."""
    return isinstance(tp, type) and (dataclasses.is_dataclass(tp) or is_typeddict(tp))


def config_class(config_cls: TypeForm[Any]) -> type:
    """Check that `config_cls` is a dataclass or a `TypedDict`, and return it."""
    if not is_config_class(config_cls):
        raise SchemaError(
            f"The config class has to be a dataclass or a TypedDict, "
            f"not {config_cls!r}."
        )
    return cast("type", config_cls)


_QUALIFIERS = (Required, NotRequired, ReadOnly)


def _strip_qualifiers(tp: Any) -> Any:
    """Strip `Required`, `NotRequired` and `ReadOnly` from a key of a `TypedDict`.

    They can be nested in each other and in `Annotated`; `Annotated` is kept.
    """
    if get_origin(tp) in _QUALIFIERS:
        return _strip_qualifiers(get_args(tp)[0])
    if get_origin(tp) is Annotated:
        inner = _strip_qualifiers(tp.__origin__)
        if inner is not tp.__origin__:
            return Annotated[inner, *tp.__metadata__]
    return tp


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


def typeddict_option(cls: type) -> str | None:
    """The option name that a `TypedDict` declares in its `_option_` key.

    `None` if it has no such key. The key has to be a required
    ``Literal["name"]`` with exactly one string.
    """
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception as exc:
        raise SchemaError(
            f"The type annotations of {cls.__name__} cannot be resolved: {exc}"
        ) from exc
    if OPTION_KEY not in hints:
        return None
    where = f"{cls.__name__}.{OPTION_KEY}"
    tp, _ = _unwrap(_strip_qualifiers(hints[OPTION_KEY]))
    args = get_args(tp)
    if get_origin(tp) is not Literal or len(args) != 1 or not isinstance(args[0], str):
        raise SchemaError(
            f"{where}: the key {OPTION_KEY!r} of a TypedDict names its option, and "
            f'has to be a Literal of one string, such as Literal["name"].'
        )
    if OPTION_KEY not in cast("Any", cls).__required_keys__:
        raise SchemaError(f"{where}: the key {OPTION_KEY!r} cannot be NotRequired.")
    try:
        return option(args[0]).name
    except SchemaError as exc:
        raise SchemaError(f"{where}: {exc}") from None


def _option_name(member: type, member_names: list[Option], where: str) -> str:
    """The name of a member of a group: from `option()`, or a TypedDict's key."""
    if is_typeddict_class(member):
        name = typeddict_option(member)
        if name is None:
            raise SchemaError(
                f"{where}: the TypedDict {member.__name__} is a member of a "
                f"group, and so has to name its option in the key "
                f'{OPTION_KEY}: Literal["..."].'
            )
        return name
    if len(member_names) != 1:
        problem = "has no name" if not member_names else "has more than one name"
        raise SchemaError(
            f"{where}: the member {member.__name__} of the group {problem}. "
            f"Every member needs one name: "
            f'Annotated[{member.__name__}, nematocyte.option("...")].'
        )
    if OPTION_KEY in class_fields(member):
        raise SchemaError(
            f"{member.__name__} is an option of a group ({where}) and so "
            f"cannot have a field called {OPTION_KEY!r}."
        )
    return member_names[0].name


def _analyse(annotation: Any, where: str) -> tuple[Kind, dict[str, type], type | None]:
    members = list(_members(annotation))
    if not any(is_config_class(member) for member, _ in members):
        return Kind.VALUE, {}, None

    names = [[m for m in metadata if isinstance(m, Option)] for _, metadata in members]
    for (member, _), member_names in zip(members, names, strict=True):
        if is_typeddict_class(member) and member_names:
            raise SchemaError(
                f"{where}: nematocyte.option() is not used for a TypedDict "
                f"({member.__name__}). A TypedDict names its option in its own "
                f'key instead: {OPTION_KEY}: Literal["..."].'
            )
    if len(members) == 1 and not names[0]:
        config_cls = members[0][0]
        assert isinstance(config_cls, type)
        if not is_typeddict_class(config_cls) or typeddict_option(config_cls) is None:
            return Kind.CONFIG, {}, config_cls

    typeddicts = [m for m, _ in members if is_typeddict_class(m)]
    dataclasses_ = [
        m for m, _ in members if is_config_class(m) and not is_typeddict_class(m)
    ]
    if typeddicts and dataclasses_:
        raise SchemaError(
            f"{where}: the options of a group are either all dataclasses or all "
            f"TypedDicts; {dataclasses_[0].__name__} is a dataclass and "
            f"{typeddicts[0].__name__} is a TypedDict."
        )

    options: dict[str, type] = {}
    for (member, _), member_names in zip(members, names, strict=True):
        if member is type(None):
            raise SchemaError(
                f"{where}: None is not allowed as a member of a group, and a "
                "subconfig cannot be optional."
            )
        if not is_config_class(member):
            raise SchemaError(
                f"{where}: a union with a dataclass or a TypedDict in it is a "
                f"config group, and all of its members have to be dataclasses "
                f"or TypedDicts; {_type_name(member)} is not."
            )
        name = _option_name(member, member_names, where)
        if name in options:
            raise SchemaError(
                f"{where}: the option name {name!r} is used for both "
                f"{options[name].__name__} and {member.__name__}."
            )
        if member in options.values():
            raise SchemaError(
                f"{where}: the class {member.__name__} is listed twice in the group."
            )
        options[name] = member
    return Kind.GROUP, options, None


_FIELDS_CACHE: dict[type, dict[str, FieldInfo]] = {}


def class_fields(cls: type) -> dict[str, FieldInfo]:
    """The config keys of a config class, in the order of its fields.

    For a dataclass, these are the arguments of its constructor: the fields
    with `init=True`, and the `InitVar` pseudo-fields. For a `TypedDict`,
    they are its keys.
    """
    cached = _FIELDS_CACHE.get(cls)
    if cached is not None:
        return cached
    if not is_config_class(cls):
        raise SchemaError(f"{_type_name(cls)} is not a dataclass or a TypedDict.")
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
        if is_typeddict(cls):
            _typeddict_fields(cls, hints, result)
        else:
            _dataclass_fields(cls, hints, result)
    except BaseException:
        del _FIELDS_CACHE[cls]
        raise
    return result


def _check_name(name: str, where: str) -> None:
    if name in (SELF_KEY, DEFAULTS_KEY):
        raise SchemaError(
            f"{where}: {name!r} is reserved and cannot be the name of a field."
        )


def _dataclass_fields(
    cls: type, hints: dict[str, Any], result: dict[str, FieldInfo]
) -> None:
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
        _check_name(name, where)
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


def _typeddict_fields(
    cls: type, hints: dict[str, Any], result: dict[str, FieldInfo]
) -> None:
    required_keys = frozenset[str](cast("Any", cls).__required_keys__)
    for name, raw_hint in hints.items():
        if name == OPTION_KEY:
            # Not a config key: nematocyte fills it in with the option's name.
            continue
        hint = _strip_qualifiers(raw_hint)
        where = f"{cls.__name__}.{name}"
        _check_name(name, where)
        kind, options, config_cls = _analyse(hint, where)
        required = name in required_keys
        if not required and kind is not Kind.VALUE:
            raise SchemaError(
                f"{where}: a subconfig or a group cannot be NotRequired; only a "
                "value can be left out."
            )
        result[name] = FieldInfo(
            name=name,
            kind=kind,
            annotation=hint,
            owner=cls,
            options=options,
            config_cls=config_cls,
            required=required,
        )


def validate_schema(cls: type, *, root: bool = True) -> None:
    """Check `cls` and every config class that can be reached from it."""
    if root and is_typeddict_class(cls) and typeddict_option(cls) is not None:
        raise SchemaError(
            f"{cls.__name__}.{OPTION_KEY}: the root config class is not an option "
            f"of a group, and cannot have the key {OPTION_KEY!r}."
        )
    if root and RESERVED_KEY in class_fields(cls):
        raise SchemaError(
            f"{cls.__name__}.{RESERVED_KEY}: {RESERVED_KEY!r} is reserved for "
            "nematocyte's own settings and cannot be the name of a field of the "
            "root config class."
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
