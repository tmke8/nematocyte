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
    TypeIs,
    Union,
    cast,
    get_args,
    get_origin,
)

# Unlike `typing.is_typeddict`, this also recognises `typing_extensions.TypedDict`,
# which is a class of its own, for `closed=True` and other newer features.
from typing_extensions import TypeForm, is_typeddict
import warnings

from pydantic import (
    Discriminator,
    PydanticSchemaGenerationError,
    PydanticUserError,
    Tag,
    TypeAdapter,
)

from ._errors import SchemaError
from ._types import type_text, unwrap

__all__ = [
    "RESERVED_KEY",
    "FieldInfo",
    "Kind",
    "class_adapter",
    "class_doc",
    "field_adapter",
    "class_fields",
    "config_class",
    "first_paragraph",
    "is_config_class",
    "is_dict_or_frozendict",
    "is_exact_dict_or_frozendict",
    "is_typeddict_class",
    "pick",
    "reachable_classes",
    "typeddict_option",
    "validate_schema",
]

RESERVED_KEY = "nematocyte"
"""The top-level key under which nematocyte's own settings live."""

SELF_KEY = "_self_"
DEFAULTS_KEY = "_defaults_"
OPTION_KEY = "_option_"
DOC_KEY = "_doc_"
"""The top-level key of a config file that describes the file."""
REPLACE_KEY = "_replace_"
"""The key of a mapping for a `TypedDict` that replaces what was set before."""
SOME = "_some_"
"""The name that switches an optional subconfig on, as a config name is given.

It is not the name of a config: no option and no config file can have it.
"""

_OPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")


def pick(value: Any) -> str | None:
    """Pick the option of a group of dataclasses, for pydantic.

    A group is a union whose members carry a pydantic ``Tag``, with this
    function as the ``Discriminator``::

        model: Annotated[
            Annotated[Fcn, Tag("fcn")] | Annotated[Cnn, Tag("cnn")],
            Discriminator(nematocyte.pick),
        ]

    The option is the ``_option_`` key of the mapping for the group, as
    nematocyte passes it to pydantic and as ``nematocyte.to_dict`` writes it.
    """
    if isinstance(value, Mapping):
        option = cast("Mapping[str, Any]", value).get(OPTION_KEY)
        if isinstance(option, str):
            return option
    return None


def _check_option_name(name: object, where: str) -> str:
    if not isinstance(name, str) or _OPTION_NAME.fullmatch(name) is None:
        raise SchemaError(
            f"{where}: {name!r} is not a valid name for an option. A name starts "
            "with a letter or an underscore, followed by letters, digits, '_' or '-'."
        )
    if name == SOME:
        raise SchemaError(
            f"{where}: {name!r} is not a valid name for an option: {SOME} "
            "switches an optional subconfig on, so it is never the name of a "
            "config."
        )
    return name


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
    tagged: bool = False
    """For a group: whether the union has a `Discriminator`.

    pydantic then puts the option into the location of an error, after the
    name of the field.
    """
    field: dataclasses.Field[Any] | None = None
    initvar: bool = False
    required: bool = True
    """False for a `NotRequired` key of a `TypedDict`, which may be left out."""
    optional: bool = False
    """Whether the type is `X | None`, so that the subconfig or group can be off."""

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


def is_exact_dict_or_frozendict(value: object) -> TypeIs[Mapping[Any, Any]]:
    """Whether `value` is a dict or a `frozendict`, and not of a subclass."""
    return type(value) in _DICT_TYPES


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


def _members(tp: Any) -> Iterator[tuple[Any, list[Any]]]:
    """The members of a union, each with its metadata; a non-union is one member."""
    inner, metadata = unwrap(tp)
    if get_origin(inner) in (Union, types.UnionType):
        for arg in get_args(inner):
            yield from _members(arg)
    else:
        yield inner, metadata


def _type_name(tp: Any) -> str:
    return tp.__name__ if isinstance(tp, type) else str(tp)


def _type_hints(cls: type) -> dict[str, Any]:
    try:
        return typing.get_type_hints(cls, include_extras=True)
    except Exception as exc:
        raise SchemaError(
            f"The type annotations of {cls.__name__} cannot be resolved: {exc}"
        ) from exc


def typeddict_option(cls: type) -> str | None:
    """The option name that a `TypedDict` declares in its `_option_` key.

    `None` if it has no such key. The key has to be a required
    ``Literal["name"]`` with exactly one string.
    """
    hints = _type_hints(cls)
    if OPTION_KEY not in hints:
        return None
    where = f"{cls.__name__}.{OPTION_KEY}"
    tp, _ = unwrap(_strip_qualifiers(hints[OPTION_KEY]))
    args = get_args(tp)
    if get_origin(tp) is not Literal or len(args) != 1 or not isinstance(args[0], str):
        raise SchemaError(
            f"{where}: the key {OPTION_KEY!r} of a TypedDict names its option, and "
            f'has to be a Literal of one string, such as Literal["name"].'
        )
    if OPTION_KEY not in cast("Any", cls).__required_keys__:
        raise SchemaError(f"{where}: the key {OPTION_KEY!r} cannot be NotRequired.")
    return _check_option_name(args[0], where)


_GROUP_SPELLING = (
    'Annotated[Annotated[A, Tag("a")] | Annotated[B, Tag("b")], '
    "Discriminator(nematocyte.pick)]"
)
_TYPEDDICT_GROUP_SPELLING = 'Annotated[A | B, Discriminator("_option_")]'


def _option_name(member: type, tags: list[Tag], where: str) -> str:
    """The name of a member of a group: from its `Tag`, or a TypedDict's key."""
    if is_typeddict_class(member):
        name = typeddict_option(member)
        if name is None:
            raise SchemaError(
                f"{where}: the TypedDict {member.__name__} is a member of a "
                f"group, and so has to name its option in the key "
                f'{OPTION_KEY}: Literal["..."].'
            )
        return name
    if len(tags) != 1:
        problem = "has no name" if not tags else "has more than one name"
        raise SchemaError(
            f"{where}: the member {member.__name__} of the group {problem}. "
            f"Every member needs one Tag, and the union needs nematocyte.pick: "
            f"{_GROUP_SPELLING}."
        )
    return _check_option_name(tags[0].tag, where)


def _check_member_extra(member: type, where: str) -> None:
    """Refuse a dataclass option whose pydantic config handles unknown keys.

    nematocyte passes the option of a group to pydantic as the key
    `_option_` next to the fields, and pydantic drops it as an unknown key.
    """
    config = getattr(member, "__pydantic_config__", None) or {}
    extra = config.get("extra")
    if extra in ("forbid", "allow"):
        raise SchemaError(
            f"{where}: the option {member.__name__} of the group sets "
            f"extra={extra!r} in its pydantic config. nematocyte passes the "
            f"option of a group to pydantic under the key {OPTION_KEY!r}, which "
            f"the class has to ignore, so it cannot set 'extra' for itself."
        )


def _check_discriminator(
    metadata: list[Any], *, typeddicts: bool, optional: bool, where: str
) -> None:
    """Check that the union of a group has the discriminator that it needs."""
    found = [m for m in metadata if isinstance(m, Discriminator)]
    if len(found) == 1 and (
        found[0].discriminator == OPTION_KEY
        if typeddicts
        else found[0].discriminator is pick
    ):
        return
    spelling = _TYPEDDICT_GROUP_SPELLING if typeddicts else _GROUP_SPELLING
    if optional:
        spelling += " | None"
    raise SchemaError(
        f"{where}: the union of a group needs exactly one Discriminator, as in "
        f"{spelling}."
    )


def _group_members(annotation: Any) -> tuple[list[tuple[Any, list[Any]]], list[Any]]:
    """The members of a union with their metadata, and the metadata of the union."""
    inner, outer = unwrap(annotation)
    if get_origin(inner) in (Union, types.UnionType):
        return list(_members(inner)), outer
    # A group with one option: its Tag and the Discriminator are both in the
    # metadata that was unwrapped.
    return [(inner, outer)], outer


@dataclass(frozen=True)
class _Analysis:
    kind: Kind
    options: dict[str, type] = dataclasses.field(default_factory=dict)
    config_cls: type | None = None
    tagged: bool = False
    optional: bool = False


def _without_none(annotation: Any) -> tuple[Any, bool]:
    """The annotation without a `None` in its outermost union, and whether it had one.

    `X | None` is an optional subconfig, and `Annotated[A | B, Discriminator(...)]
    | None` an optional group. A `None` inside the union that carries the
    `Discriminator` is not taken off, and is an error later.
    """
    inner, outer = unwrap(annotation)
    if get_origin(inner) not in (Union, types.UnionType) or any(
        isinstance(m, Discriminator) for m in outer
    ):
        return annotation, False
    args = get_args(inner)
    rest = tuple(arg for arg in args if arg is not type(None))
    if len(rest) == len(args):
        return annotation, False
    if len(rest) == 1:
        return rest[0], True
    return Union[rest], True  # noqa: UP007


def _analyse(annotation: Any, where: str) -> _Analysis:
    members, _ = _group_members(annotation)
    if not any(is_config_class(member) for member, _ in members):
        return _Analysis(Kind.VALUE)
    present, optional = _without_none(annotation)
    analysis = _analyse_present(present, where, optional=optional)
    if optional and analysis.kind is Kind.GROUP and len(analysis.options) == 1:
        # A name that switches on the only class: `_some_` does that for an
        # optional subconfig.
        (name, cls), *_ = analysis.options.items()
        typeddict = (
            f", where {cls.__name__} is a TypedDict without the key {OPTION_KEY!r}"
            if is_typeddict_class(cls)
            else ""
        )
        raise SchemaError(
            f"{where}: None is not allowed next to a group with one option "
            f"({name!r}). An optional group needs two or more options. For one "
            f"class, an optional subconfig is typed {cls.__name__} | None"
            f"{typeddict}, and {SOME} switches it on."
        )
    return dataclasses.replace(analysis, optional=optional)


def _analyse_present(annotation: Any, where: str, *, optional: bool) -> _Analysis:
    """Analyse a subconfig or a group, without the `None` of an optional one."""
    members, outer = _group_members(annotation)
    tags = [[m for m in metadata if isinstance(m, Tag)] for _, metadata in members]
    for (member, _), member_tags in zip(members, tags, strict=True):
        if is_typeddict_class(member) and member_tags:
            raise SchemaError(
                f"{where}: a Tag is not used for a TypedDict "
                f"({member.__name__}). A TypedDict names its option in its own "
                f'key instead: {OPTION_KEY}: Literal["..."].'
            )
    discriminated = any(isinstance(m, Discriminator) for m in outer)
    if len(members) == 1 and not tags[0] and not discriminated:
        config_cls = members[0][0]
        if is_config_class(config_cls) and (
            not is_typeddict_class(config_cls) or typeddict_option(config_cls) is None
        ):
            return _Analysis(Kind.CONFIG, config_cls=config_cls)

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

    spelling = _TYPEDDICT_GROUP_SPELLING if typeddicts else _GROUP_SPELLING
    options: dict[str, type] = {}
    for (member, _), member_tags in zip(members, tags, strict=True):
        if member is type(None):
            raise SchemaError(
                f"{where}: None is not allowed inside the union of a group. An "
                f"optional group has None outside the union that carries the "
                f"Discriminator: {spelling} | None."
            )
        if not is_config_class(member):
            raise SchemaError(
                f"{where}: a union with a dataclass or a TypedDict in it is a "
                f"config group, and all of its members have to be dataclasses "
                f"or TypedDicts; {_type_name(member)} is not."
            )
        name = _option_name(member, member_tags, where)
        if name in options:
            raise SchemaError(
                f"{where}: the option name {name!r} is used for both "
                f"{options[name].__name__} and {member.__name__}."
            )
        if member in options.values():
            raise SchemaError(
                f"{where}: the class {member.__name__} is listed twice in the group."
            )
        if not typeddicts:
            _check_member_extra(member, where)
        options[name] = member
    if len(options) > 1:
        # pydantic needs no discriminator to pick the only option.
        _check_discriminator(
            outer, typeddicts=bool(typeddicts), optional=optional, where=where
        )
    return _Analysis(Kind.GROUP, options=options, tagged=discriminated)


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
    hints = _type_hints(cls)

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
    if name in (SELF_KEY, DEFAULTS_KEY, OPTION_KEY, DOC_KEY, REPLACE_KEY):
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
        analysis = _analyse(hint, where)
        result[name] = FieldInfo(
            name=name,
            kind=analysis.kind,
            annotation=hint,
            owner=cls,
            options=analysis.options,
            config_cls=analysis.config_cls,
            tagged=analysis.tagged,
            field=field,
            initvar=initvar,
            optional=analysis.optional,
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
        analysis = _analyse(hint, where)
        required = name in required_keys
        if not required and analysis.kind is not Kind.VALUE:
            raise SchemaError(
                f"{where}: a subconfig or a group cannot be NotRequired; only a "
                "value can be left out. A subconfig or a group that can be "
                "switched off is typed X | None."
            )
        result[name] = FieldInfo(
            name=name,
            kind=analysis.kind,
            annotation=hint,
            owner=cls,
            options=analysis.options,
            config_cls=analysis.config_cls,
            tagged=analysis.tagged,
            required=required,
            optional=analysis.optional,
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
    for current in reachable_classes(cls):
        for info in class_fields(current).values():
            _check_option_directories(info, f"{current.__name__}.{info.name}")
    class_adapter(cls)


def _check_option_directories(info: FieldInfo, where: str) -> None:
    """No option of a group has a field named after an option.

    The directory of an option, in the group's directory, has the name of the
    option, and a directory named after a field is the field's: for its named
    configs, or, for a value, an error.
    """
    for option_name, option_cls in info.options.items():
        for name in class_fields(option_cls):
            if name in info.options:
                raise SchemaError(
                    f"{where}: {name!r} is an option of the group, and also a "
                    f"field of its option {option_name!r} "
                    f"({option_cls.__name__}.{name}). The directory {name}/ in "
                    f"the group's directory would then be both the option's "
                    f"and the field's. Rename the option or the field."
                )


def reachable_classes(cls: type) -> Iterator[type]:
    """`cls` and every config class under it, each checked by `class_fields`."""
    seen: set[type] = set()
    todo = [cls]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        yield current
        for info in class_fields(current).values():
            if info.config_cls is not None:
                todo.append(info.config_cls)
            todo.extend(info.options.values())


_CLASS_ADAPTERS: dict[type, TypeAdapter[Any]] = {}


def class_adapter(cls: type) -> TypeAdapter[Any]:
    """The adapter for a whole config class, with everything under it.

    A field whose type pydantic cannot read is a `SchemaError`.
    """
    cached = _CLASS_ADAPTERS.get(cls)
    if cached is None:
        try:
            with warnings.catch_warnings():
                # pydantic warns that it does not enforce `ReadOnly` in a
                # TypedDict, which nematocyte ignores anyway.
                warnings.filterwarnings("ignore", "Item .* `ReadOnly` qualifier")
                cached = TypeAdapter(cls)
        except (PydanticSchemaGenerationError, PydanticUserError) as exc:
            raise SchemaError(_unreadable(cls, exc)) from exc
        _CLASS_ADAPTERS[cls] = cached
    return cached


def _adapter_for(annotation: Any, owner: type | None) -> TypeAdapter[Any] | None:
    """An adapter for the type of a field, with the pydantic config of its class.

    `None` if pydantic cannot read the type.
    """
    config = getattr(owner, "__pydantic_config__", None)
    for attempt in (config, None) if config is not None else (None,):
        try:
            return TypeAdapter(annotation, config=attempt)
        except PydanticUserError:
            # A type with a config of its own, such as a dataclass, refuses
            # another one, and is tried without; a type that pydantic cannot
            # read fails either way.
            continue
    return None


_FIELD_ADAPTERS: dict[tuple[type | None, str], TypeAdapter[Any] | None] = {}


def field_adapter(info: FieldInfo) -> TypeAdapter[Any] | None:
    """The adapter for the type of one field; `None` if pydantic cannot read it."""
    key = (info.owner, info.name)
    if key not in _FIELD_ADAPTERS:
        _FIELD_ADAPTERS[key] = _adapter_for(info.annotation, info.owner)
    return _FIELD_ADAPTERS[key]


def _unreadable(cls: type, exc: Exception) -> str:
    """The error for a config class that pydantic cannot read, naming the field."""
    for config_cls in reachable_classes(cls):
        for info in class_fields(config_cls).values():
            if info.kind is Kind.VALUE and field_adapter(info) is None:
                return (
                    f"{config_cls.__name__}.{info.name}: pydantic cannot handle "
                    f"the type {type_text(info.annotation)}, so a config file "
                    f"could not set the field. A value that is made from the "
                    f"config, such as a torch.device, can be a property of the "
                    f"class instead."
                )
        for name, annotation in _hidden_fields(config_cls):
            if _adapter_for(annotation, config_cls) is None:
                return (
                    f"{config_cls.__name__}.{name}: pydantic cannot handle the "
                    f"type {type_text(annotation)}. It reads a field with "
                    f"init=False too, if the field has a default. Leave out the "
                    f"default and set the field in __post_init__, or make the "
                    f"value a property of the class."
                )
    return f"{cls.__name__}: pydantic cannot handle the class:\n{exc}"


def _hidden_fields(cls: type) -> Iterator[tuple[str, Any]]:
    """The fields with `init=False` and a default, which pydantic reads too."""
    if not dataclasses.is_dataclass(cls):
        return
    hints = typing.get_type_hints(cls, include_extras=True)
    for field in dataclasses.fields(cls):
        has_default = (
            field.default is not MISSING or field.default_factory is not MISSING
        )
        if not field.init and has_default:
            yield field.name, hints.get(field.name, field.type)


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
