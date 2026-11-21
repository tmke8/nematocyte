# Deserializer

How the composed config tree becomes dataclass instances.

## Replacing OmegaConf: cattrs or pydantic?

### Conclusion

Use **pydantic**, and hand it the whole tree in one call. nematocyte itself
owns the config tree (plain `dict`/`list` until the very end) and the choice
of the option of every group, which it attaches to the tree for pydantic to
read.

Neither library replaces OmegaConf on its own. OmegaConf is three things: a
mergeable tree, an interpolation engine, and a type validator. cattrs and
pydantic only cover the third. The first two are small and have to be written
either way (see [interpolation.md](interpolation.md) for the second), so the
choice comes down to which library is the better last step: "turn this fully
merged, fully resolved raw tree into dataclass instances, or tell the user
what is wrong with it".

Both can do that for every type used in `hydra-example`. pydantic wins because
its defaults are already what a config library wants, whereas cattrs has to be
corrected first. cattrs wins on one point, external customisation, and that
point has a workable answer in pydantic.

### What was tested

Throwaway prototypes against a torch-free copy of the `hydra-example`
dataclasses, with cattrs 26.2.1 and pydantic 2.14.3 on Python 3.12. Everything
in the table was observed, not taken from documentation.

| Requirement | cattrs | pydantic |
| --- | --- | --- |
| Unmodified stdlib dataclasses: `kw_only`, `_: KW_ONLY`, `frozen`, `slots`, inheritance, ABCs, `default_factory`, `Path`, `X \| None`, `dict[int, int]`, PEP 695 `type A = Literal[...]` | Works | Works through `TypeAdapter`; the classes are not touched and the results are ordinary instances |
| `__post_init__` checks | Run. The message is replaced by "invalid value" unless a custom error formatter is supplied | Run. Reported with path and the original message |
| `InitVar` fields | Not supported (rejected as an unknown key) | Supported |
| Scalar coercion out of the box | Unsafe: `1.5` becomes `1` for `int`, `"false"` becomes `True`, `None` becomes `"None"`, `"abc"` becomes `["a", "b", "c"]` for `list[str]`. Every primitive hook has to be replaced (about 30 lines) | Lax mode is close to ideal: `1.5` is rejected for `int`, `"false"` is `False`, `123` is rejected for `str`, and `"1e-3"` is accepted for `float` (so the `1.e-3` workaround for YAML 1.1 goes away). Warts: `True` is accepted as `1`, `"123"` as `123` |
| Unions of plain values: `int \| str`, `int \| list[int]`, `float \| Literal["auto"]` | Unsupported until `configure_union_passthrough` is applied; then values pass through on an `isinstance` match with no coercion, and errors are confusing | Work, with one error line per union member |
| Unknown keys | `Converter(forbid_extra_keys=True)` | `validate_python(..., extra="forbid")`, per call, pydantic 2.12 or later; applies to nested dataclasses too. `TypeAdapter(config=...)` is refused for dataclasses |
| Error reporting | All errors collected, with paths. Wording is generic ("invalid value for type, expected int") and needs a custom formatter | All errors collected, with paths, specific wording and the offending input |
| Value constraints, e.g. `Annotated[float, Field(ge=0, le=1)]` | Ignored | Enforced |
| Picking the concrete class for a field typed as an abstract base | One hook on the converter. The composer hands over a `dict` subclass that carries the chosen class; single pass, error paths intact, works at any depth | No external hook. Either the node is instantiated first and passed in as an instance (accepted for a base-typed field; an instance of the wrong class is rejected), or the field is typed as a tagged union with a callable `Discriminator`, which can read the choice from a `dict` subclass; single pass, error paths intact, with the tag in them |
| Changing how a type is handled without editing user code (enum by name, `torch.device`, ...) | Yes: hooks live on the converter and also apply inside containers | No: behaviour is attached to annotations or class config. An unknown class inside a dataclass is a schema error; `arbitrary_types_allowed` can only be passed for non-dataclass adapters and then only does an `isinstance` check |
| Enums | By value only | By value only |
| Dumping to a plain dict (for logging) | **Silently drops subclass fields** of a base-typed field unless a hook dispatches on the runtime type | **Silently drops subclass fields** unless `serialize_as_any=True` |
| JSON schema | None | Generated, including for stdlib dataclasses |
| Footprint | Pure Python, 0.6 MB including attrs. Import 48 ms; first use of 4 classes 1 ms | Compiled core, 6.5 MB. Import about 90 ms; building 4 adapters 27 ms |

Validation uses pydantic's **lax mode**. Strict mode on Python input
(`validate_python(value, strict=True)`) is not usable: it rejects `str` for
`Path`, `list` for `tuple` and `set`, a value for its enum member, and `dict`
for a dataclass. YAML can produce none of the types it insists on. A strict
variant that does work was considered and rejected; see
[Rejected ideas](#rejected-ideas).

### Why pydantic

- **Coercion is right by default.** With cattrs we would write and maintain
  the rules for every primitive, and the union story on top of that.
- **Error messages are the product.** For a config library, most of the
  perceived usability is what happens on a typo. pydantic's messages need
  light relabelling; cattrs's need to be written.
- **Constraints in the type** (`Annotated[float, Field(ge=0, le=1)]`) remove a
  lot of `__post_init__` boilerplate, and still work with plain dataclasses.
- **JSON schema** opens the door to editor completion for the YAML files,
  which Hydra cannot offer. It is not used to validate them; see
  [validation.md](validation.md).
- **`InitVar` works**, which matters if the library is meant to feel like
  "just dataclasses".

### What pydantic cannot do, and how to live with it

pydantic has no converter object that nematocyte could own and customise. Two
consequences:

1. **Config groups cannot be resolved by nematocyte inside pydantic.** There
   is no way to say "at path `model`, build a `SimpleCNNFactory`" without
   annotating the user's classes. So the user's annotation does it: a group
   is a tagged union with `nematocyte.pick` as its discriminator
   ([unions.md](unions.md)), and nematocyte hands each node of a group to
   pydantic as a `Selected`, a `dict` that carries the option that
   composition chose. `pick` returns that option, and pydantic builds the
   member with that `Tag`.
2. **Per-type behaviour cannot be changed globally**, e.g. accepting enum
   members by name as Hydra does, or converting a string to `torch.device`.
   A type that pydantic cannot read is an error in the schema; a value made
   from the config, such as a `torch.device`, belongs in a property of the
   class (as `Config.device` in the example).

With that, the whole tree goes to pydantic in one call:

```text
build(cls, node):
    data = node as plain dicts, each node of a group as Selected(option, fields)
    instance = TypeAdapter(cls).validate_python(data, extra="forbid")
        # checks every value and runs every constructor and __post_init__
    on errors: map each location back to its node and field, skipping the
               tag of a group, and report it with where the value was set
    check bools and defaults against instance
```

A field without a default has to be given a value, and a missing one is
reported with its path. Hydra's marker for a mandatory value, `???`, is not
implemented: a dataclass field without a default already says the same. In
a config file, `???` is an ordinary string.

nematocyte still walks the tree: to hand it over, to map pydantic's errors
back to the file or override that set a value and to the doc of the field,
and for the checks pydantic does not make. The same schema traversal is
needed anyway for `--help`, for checking that a CLI override names a real
key, and for interpolation ([interpolation.md](interpolation.md)).

Costs of this approach, to be aware of:

- A key missing from the config does not always fall back to the class
  default. Unless an option of a group is named for the field, the value
  comes from the field's default instance (see
  [unions.md](unions.md#defaults)). nematocyte fills in every value before
  the tree goes to pydantic.
- Group fields are only recognised as direct fields of a dataclass, not
  nested inside `list[...]` or `dict[...]`. Hydra has the same limitation.
- Enum by name needs a validator in the annotation, such as pydantic's
  `BeforeValidator`. Using `Literal` or `StrEnum` avoids the problem.
- pydantic's lax mode accepts `seed: true` as `1`. nematocyte rejects it: a
  bool that pydantic turned into a number is an error, also inside a list,
  tuple or dict. Where the type allows a bool (`int | bool`, `Any`), it is
  kept. The check compares the instance with the tree, so it is only made
  once pydantic accepts everything else.
- pydantic runs the constructors as it validates, so they cannot be
  skipped ([validation.md](validation.md#skipping-the-constructors)).
- A number is rejected for a `str` field, in config files and on the command
  line alike, and has to be quoted. See [cli.md](cli.md) for why this is not
  relaxed.

nematocyte used to call pydantic once per field instead; see
[Rejected ideas](#validating-each-field-on-its-own).

### When cattrs would be the better choice

If a small pure-Python dependency matters more than error quality, or if
user-registered converters for arbitrary types turn out to be a central
feature. The cattrs route is entirely viable: about 30 lines of strict
primitive hooks, one hook for group bases, one custom error formatter, and one
unstructure hook. Its group handling is more elegant than anything pydantic
allows. It loses on constraints, JSON schema, `InitVar`, and the amount of
validation behaviour we would own.

## Decided

### Details of building

- The whole tree is validated in one call to `validate_python` with
  `extra="forbid"`, on the adapter of the root class. pydantic runs every
  constructor, and with it every `__post_init__`, as part of it.
- Default values are validated by pydantic as well. Every value goes through
  one path: the resolved tree in plain form, then pydantic, which calls the
  constructor. A value that nothing set is the default, converted to plain
  form (`Path` to `str` and so on) and validated like any other. This is
  what lets a job build its config from the resolved tree alone
  ([multirun.md](multirun.md)), and it makes a single run behave exactly
  like a job. A sloppy default such as `size: int = None` is therefore an
  error, which pydantic itself would let through; OmegaConf rejects it too.
- A default that loses its type in plain form is an error. Where the type
  of the field does not say what to build from the plain form, pydantic
  keeps it as it is: under `Any` or `object`, also inside `list[Any]` or
  `dict[str, Any]`, and in a union with a member that takes the plain form,
  such as `Path | str` or `Color | str`. A default `Path("/tmp")`,
  `Color.RED` or `Adam()` would silently become `'/tmp'`, `'red'` or
  `{'lr': 0.1}`. The error names the field, the class, the default and what
  it becomes. A default that pydantic validates to the same result from
  both forms is fine. The check compares the instance that pydantic built
  with the default, so it is made once pydantic accepts everything else.

  | Field | Default | Result |
  | --- | --- | --- |
  | `Any` | `Color.RED`, `Path("/tmp")`, `Adam()` | Error |
  | `list[Any]` | `[Path("/tmp")]` | Error |
  | `Path \| str` | `Path("/tmp")` | Error |
  | `Level \| int` | `Level.LOW` (an `IntEnum`) | Error |
  | `Any` | `(1, 2)`, `{1, 2}` | `[1, 2]` |
  | `Sequence[int]` | `(1, 2)` | `[1, 2]` |
  | `Color`, `Path`, `list[Adam]` | `Color.RED`, `Path("/tmp")`, `[Adam()]` | Unchanged |
  | `float` | `1` | `1.0`, as pydantic converts the default itself |

  A tuple, list, set and frozenset count as the same, as do a dict and a
  frozendict, since a config file cannot tell them apart: a tuple default
  for `Sequence[int]`, a common way to avoid `default_factory`, stays
  allowed and arrives as a list. Only a default that nothing sets is
  checked; a value from a file or an override is what that source says.
  The rule is not limited to `Any`, because which unions lose the type
  depends on how pydantic matches their members, which the annotation does
  not show.

  A default made only of plain values (`str`, `int`, `float`, `bool`,
  `None`) in lists, tuples, sets, frozensets, dicts and frozendicts, and
  not of their subclasses, is not checked. Its plain form differs only in
  the kinds of the containers and the order of a set, so validating it a
  second time and comparing the results would be wasted on almost every
  default. The price is a contrived union, where the kind of container
  changes which member pydantic picks: `tuple[Path, ...] | list[str]` with
  the default `("x",)` builds `(Path("x"),)` from the default but `["x"]`
  from its plain form, and the field arrives as `["x"]` without an error.
- A bool that pydantic turns into a number is an error (see above).
- An `InitVar` is not kept on an instance, so under a default instance its
  value falls back to the default in the class. For the same reason, the
  checks for a bool and for a default that loses its type do not apply to
  it.
- `init=False` fields are not config keys. They cannot be set, and `to_dict`
  leaves them out, since a config file could not hold them.
- A field whose type pydantic cannot handle, such as `torch.device` or a
  sentinel from `typing_extensions`, is an error in the config class, which
  names the field. It could not be set from a config file anyway. It is
  found when the classes are checked, before any config file is read. A
  value made from the config, such as a `torch.device`, belongs in a
  property of the class.
- A `ValueError` or an `AssertionError` from a constructor is reported with
  the path and the class of its node, next to the other errors in the tree.
  pydantic passes any other exception on: it stops the validation, and is
  reported with its type and message, but without its path and without the
  other errors.
- An invalid value names the class and the type of the field in its error,
  next to the doc: `Field: Opt.lr (float). Learning rate.`

### Reserved names

A name that is reserved below the root starts and ends with an underscore,
so that it cannot get in the way of a field that a user would want to
write. A name that is reserved only at the root may go without.

| Name | Where it is reserved | Why |
| --- | --- | --- |
| `_defaults_` | Every config class | The key of every config file that says what the file builds on ([composition.md](composition.md)) |
| `_self_` | Every config class | A key in `_defaults_`, next to the names of fields |
| `_option_` | Every config class | The key that names the option of a group, in `to_dict` and in a TypedDict option. Reserved on every class, not only on options: one rule is simpler, and it keeps the name free |
| `_doc_` | Every config class | The key of a config file that describes the file, for `--help` ([helptext.md](helptext.md#docs-of-config-files)) |
| `nematocyte` | The root config class | nematocyte's own settings ([cli.md](cli.md)) |

- A field with a reserved name is an error that names the class and the
  field. A TypedDict option declares its option in its `_option_` key, and
  that key is not a field ([TypedDicts as options](unions.md#typeddicts-as-options)).
- `defaults`, Hydra's name for the key, is not reserved.

### TypedDict configs

A config class can also be a `TypedDict`. Its keys are config keys, as the
fields of a dataclass are.

```python
class Opt(TypedDict):
    """The optimizer."""

    lr: float
    wd: NotRequired[float]
```

- A TypedDict can be used wherever a dataclass can: as the root, as a plain
  subconfig, or as an option of a group. Dataclasses and TypedDicts can be
  mixed in one tree, but not within a group. An option of a group names
  itself in a key `_option_`, not with a `Tag`; see
  [unions.md](unions.md#typeddicts-as-options).
- `typing.TypedDict` and `typing_extensions.TypedDict` are both accepted;
  the second is a class of its own, which `typing.is_typeddict` does not
  recognise, and is needed for features such as `closed=True`.
- A TypedDict node is built as a plain `dict`. There is no constructor and
  no `__post_init__` to run.
- A TypedDict has no defaults. A `Required` key that nothing sets is an
  error ("a value is required"), as for a dataclass field without a
  default. A `NotRequired` key that nothing sets is left out of the dict.
  `total=False` and inheritance are read through `__required_keys__`.
- `NotRequired` is only for values. A subconfig or a group cannot be
  `NotRequired`, for the reasons in
  [unions.md](unions.md#no-optional-dataclass-fields-for-now).
- `Required`, `NotRequired` and `ReadOnly` are stripped before nematocyte
  reads the type of a key; pydantic reads them itself. `ReadOnly` has no
  other effect: pydantic warns that it does not enforce it, and nematocyte
  silences that warning.
- A TypedDict key has no doc; see [helptext.md](helptext.md).
- The default of a dataclass field typed as a TypedDict is a dict, such as
  `field(default_factory=lambda: Opt(lr=0.1))`, or, as an experimental
  feature, a `frozendict`
  ([below](#experimental-a-frozendict-as-the-default-of-a-typeddict-field)).
  It is the default instance of the node ([unions.md](unions.md#defaults)):
  a key it leaves out is not set, so it is an error if the key is
  `Required`.
- A program whose TypedDicts have `Required` keys, or a group of TypedDicts,
  that no default fills in needs a primary config (`config_name`) or
  overrides to run.
- `to_dict` keeps a TypedDict as it is, so every TypedDict in a config given
  to it may hold only values ([api.md](api.md#a-dict-for-logging)).

### Experimental: a `frozendict` as the default of a TypedDict field

**This feature is experimental.** It can change or go away in any release,
without a deprecation period.

On Python 3.15, the default of a dataclass field typed as a TypedDict can be
a `frozendict` ([PEP 814](https://peps.python.org/pep-0814/)), written
without `default_factory`. A dataclass takes it, because the class of a
`frozendict` is hashable, while a `dict` is refused as a mutable default:

```python
@dataclass
class Config:
    opt: Opt = cast("Opt", frozendict(lr=0.1))
```

- No type checker accepts a `frozendict` where a TypedDict is declared, so
  the default is written with `cast`, which every checker understands.
  ty still warns about it (`disjoint-cast`), because a TypedDict is a
  `dict` at runtime and a `frozendict` is not one.
- A `frozendict` default works as a `dict` default does: it is the default
  instance of the node, a key it leaves out is not set, and the default of
  a group of TypedDicts names its option in `_option_`
  ([unions.md](unions.md#typeddicts-as-options)). Inside it, a `frozendict`
  can be the default of a nested TypedDict, or the value of a key, such as
  one typed `dict[str, float]`.
- The built node is a plain `dict`, as always, never a `frozendict`.
- Only `dict` and `frozendict` are accepted. Another mapping, such as a
  `MappingProxyType`, is still an error that names the field and the class.
- On Python 3.14 there is no `frozendict`, and nothing changes.

Why experimental: the typing of a frozen TypedDict is not settled. PEP 814
defers it, with `class TD(TypedDict, frozen=True)` and `Frozen[TD]` as the
spellings proposed so far. If the typing spec adds one, nematocyte will
require it, so that the default can be written without `cast`, and a
`frozendict` default of a TypedDict declared without it will become an
error.

## Rejected ideas

### Strict validation through JSON

pydantic's strict mode behaves differently on JSON input:
`validate_json(json.dumps(value), strict=True)`. JSON has no paths, tuples or
enums, so in this mode pydantic allows exactly the conversions that strict
mode on Python input refuses, and still rejects everything else.

| Value | Lax | Strict, JSON input |
| --- | --- | --- |
| `True` for `int` or `float` | accepted as `1` | rejected |
| `"123"` for `int` | accepted | rejected |
| `1.0` for `int` | accepted | rejected |
| `"false"` or `0` for `bool` | accepted | rejected |
| `1` for `float` | accepted | accepted |
| `"1e-3"` for `float` | accepted | rejected |
| `str` for `Path`, `list` for `tuple`/`set`, value for enum | accepted | accepted |
| Unions, `Literal`, `dict[int, int]`, `dict[str, Any]` | accepted | accepted |

These are cleaner rules than lax mode, and they remove the `seed: true` wart
without a special case.

**Rejected because** every value would take a round trip through `json.dumps`
just to reach the validator. That also requires every value to be
JSON-serialisable, which a YAML parser does not guarantee: an unquoted
`2026-11-16` becomes a `date` object.

Rejecting `lr: 1e-3` (which PyYAML reads as a string) was not part of the
reason. Having to write `1.e-3` is acceptable, so that lax mode accepts
`1e-3` is a side effect and not a requirement. Since config files are read
as YAML 1.2 ([composition.md](composition.md)), the question no longer
arises: `1e-3` is a float.

### Validation from strings

`TypeAdapter.validate_strings(value, strict=True)` validates input made of
strings, as from environment variables.

**Rejected because** it takes only a string or a dict of strings: there is
no input for a list or for `None`, so `list[int]`, `tuple[...]` and
`X | None` cannot be validated at all. It would also throw away the types
that YAML already gives: `123` and `"123"` would be the same, a `str` field
would take a number, `"yes"` would be a `bool`, and a field typed `Any`
would hold `"0.1"` instead of `0.1`.

### Validating each field on its own

nematocyte walked the dataclass nodes itself, called pydantic with a
`TypeAdapter` for the type of each field, and called the constructor of
each node with the results. Groups needed nothing in pydantic, so they were
written in tyro's spelling, `Annotated[Fcn, option("fcn")] | ...`.

It allowed what whole-class validation cannot:

- skipping the constructors (`construct=False` in
  [validation.md](validation.md#skipping-the-constructors)),
- a field whose type pydantic cannot read, with a default that was checked
  with `isinstance`,
- the path of a node in the error from any exception its constructor
  raised, and the other errors next to it,
- the check for a bool next to the other errors.

**Rejected in favour of** one call for the whole tree. The groups are then
pydantic's own tagged unions, so `TypeAdapter(Config)` validates and
describes a whole config without nematocyte, and nematocyte keeps no
validation machinery of its own beyond mapping the errors back and the two
checks pydantic does not make.
