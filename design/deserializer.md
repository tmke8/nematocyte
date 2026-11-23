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

Validation uses pydantic's **strict mode on JSON input**: the tree is
dumped as JSON and validated with `validate_json(text, strict=True)`; see
[Strict validation through JSON](#strict-validation-through-json). Strict
mode on Python input (`validate_python(value, strict=True)`) is not usable:
it rejects `str` for `Path`, `list` for `tuple` and `set`, a value for its
enum member, and `dict` for a dataclass. YAML can produce none of the types
it insists on. Lax mode, which nematocyte used before, accepts too much:
`true` for an `int`, `"123"` for an `int`, `1.0` for an `int`.

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
   pydantic as a mapping with the option that composition chose under the
   key `_option_`, next to the fields. `pick` returns that option, and
   pydantic builds the member with that `Tag`; the member ignores the key
   ([Unknown keys](#unknown-keys)).
2. **Per-type behaviour cannot be changed globally**, e.g. accepting enum
   members by name as Hydra does, or converting a string to `torch.device`.
   A type that pydantic cannot read is an error in the schema; a value made
   from the config, such as a `torch.device`, belongs in a property of the
   class (as `Config.device` in the example).

With that, the whole tree goes to pydantic in one call:

```text
build(cls, node):
    data = node in JSON form, each node of a group with its option in `_option_`
    instance = TypeAdapter(cls).validate_json(json.dumps(data), strict=True)
        # checks every value and runs every constructor and __post_init__
    on errors: map each location back to its node and field, skipping the
               tag of a group, and report it with where the value was set
    check the defaults that nothing set, each against its field alone
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
- A bool is rejected for a number, as OmegaConf does, and so are `1.0` for
  an `int` and `"false"` for a `bool`. Where the type allows a bool
  (`int | bool`, `Any`), it stays a bool. See
  [Strict validation through JSON](#strict-validation-through-json).
- Every value has to have a JSON form. A value that a config file cannot
  hold, such as a `torch.device`, belongs in a field with `init=False` or in
  a property.
- A typo in a key inside a value, such as an item of `list[Point]`, is only
  reported if the class opts in; see [Unknown keys](#unknown-keys).
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

- The whole tree is dumped as JSON and validated in one call to
  `validate_json(text, strict=True)`, on the adapter of the root class,
  without `extra`. pydantic runs every constructor, and with it every
  `__post_init__`, as part of it. What strict mode accepts is in
  [Strict validation through JSON](#strict-validation-through-json).
- Default values are validated by pydantic as well. Every value goes through
  one path: the resolved tree in JSON form, then pydantic, which calls the
  constructor. A value that nothing set is the default, converted to JSON
  form (`Path` to `str` and so on) and validated like any other. This is
  what lets a job build its config from the resolved tree alone
  ([multirun.md](multirun.md)), and `from_run_config` rebuild the config
  object of a run from its saved `config.json`
  ([api.md](api.md#reading-the-config-of-a-run-back)). It also makes a
  single run behave exactly like a job. A sloppy default such as
  `size: int = None` is therefore an error, which pydantic itself would let
  through; OmegaConf rejects it too.
- A default that loses its type in JSON form is an error. Where the type
  of the field does not say what to build from the JSON form, pydantic
  keeps it as it is: under `Any` or `object`, also inside `list[Any]` or
  `dict[str, Any]`, and in a union with a member that takes the JSON form,
  such as `Color | str`. A default `Path("/tmp")`, `Color.RED`, `Adam()`
  or `date(2026, 1, 1)` would silently become `'/tmp'`, `'red'`,
  `{'lr': 0.1}` or `'2026-01-01'`. Building every value from its JSON form is what makes a
  single run, a job of a multirun and `from_run_config` agree; this error
  keeps that from changing a default without a word. The error names the
  field, the class, the default and what it becomes. A default that
  pydantic validates to the same result from both forms is fine. Both
  forms are validated against the type of the field alone, so a
  constructor that changes the field later, such as a `__post_init__` that
  resolves a path, does not count.

  | Field | Default | Result |
  | --- | --- | --- |
  | `Any` | `Color.RED`, `Path("/tmp")`, `Adam()` | Error |
  | `list[Any]` | `[Path("/tmp")]` | Error |
  | `Any` | `date(2026, 1, 1)` | Error |
  | `Color \| str` | `Color.RED` | Error |
  | `Level \| int` | `Level.LOW` (an `IntEnum`) | Error |
  | `Path \| str` | `Path("/tmp")` | Unchanged: strict mode builds the `Path` |
  | `Any` | `(1, 2)`, `{1, 2}` | `[1, 2]` |
  | `Sequence[int]` | `(1, 2)` | `[1, 2]` |
  | `Color`, `Path`, `list[Adam]`, `date` | `Color.RED`, `Path("/tmp")`, `[Adam()]`, `date(2026, 1, 1)` | Unchanged |
  | `float` | `1` | `1.0`, as pydantic converts the default itself |

  A tuple, list, set and frozenset count as the same, as do a dict and a
  frozendict, since a config file cannot tell them apart: a tuple default
  for `Sequence[int]`, a common way to avoid `default_factory`, stays
  allowed and arrives as a list. Only a default that nothing sets is
  checked; a value from a file or an override is what that source says.
  The rule is not limited to `Any`, because which unions lose the type
  depends on how pydantic matches their members, which the annotation does
  not show.

  A default that the conversion to JSON form changes only in the kinds of
  its containers and the order of a set is not checked. Only an enum
  member, a path, a date, a datetime, a time, a dataclass instance and a
  subclass of a container, such as a `NamedTuple`, are converted; any other
  value, such as a `str`, an `int` or `None`, goes into both forms
  unchanged, inside lists, tuples, sets, frozensets, dicts and frozendicts. Validating the field
  twice on its own, once for each form, and comparing the results would be
  wasted on almost every default. The price is a contrived union, where
  the kind of container changes which member pydantic picks:
  `list[Path] | set[str]` with the default `{"x"}` builds `{"x"}` from the
  default but `[Path("x")]` from its JSON form, and the field arrives as
  `[Path("x")]` without an error.
- A date, a datetime and a time are written in JSON as their ISO strings,
  with `isoformat()`, whether they come from a default or from YAML, where
  an unquoted `2026-11-16` is a `date`. A `date` field gets its date back;
  under `Any`, a date from YAML arrives as the string `'2026-11-16'`, as it
  does from JSON5, and a `str` field accepts it.
- A value that has no JSON form is an error that names the field, the
  class, the value and where it was set, with the part of the value at
  fault: an object such as a `Decimal` or a `torch.device` as a default,
  `bytes` from YAML's `!!binary`, or a key that JSON cannot have, such as
  the tuple from a YAML complex key. A value made from the config belongs
  in a field with `init=False`, set in `__post_init__`, or in a property.
- An `InitVar` is not kept on an instance, so under a default instance its
  value falls back to the default in the class.
- `init=False` fields are not config keys. They cannot be set, and `to_dict`
  leaves them out, since a config file could not hold them.
- A field whose type pydantic cannot handle, such as `torch.device` or a
  sentinel from `typing_extensions`, is an error in the config class, which
  names the field. It could not be set from a config file anyway. It is
  found when the classes are checked, before any config file is read. A
  value made from the config, such as a `torch.device`, belongs in a
  property of the class.
- This includes a field with `init=False` that has a default: pydantic
  reads its type too, although it is not a config key. Without a default,
  pydantic leaves the field out, and `__post_init__` can set it.
- The pydantic config of a class (`__pydantic_config__`, or
  `pydantic.with_config`) applies to its fields, also in the check for a
  default that loses its type, except for `strict`, which the call sets.
  With `arbitrary_types_allowed=True`, a field of any type is accepted by
  the schema, but its value has no JSON form, so a field of such a type
  that is a config key is an error when the config is built. The check for
  a default that loses its type skips a field whose type pydantic can only
  read with that setting.
- pydantic does not call a hand-written `__init__` of a config dataclass:
  it creates the instance, sets the fields and calls `__post_init__`. Work
  that has to run when the config is built belongs in `__post_init__`.
- A `ValueError` or an `AssertionError` from a constructor is reported with
  the path and the class of its node, next to the other errors in the tree.
  pydantic passes any other exception on: it stops the validation, and is
  reported with its type and message, but without its path and without the
  other errors.
- An invalid value names the class and the type of the field in its error,
  next to the doc: `Field: Opt.lr (float). Learning rate.`

### Strict validation through JSON

pydantic's strict mode behaves differently on JSON input:
`validate_json(json.dumps(value), strict=True)`. JSON has no paths, tuples or
enums, so in this mode pydantic allows exactly the conversions that strict
mode on Python input refuses, and still rejects everything else. nematocyte
validates the tree this way.

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

These are cleaner rules than lax mode, closer to OmegaConf's, and they
reject `seed: true` without a special case. A config that is run and a
config that is read back from the `config.json` of a run go through the
same JSON, so they cannot differ.

This was rejected at first, because every value takes a round trip through
`json.dumps` just to reach the validator, and a YAML parser gives values
that JSON does not have. It is used for its rules, which pydantic offers in
no other way; [pydantic#11154](https://github.com/pydantic/pydantic/issues/11154)
might make the round trip unnecessary one day. Dates are written as ISO strings, and any
other value without a JSON form is an error
([Details of building](#details-of-building)). Since config files are read
as YAML 1.2 ([composition.md](composition.md)), `lr: 1e-3` is a float, and
the rejection of the string `"1e-3"` does not matter in a config file.

### Unknown keys

- Composition rejects a key that is not a field of a config class, in a
  config file, an override, a key of `_defaults_` and the dict given to
  `from_run_config`, with the file or override and a suggestion
  ([composition.md](composition.md)). This covers every node of the tree.
- pydantic gets no `extra` in the call, so each class decides for itself,
  through its pydantic config, as it would without nematocyte. pydantic's
  default for a dataclass and a TypedDict is to ignore an unknown key.
- So a key inside a value, which composition does not look into, such as a
  typo in an item of `list[Point]`, is dropped without an error, and the
  field takes its default. A class can opt in with
  `@with_config(ConfigDict(extra="forbid"))`; then the error names the
  field, the item and where the value was set.
- An option of a group of dataclasses cannot set `extra` to `"forbid"` or
  `"allow"`: it receives its option under `_option_`, which it has to
  ignore. That is an error in the schema, which names the group and the
  class. A TypedDict option declares `_option_` as a key, and is not
  affected.

Before the switch to JSON, the call passed `extra="forbid"`, which forbade
unknown keys inside values as well. Every class ignoring the key
`_option_` was the price of passing the option in the data; see
[Picking the option without a key in the data](#picking-the-option-without-a-key-in-the-data).
Checking the keys inside values in nematocyte, by walking each value along
its type, would bring the protection back for every class, and can be added
later.

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
  default. A `NotRequired` key that nothing sets is left out of the dict,
  and `~KEY` on the command line leaves it out even if something set it
  ([cli.md](cli.md#key-leaves-out-an-optional-key-of-a-typeddict)).
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
  with `isinstance`, without opting in through the pydantic config of the
  class,
- a hand-written `__init__`, which was called like any constructor,
- the path of a node in the error from any exception its constructor
  raised, and the other errors next to it.

**Rejected in favour of** one call for the whole tree. The groups are then
pydantic's own tagged unions, so `TypeAdapter(Config)` validates and
describes a whole config without nematocyte, and nematocyte keeps no
validation machinery of its own beyond mapping the errors back and the
check for a default that loses its type, which pydantic does not make. That
check still validates a field on its own, but only for a default that is
not in JSON form.

### Rejecting a bool for a number

nematocyte rejected `seed: true` for an `int` or `float` field, also inside
a list, tuple or dict, as OmegaConf does. Since pydantic's lax mode accepts
it, every value that held a bool was validated a second time against the
type of its field alone, and the result was searched for a bool that had
become a number.

**Rejected because** a `bool` is an `int` in Python, and a check that works
against the language is not worth what it costs. With one pydantic call for
the whole tree, the second validation was spent on every `bool` field that
a config set, on the chance that the field was a number.

[Strict validation through JSON](#strict-validation-through-json) rejects
a bool for a number since, as part of the one call, at no cost.

### Picking the option without a key in the data

With JSON input, the option of a group has to be in the JSON, where the
member does not expect it. Ways to keep it out, so that the call could
still pass `extra="forbid"`:

- **`Discriminator("_option_")` with a `BeforeValidator` on each member
  that removes the key.** A string discriminator needs a `Literal` field of
  that name in every member, so pydantic refuses the schema for a
  dataclass. With `nematocyte.pick` instead, the validator works, but
  pydantic hands it a Python object and validates what it returns as Python
  input: below the group, strict mode rejects `[1, 2]` for a
  `tuple[int, int]`. A `WrapValidator` has the same problem.
- **The `context` of `validate_json`.** A discriminator is called with the
  value alone, and never sees the context. A validator sees it, but turns
  the value into Python input, as above.
- **A queue of the options in a `ContextVar`**, which `pick` takes from in
  the order that pydantic visits the groups. It works, but that order is
  pydantic's, not the order of the fields: the fields of `__init__` come
  first, the `kw_only` fields after them. One option out of place gives the
  wrong option to every group after it, and a saved `config.json` could no
  longer be validated without nematocyte.

**Rejected because** each either loses what the switch to JSON is for, or
rests on how pydantic works inside. `_option_` in the data, as `to_dict`
writes it anyway, is what pydantic sees; the option classes ignore it.

### Ignoring unknown keys in the call

`validate_json(..., extra="ignore")` drops `_option_` with every other
unknown key, for every class.

**Rejected because** the call's setting overrides the class's own, so no
class could opt in to forbidding unknown keys inside a value.
