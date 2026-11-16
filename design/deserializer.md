# Deserializer

How the composed config tree becomes dataclass instances.

## Replacing OmegaConf: cattrs or pydantic?

### Conclusion

Use **pydantic**, but only as the validator for individual field values.
nematocyte itself should own the config tree (plain `dict`/`list` until the very
end) and the walk over dataclass nodes.

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
| Unmodified stdlib dataclasses: `kw_only`, inheritance, ABCs, `default_factory`, `Path`, `X \| None`, `dict[int, int]`, PEP 695 `type A = Literal[...]` | Works | Works through `TypeAdapter`; the classes are not touched and the results are ordinary instances |
| `__post_init__` checks | Run. The message is replaced by "invalid value" unless a custom error formatter is supplied | Run. Reported with path and the original message |
| `InitVar` fields | Not supported (rejected as an unknown key) | Supported |
| Scalar coercion out of the box | Unsafe: `1.5` becomes `1` for `int`, `"false"` becomes `True`, `None` becomes `"None"`, `"abc"` becomes `["a", "b", "c"]` for `list[str]`. Every primitive hook has to be replaced (about 30 lines) | Lax mode is close to ideal: `1.5` is rejected for `int`, `"false"` is `False`, `123` is rejected for `str`, and `"1e-3"` is accepted for `float` (so the `1.e-3` workaround for YAML 1.1 goes away). Warts: `True` is accepted as `1`, `"123"` as `123` |
| Unions of plain values: `int \| str`, `int \| list[int]`, `float \| Literal["auto"]` | Unsupported until `configure_union_passthrough` is applied; then values pass through on an `isinstance` match with no coercion, and errors are confusing | Work, with one error line per union member |
| Unknown keys | `Converter(forbid_extra_keys=True)` | `validate_python(..., extra="forbid")`, per call, pydantic 2.12 or later; applies to nested dataclasses too. `TypeAdapter(config=...)` is refused for dataclasses |
| Error reporting | All errors collected, with paths. Wording is generic ("invalid value for type, expected int") and needs a custom formatter | All errors collected, with paths, specific wording and the offending input |
| Value constraints, e.g. `Annotated[float, Field(ge=0, le=1)]` | Ignored | Enforced |
| Picking the concrete class for a field typed as an abstract base | One hook on the converter. The composer hands over a `dict` subclass that carries the chosen class; single pass, error paths intact, works at any depth | No external hook. The node has to be instantiated first and passed in as an instance (accepted for a base-typed field; an instance of the wrong class is rejected) |
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

1. **Config groups cannot be resolved inside pydantic.** There is no way to
   say "at path `model`, build a `SimpleCNNFactory`" without annotating the
   user's classes. (Where that choice comes from is in [unions.md](unions.md).)
2. **Per-type behaviour cannot be changed globally**, e.g. accepting enum
   members by name as Hydra does, or converting a string to `torch.device`.

Both are solved by the same structure: nematocyte walks the dataclass nodes
itself and calls pydantic per field.

```text
build(cls, node):
    for each field of cls:
        group field        -> build(chosen_class, node[field])
        dataclass field    -> build(field_type, node[field])
        registered type    -> user-supplied converter
        anything else      -> TypeAdapter(field_type).validate_python(node[field])
    report unknown and missing keys
    return cls(**kwargs)          # runs __post_init__
```

The `registered type` branch is not in the first version. No field in
`hydra-example` needs a converter, and the walk leaves room to add one.

A field without a default has to be given a value, and a missing one is
reported with its path. Hydra's marker for a mandatory value, `???`, is not
implemented: a dataclass field without a default already says the same. In
a config file, `???` is an ordinary string.

A sketch of this (about 30 lines) built the `hydra-example` config, collected
every error across the tree with full paths, and surfaced the `__post_init__`
message. The walk is not wasted effort: the same schema traversal is needed
anyway for `--help`, for checking that a CLI override names a real key, and
for interpolation ([interpolation.md](interpolation.md)).

Costs of this approach, to be aware of:

- nematocyte handles `InitVar` and `init=False` fields itself.
- A key missing from the config does not always fall back to the class
  default. Unless an option of a group is named for the field, the value
  comes from the field's default instance (see
  [unions.md](unions.md#defaults)).
- Group fields and registered converters are only recognised as direct fields
  of a dataclass, not nested inside `list[...]` or `dict[...]`. Hydra has the
  same limitation for groups.
- Enum-by-name inside a container would need annotation rewriting. Using
  `Literal` or `StrEnum` avoids the problem.
- `seed: true` is accepted as `1`. If that matters, it is a one-line check in
  the walk.
- A number is rejected for a `str` field, in config files and on the command
  line alike, and has to be quoted. See [cli.md](cli.md) for why this is not
  relaxed.

The alternative is to hand whole dataclass nodes to pydantic and only
instantiate group nodes ahead of time. That is less code and also worked in
testing, but gives up control over third-party types and the wording of
structural errors.

### When cattrs would be the better choice

If a small pure-Python dependency matters more than error quality, or if
user-registered converters for arbitrary types turn out to be a central
feature. The cattrs route is entirely viable: about 30 lines of strict
primitive hooks, one hook for group bases, one custom error formatter, and one
unstructure hook. Its group handling is more elegant than anything pydantic
allows. It loses on constraints, JSON schema, `InitVar`, and the amount of
validation behaviour we would own.

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
