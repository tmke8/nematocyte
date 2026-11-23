# Unions and config groups

How the concrete class is chosen for a field that can hold one of several
config classes.

## Declaring a group

A group is declared in the type of the field, as a union whose members carry
their names. It is written as pydantic's own tagged union: each member has a
`Tag` with its name, and the union has `nematocyte.pick` as its
`Discriminator`.

```python
@dataclass
class Config:
    data: Annotated[
        Annotated[WikiDataConfig, Tag("wiki")]
        | Annotated[HelpSteerDataConfig, Tag("helpsteer")],
        Discriminator(nematocyte.pick),
    ]
```

The members are the options of the group, the word Hydra uses
(`group=option`, "Available options in 'model'").

pydantic validates the whole config in one call
([deserializer.md](deserializer.md)), and so has to know which member a node
of the group is. In this spelling it needs no help from nematocyte beyond
`pick`, which reads the option that composition selected. The spelling is
longer than tyro's `Annotated[X, option("x")]`, which nematocyte used before,
but it is standard pydantic, and `TypeAdapter(Config)` understands it.

This replaces the registry in `hydra-example` (`CONFIG_GROUPS`, passed to
`register_hydra_config`), where the field is typed as an abstract base and the
allowed classes are listed somewhere else.

The names and classes are read back with
`typing.get_type_hints(Config, include_extras=True)`. This covers:

- **A reusable alias.**
  `type Model = Annotated[Annotated[Fcn, Tag("fcn")] | Annotated[Cnn, Tag("cnn")], Discriminator(nematocyte.pick)]`
  can be used for several fields. The alias has to be unwrapped through
  `__value__`.
- **No common base class.** The members only have to be dataclasses. A base
  class is still possible, and useful for sharing an interface.
- **Telling a group from an ordinary union.** `str | None` has no named
  dataclass members and stays a value.

Things to get right:

- `include_extras=True` is required. Without it `get_type_hints` strips
  `Annotated`, and the names are gone.
- The union reaches pydantic as it is written. nematocyte hands over the node
  of a group as a mapping that holds the name of the selected option under
  the key `_option_`, next to the fields, as `to_dict` writes it. `pick`
  returns that name, and the member ignores the key, so an option of a
  group of dataclasses cannot forbid unknown keys in its pydantic config
  (see [deserializer.md](deserializer.md#unknown-keys)).

The names in the annotations are the built-in options. YAML files in the
group's directory add further options on top, as before (see
[composition.md](composition.md)).

### Every member needs a name

A union of dataclasses with a member without a `Tag`, such as a plain
`WikiDataConfig | HelpSteerDataConfig`, is an error. Names are not derived
from the class names, unlike in tyro.

### The union is closed

Every option is listed where the field is declared. Another module cannot
add one later, as it could with a registry, so there are no plugins.

This is accepted. It also settles how launchers and sweepers are provided:
what Hydra ships as plugins (submitit, Optuna) is built into nematocyte, and
the `launcher` and `sweeper` fields under the reserved key are ordinary
closed unions.

The built-in options keep Hydra's names ([multirun.md](multirun.md)):

| Field | Options |
| --- | --- |
| `nematocyte.launcher` | `basic` (the default), `submitit_slurm` |
| `nematocyte.sweeper` | `basic` (the default), `optuna` |

submitit and Optuna are optional extras of the nematocyte package. They are
imported only when their option is selected.

## Optional subconfigs

A plain subconfig can be switched off: its field is typed `X | None`, where
`X` is a dataclass or a TypedDict.

```python
@dataclass
class Config:
    ema: EmaCfg | None = None
    """Off unless a config switches it on."""
    augment: AugmentCfg | None = field(default_factory=AugmentCfg)
    """On unless a config switches it off."""
```

### How Hydra does it

Tested with hydra-core, with `ema: Optional[EmaCfg] = None` and
`ema_on: Optional[EmaCfg] = field(default_factory=EmaCfg)`, where `EmaCfg`
has `decay: float = 0.99` and `every: int = 1`:

| Where | Written | Result |
| --- | --- | --- |
| File | `ema_on: null` | `None` |
| Command line | `ema_on=null` | `None` |
| Command line | `~ema_on` | `EmaCfg(decay=0.99, every=1)`: the key is deleted, and `OmegaConf.to_object` puts the default of the dataclass back |
| File | `ema: {}` | `EmaCfg()` |
| File | `ema: {decay: 0.5}` | `EmaCfg(decay=0.5, every=1)` |
| Command line | `ema={}`, `ema={decay:0.5}` | The same as in a file |
| Command line | `ema.decay=0.5`, with `ema` at `None` | "Cannot set 'ema.decay' because 'ema' is None" |

### Decided

- **The type.** The field is `X | None` (or `Optional[X]`), with one
  dataclass or TypedDict `X` that carries no `Tag`. Its default is `None`
  (off), an instance of `X` (on), or there is none. `None` is still not
  allowed in a group, and a subconfig still cannot be a `NotRequired` key
  of a TypedDict ([deserializer.md](deserializer.md#typeddict-configs)).
- **`null` switches it off**: `ema: null` in a file, `ema=null` on the
  command line. In a file, a key with nothing under it (`ema:` and then
  only comments) is the same value to YAML, and switches it off too. For a
  subconfig that is not optional, and for a group, such a key is still
  ignored ([composition.md](composition.md)). What lower layers set under
  the field is dropped, without a `DroppedValueWarning`: that is what
  `null` asks for.
- **A mapping switches it on**: `ema: {decay: 0.5}` in a file,
  `ema={decay:0.5}` on the command line ([cli.md](cli.md)), or a preset
  (`ema=fast`, or `ema: fast` under `_defaults_`). If the field is off, the
  node starts from the class defaults of `X`, as when an option of a group
  is selected, not from the default instance of the field, and the mapping
  or preset is applied on top. If it is on, the mapping is merged into it,
  as for any subconfig. `ema={}` switches it on with the class defaults,
  and leaves it as it is if it is on.
- **A dotted key does not switch it on.** `ema.decay=0.5` while `ema` is off
  is an error that names the key and the class, and shows `ema={decay:0.5}`.
  Setting one value should not bring back all the others. The same holds
  for a dotted entry under `_defaults_`, such as `ema.inner: fast`.
- **The order on the command line has no meaning**, as for every override
  ([cli.md](cli.md#group-or-value-from-the-fields-type)). `ema=null` and
  `ema={...}` are applied with the selections, shortest path first, so
  `ema=null ema.decay=0.5` is an error in either order, and
  `ema={} ema.decay=0.5` works in either order. They are the same key as
  `ema=fast`: the last one wins.
- **Nothing else changes for the node.** An interpolation that refers to a
  key under a subconfig that is off is an error, as for a key that has no
  value ([interpolation.md](interpolation.md)). `nematocyte.to_dict` writes
  `null` for it. `--help` and `--cfg` show it as `null`
  ([helptext.md](helptext.md)).

### Rejected ideas

- **`~ema` switches it off.** In Hydra, `~ema` deletes the key, and the
  default of the dataclass comes back, which is "off" only when the default
  is `None`. In nematocyte, `~` leaves out a key of a TypedDict
  ([cli.md](cli.md#key-leaves-out-an-optional-key-of-a-typeddict)); on a
  field of a dataclass it is an error, which points to `ema=null`.
- **A name for "on"**, as in `Annotated[EmaCfg, Tag("on")] | None` with
  `ema=on`, under the rule that every member of a union needs a name. A
  mapping switches it on without a name, and `on` would look like a preset.
- **A dotted key switches it on**, as a mapping does. It would make
  `ema.decay=0.5` bring back every other value of `EmaCfg` from its class
  defaults, which is too surprising for a change to one value.

## Defaults

### A group field can have a default

```python
decoder: Model = field(default_factory=Cnn)
```

The default option (`cnn`) follows from the type of the default value. The
primary config then does not have to choose one.

### A default instance is the config used when nothing is selected

```python
decoder: Model = field(default_factory=lambda: Cnn(kernel_size=3))
```

The default is one specific config. It is not a change to what `cnn` means.

- With nothing selected, the field holds the default instance. Values set in
  a config file or on the command line apply on top of it:
  `decoder.stride=1` gives `Cnn(kernel_size=3, stride=1)`.
- Selecting an option by name starts from that option's class defaults.
  `decoder=cnn` gives `kernel_size=5`, the default on the class, and
  `decoder=fcn` gives an `Fcn`.

Observed with Hydra 1.4.0rc3, where the class default is `kernel_size=5`:

| Case | Hydra | nematocyte |
| --- | --- | --- |
| `model = Cnn(kernel_size=3)`, no selection | `kernel_size=3` | same |
| ... and `model.stride=1` on the command line | `kernel_size=3` | same |
| ... and the primary config selects `model: cnn` | `kernel_size=5` | same |
| ... and `fcn` is selected | Validation error while composing | `Fcn` with its class defaults |

The last row is the only difference, and it is why `hydra-example` gives its
group fields no default: in Hydra a default instance fixes the class.

### Class defaults are requested by naming an option

The rule behind the two cases above: a node starts from the class defaults
only when an option of a group is named. Everything else applies on top of
the field's default.

Naming an option also drops whatever lower layers selected or set under the
node; see [composition.md](composition.md#layers).

An option is named either directly (`decoder=cnn`) or by `_self_` in a
file's `_defaults_` ([composition.md](composition.md)). A file in a group's
directory has to say which option it extends, and the chain ends at an
option from the annotations:

```yaml
# configs/decoder/small_cnn.yaml
_defaults_:
  _self_: cnn

stride: 1
```

`decoder=small_cnn` therefore gives `kernel_size=5`, the class default.

A preset for a plain subconfig names no option, because there is only one
class. Its `_self_`, if it has one, can only name another preset. So a
preset never requests the class defaults: the chain of presets applies on
top of the field's default. `opt=preset` with a file containing `wd: 0.1`
gives the same result as `opt.wd=0.1`.

This matches Hydra. Observed with
`opt: Opt = field(default_factory=lambda: Opt(lr=0.1))` and a class default
of `lr=5e-05`:

| Case | Hydra | nematocyte |
| --- | --- | --- |
| Nothing else | `lr=0.1` | same |
| `opt.wd=0.5` on the command line | `lr=0.1` | same |
| The primary config sets `opt.wd` | `lr=0.1` | same |
| `opt=preset`, a file that sets only `wd` | `lr=0.1` | same |

For the deserializer this means that a node is not always built from the
class alone. Unless an option is named for the field, a key missing from the
config takes its value from the field's default instance, not from the class
definition.

## Decided

### Details of declaring a group

- A union that mixes a dataclass with something else, such as `Sub | int`,
  is an error. A union with a dataclass in it is a group, and all of its
  members have to be named dataclasses.
- A union of two or more options needs exactly one `Discriminator`, and it
  has to be `nematocyte.pick`. Without one, pydantic would guess the member
  from the fields of the node, and a node of `Fcn` could be built as a
  `Cnn`. A missing or different discriminator is an error that names the
  field.
- A single `Annotated[X, Tag("x")]` is a group with one option. It needs no
  `Discriminator`, since there is nothing to choose; one is allowed.
- A member with more than one `Tag` is an error.
- A class can be in a group only once, and an option name only once.
- An option name, the string in the `Tag`, starts with a letter or an
  underscore, followed by letters, digits, `_` or `-`, so that it can be
  typed unquoted.
- The default of a plain dataclass field has to be an instance of exactly
  the declared class; an instance of a subclass is an error, since the node
  is built as the declared class. The default of a group field has to be an
  instance of exactly one of the option classes.

### TypedDicts as options

The options of a group can be TypedDicts
([deserializer.md](deserializer.md#typeddict-configs)). A dict does not
know its class, so a TypedDict that is an option names itself, in the key
`_option_`, and nematocyte fills it in:

```python
class BatchNorm(TypedDict):
    _option_: Literal["BN"]
    momentum: NotRequired[float]


class LayerNorm(TypedDict):
    _option_: Literal["LN"]
    eps: NotRequired[float]


@dataclass
class Model:
    norm: Annotated[BatchNorm | LayerNorm, Discriminator("_option_")] = field(
        default_factory=lambda: LayerNorm(_option_="LN")
    )
```

`model.norm=BN model.norm.momentum=0.2` gives
`{"_option_": "BN", "momentum": 0.2}`, and the program tells the options
apart by that key, which type checkers use to narrow the union:
`match norm: case {"_option_": "BN"}: ...`.

- The key is a `Literal` of one string, which is the option name and follows
  the rules for one. It cannot be `NotRequired`; `ReadOnly` is allowed.
- In the built dict, `_option_` is the first key, as it is in the output of
  `to_dict` for a dataclass option.
- A `Tag` is not used for a TypedDict; it is an error there. A union of two
  or more TypedDicts has pydantic's own `Discriminator("_option_")`, which
  reads the key; without it, the union is an error that names the field.
- A TypedDict in a group without the key is an error. A single TypedDict
  with the key, as the type of a field, is a group with one option, like a
  single `Annotated[X, Tag("x")]`; one without the key is a plain
  subconfig. The root cannot have the key.
- `_option_` is not a config key. A config file or an override cannot set
  it; the option is selected as for any group, and a file in the group's
  directory gets the name of the option its `_self_` chain ends at.
- A group's options are either all dataclasses or all TypedDicts. A mix is
  an error.
- A group field of TypedDicts can have a default in the class, a dict that
  names its option in `_option_`:
  `field(default_factory=lambda: LayerNorm(_option_="LN"))`. That option is
  used when nothing selects one, and the other keys of the dict are the
  default instance ([Defaults](#defaults)): they apply until an option is
  named. A default without the key, or with a name that is not an option,
  is an error. On Python 3.15 the dict can be a `frozendict`, without
  `default_factory`; that is experimental
  ([deserializer.md](deserializer.md#experimental-a-frozendict-as-the-default-of-a-typeddict-field)).

## No tagging

No tagging strategy is needed. In Hydra the choice of class never lives in the
data: it lives in the defaults list (`- model: cnn`), in a group file's
inheritance chain (`celeba_male_smiling_small` -> `celeba_male_smiling` ->
`celeba` -> `CelebADataModule`), or in a CLI override (`model=cnn`). All of
these are resolved during composition, so by the time values are validated
the composer holds a concrete class for every group node. Structuring never
has to guess which member of a union a dict belongs to.

Two things follow:

- The same group can appear at several paths with different choices
  (`encoder: Model`, `decoder: Model`); the choice is keyed by path, not by
  type. Tested with both cattrs and pydantic.
- The `_option_` key of a TypedDict option, and the `Tag` of a dataclass
  option, are not tags in this sense. nematocyte attaches the option to the
  data it hands to pydantic, after composition chose it, and pydantic's
  discriminator reads it there. It is never read from a config file.
- Because the data carries no tag, a dumped config is not self-describing.
  Whatever is written out for reproducibility has to record the choices next
  to the values, e.g. in `_defaults_`, so that it can be loaded again.
  The dict for logging records each choice under the key `_option_`
  ([api.md](api.md)).

How the chosen class is then handed to the validator is covered in
[deserializer.md](deserializer.md).

## No `_target_`

In Hydra a config can name a class or function by its import path under the
key `_target_`, and `hydra.utils.instantiate` calls it with the other keys
as arguments. nematocyte does not implement this.

The class of every node comes from the schema: the type of the field, or
the option of a group that is selected by name. A config file never names
Python code.

A program that wants to name a class in its config holds the import path in
a `str` field and imports it itself, as `opt.optimizer_cls` does in the
example.
