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
| File | `plain: null`, or `plain:` with nothing under it, for `plain: EmaCfg` | "field 'plain' is not Optional" |
| File | The same for a group field typed as its base class, `model: ModelCfg = MISSING` | "field 'model' is not Optional"; with `Any` as the type, `None` |

### Decided

- **The type.** The field is `X | None` (or `Optional[X]`), with one
  dataclass or TypedDict `X` that carries no `Tag`. Its default is `None`
  (off), an instance of `X` (on), or there is none. A group can be optional
  too ([below](#optional-groups)), and a subconfig still cannot be a
  `NotRequired` key of a TypedDict
  ([deserializer.md](deserializer.md#typeddict-configs)).
- **`null` switches it off**: `ema: null` under `_defaults_` in a file,
  as for an optional group ([below](#optional-groups)), and `ema=null` on
  the command line. A dotted entry such as `outer.ema: null` works too.
  What lower layers set under the field is dropped, without a
  `DroppedValueWarning`: that is what `null` asks for. For a subconfig or
  a group that is not optional, null under `_defaults_` or on the command
  line is an error that shows the spelling with `| None`, as in Hydra
  ("field 'plain' is not Optional").
- **Among the values of a file, null is an error**, for a subconfig or a
  group, optional or not: `ema: null`, and a key with nothing under it
  (`ema:` and then only comments), which YAML reads as null. Hydra switches
  an optional subconfig off there, so ignoring it would surprise, and a key
  whose lines were all commented out may have been meant to leave the field
  as it is. The error shows what may have been meant: the entry under
  `_defaults_`, with the path from the node of the file, for an optional
  field; leaving out the key; and, for a TypedDict, `ema: {_replace_: true}`,
  which drops what was set in it before
  ([composition.md](composition.md#_replace_-true-replaces-instead-of-merging)).
- **In the file that switches the field off** under `_defaults_`, any value
  for it among the values is an error: the file says two things about the
  field. A file above it can switch it on again with a mapping or a
  preset.
- **A mapping or a preset switches it on**: `ema: {decay: 0.5}` in a
  file, or a preset (`ema=fast`, or `ema: fast` under `_defaults_`). If the
  field is off, the node starts from the class defaults of `X`, as when an
  option of a group is selected, not from the default instance of the
  field, and the mapping or preset is applied on top. If it is on, the
  mapping is merged into it, as for any subconfig. `ema: {}` switches it on
  with the class defaults, and leaves it as it is if it is on.
- **On the command line, a dataclass is switched on by a preset**,
  `ema=fast`: a dataclass is not given a dict there
  ([cli.md](cli.md#a-dataclass-is-not-given-a-dict)). To switch it on with
  values of one's own, a config file in `configs/ema/` sets at least one of
  them, and is selected; the error for `ema={decay:0.5}` says so. A
  TypedDict is switched on by a dict, which replaces it
  ([cli.md](cli.md#a-dict-for-a-typeddict-replaces-it)): `ema={decay:0.5}`
  sets only `decay`, also if it was on, and `ema={}` switches it on with
  nothing set.
- **A dotted key does not switch it on.** `ema.decay=0.5` while `ema` is off
  is an error that names the key and the class. For a dataclass, it shows
  `ema=NAME`; for a TypedDict, `ema={decay:0.5}`.
  Setting one value should not bring back all the others. The same holds
  for a dotted entry under `_defaults_`, such as `ema.inner: fast`.
- **The order on the command line**
  ([cli.md](cli.md#group-or-value-from-the-fields-type)). `ema=null` is
  applied with the selections, so `ema=null ema.decay=0.5` is an error in
  either order, and `ema=null`, `ema=fast` and, for a TypedDict,
  `ema={decay:0.5}` are the same key: the last one wins. The dotted keys
  apply after them, so `ema.every=4 ema={decay:0.5}` sets both, in either
  order.
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

### Optional groups

A group can be switched off as well: `None` goes outside the union that
carries the `Discriminator`.

```python
@dataclass
class Config:
    model: Model | None = None
    """Off unless a config selects an option."""
    decoder: Model | None = field(default_factory=Cnn)
    """On unless a config switches it off."""
```

#### How Hydra does it

What Hydra does with null for a group, tested with hydra-core:

| Where | Hydra |
| --- | --- |
| Defaults list, `- model: null` | Loads no file for the group |
| Body of a file, `model: null`, field typed as its base class (`model: ModelCfg = MISSING`) | "field 'model' is not Optional" |
| Body of a file, field typed `Any` | `None` |
| Command line, `model=null` | Crashes: "Config group override must be a string or a list" |

With the field typed `Optional[ModelCfg]`, and its default either `None` or
`Fcn(width=5)`:

| Written | Default `None` | Default `Fcn(width=5)` |
| --- | --- | --- |
| Defaults list, `- model: null` | `None` | `Fcn(width=5)` |
| `- override model: null`, after a config that selected `fcn` | `None` | `Fcn(width=5)` |
| Body of a file, `model: null`, after `fcn` was selected | `None` | `None` |

So `- model: null` in Hydra does not switch the group off: it selects
nothing, and the field keeps the default of the dataclass, which is `None`
only if the default is. `hydra-example` uses it for that, in
`- opt: null`, so that `opt=NAME` works on the command line without `+`
([cli.md](cli.md)). Only null in the body of a file switches the group off.

#### How pydantic reads it

Tested with pydantic 2.14, with `Model` the alias from
[Declaring a group](#declaring-a-group):

| Type | pydantic |
| --- | --- |
| `Model \| None`, or `Annotated[A \| B, Discriminator(pick)] \| None` | `None` gives `None`; the rest goes to the tagged union, as without `None` |
| `None` as an untagged member inside the union of `Discriminator(pick)` | Refused when the schema is built: "`Tag` not provided for choice" |
| `Annotated[None, Tag("none")]` inside it | `pick` would have to return `"none"` for `None`, which looks like an option |
| `None` inside a union of TypedDicts with `Discriminator("_option_")` | Accepted: pydantic takes `None` out of a union with a field discriminator |

With `None` outside, pydantic checks for `None` before the discriminator,
so `pick` never sees it, and the location of an error inside the group is
the same as without `None`.

#### Decided

- **The type.** `Model | None` (or `Optional[Model]`), where `Model` is a
  group of two or more options, of dataclasses or of TypedDicts, as an
  alias or inline: `Annotated[A | B, Discriminator(...)] | None`. Its
  default is `None` (off), an instance of an option (on), or there is
  none.
- **`None` inside the union is an error**, also for TypedDicts, where
  pydantic would accept it, so that there is one spelling. The error shows
  the spelling with `| None` outside.
- **One option is not enough.** `Annotated[A, Tag("a")] | None`, or a
  TypedDict with the key `_option_` and `| None`, is an error that points
  to the optional subconfig `A | None`: a name that switches on the only
  class is the [rejected idea](#rejected-ideas) of a name for "on".
- **`null` switches it off**: `model=null` on the command line, and
  `model: null` under `_defaults_` in a file. Being off is one of the
  choices for the group, so it is written where the options are selected,
  next to `model: cnn`. A dotted entry such as `outer.model: null` works
  too. What lower layers selected or set under the group is dropped,
  without a `DroppedValueWarning`. This is not what `- model: null` means
  in Hydra, where it selects nothing and keeps the default of the field
  ([above](#how-hydra-does-it-1)); nematocyte has no use for that, since a
  field is a group by its type, without an entry in the defaults list.
- **Among the values of a file, null is an error**, as for a subconfig
  ([above](#optional-subconfigs)): `model: null` outside `_defaults_`, and
  a key with nothing under it. Among the values, `model` holds the fields
  of the option that is selected. Hydra switches the group off there. In
  a dict for a TypedDict on the command line, null switches it off:
  `outer={model:null}`.
- **Null for anything else.** For a group that is not optional, null under
  `_defaults_` or on the command line is an error that shows the spelling
  with `| None`.
- **Selecting an option switches it on**: `model=cnn` on the command line,
  `model: cnn` under `_defaults_`, or a config in the group's directory.
  The node starts from the class defaults of the option, as for any
  selection ([Defaults](#defaults)). `model=null model=cnn` are the same
  key, and the last one wins.
- **A mapping does not switch it on.** With `model` off, a mapping names
  no class: `model.kernel=2`, `model={kernel:2}` on the command line (for
  a group of TypedDicts; for one of dataclasses it is an error anyway),
  `model: {kernel: 2}` in a file, and a dotted entry under `_defaults_`
  such as `model.norm: small` are errors. The error names the key and the
  options, and shows `model=fcn`. While the group is on, a mapping goes into
  the selected option, as for any group.
- **No default, nothing selected.** As for any group, it is an error that
  lists the options. It also says that `model=null` switches the group off.
- **Nothing else changes for the node**, as for an optional subconfig. An
  interpolation that refers to a key under a group that is off is an error.
  `nematocyte.to_dict` writes `null`, and `from_run_config` reads it back.
  `--help` shows `model=null` with the options, and `--cfg` shows
  `model: null` with "Selected: null" ([helptext.md](helptext.md)).
  `-m model=null,fcn,cnn` sweeps over the three. The validator checks the
  configs in the group's directory by selecting their options, as for any
  group that has no option selected.

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
- No option can have a field whose name is an option of the same group,
  since the directory with that name would be both the option's and the
  field's
  ([composition.md](composition.md#the-configs-of-a-group-can-be-grouped-by-option)).
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
