# Unions and config groups

How the concrete class is chosen for a field that can hold one of several
config classes.

## Declaring a group

A group is declared in the type of the field, as a union whose members carry
their names. This is how tyro does it.

```python
@dataclass
class Config:
    data: (
        Annotated[WikiDataConfig, nematocyte.option("wiki")]
        | Annotated[HelpSteerDataConfig, nematocyte.option("helpsteer")]
    )
```

The marker is called `option`, the word Hydra uses for a member of a group
(`group=option`, "Available options in 'model'").

This replaces the registry in `hydra-example` (`CONFIG_GROUPS`, passed to
`register_hydra_config`), where the field is typed as an abstract base and the
allowed classes are listed somewhere else.

A prototype on Python 3.15 read the names and classes back with
`typing.get_type_hints(Config, include_extras=True)`. It also covered:

- **A reusable alias.**
  `type Model = Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")]`
  can be used for several fields. The alias has to be unwrapped through
  `__value__`.
- **No common base class.** The members only have to be dataclasses. A base
  class is still possible, and useful for sharing an interface.
- **Telling a group from an ordinary union.** `str | None` has no named
  dataclass members and stays a value.

Things to get right:

- `include_extras=True` is required. Without it `get_type_hints` strips
  `Annotated`, and the names are gone.
- The union never reaches pydantic. nematocyte handles group fields itself
  and gives pydantic the fields of the chosen class (see
  [deserializer.md](deserializer.md)).

The names in the annotations are the built-in options. YAML files in the
group's directory add further options on top, as before (see
[composition.md](composition.md)).

### Every member needs a name

A union of dataclasses with an unnamed member, such as a plain
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

### No optional dataclass fields, for now

`None` is not allowed as a member of a group, and a plain dataclass field
cannot be `X | None`. `hydra-example` has neither.

For when this is added, the points that need an answer:

- Switching the field off. `scheduler=null` matches how `wandb.name=null`
  gives `None`. In Hydra, `- opt: null` in a defaults list means something
  else ("load no file, keep the dataclass defaults"), but nematocyte does not
  need that entry, so `null` is free.
- Setting a field while it is off. With `scheduler` at `None`,
  `scheduler.t_max=10` names no class.
- A single optional dataclass, `ema: EmaCfg | None = None`. There is nothing
  to select by name, and a dict value (`ema={}`) is not allowed. Under the
  rule that every member needs a name it would be
  `Annotated[EmaCfg, option("on")] | None`, switched on with `ema=on`.

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
file's `defaults` ([composition.md](composition.md)). A file in a group's
directory has to say which option it extends, and the chain ends at an
option from the annotations:

```yaml
# configs/decoder/small_cnn.yaml
defaults:
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

- `defaults` is reserved, like `_self_`: no config dataclass can have a
  field called `defaults`, because that key means something else at the top
  of every config file.
- A union that mixes a dataclass with something else, such as `Sub | int`,
  is an error. A union with a dataclass in it is a group, and all of its
  members have to be named dataclasses.
- A single `Annotated[X, option("x")]` is a group with one option.
- A class can be in a group only once, and an option name only once.
- An option name starts with a letter or an underscore, followed by letters,
  digits, `_` or `-`, so that it can be typed unquoted.
- The default of a plain dataclass field has to be an instance of exactly
  the declared class; an instance of a subclass is an error, since the node
  is built as the declared class. The default of a group field has to be an
  instance of exactly one of the option classes.

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
- Because the data carries no tag, a dumped config is not self-describing.
  Whatever is written out for reproducibility has to record the choices next
  to the values, e.g. in `defaults`, so that it can be loaded again.
  The dict for logging records each choice under the key `_option_`
  ([api.md](api.md)).

How the chosen class is then handed to the validator is covered in
[deserializer.md](deserializer.md): pydantic cannot take it directly, so
nematocyte walks the dataclass nodes itself.

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
