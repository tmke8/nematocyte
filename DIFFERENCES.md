# Differences from Hydra

What works differently in nematocyte, for someone who knows Hydra. Each
entry says what changes in a config file or on the command line, and links
to the design document that explains why.

## Values

### Enums are read by value, not by name

With an enum such as

```python
class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"
```

a field of type `Color` is set to `BLUE` like this:

| Written | Hydra (OmegaConf 2.4) | nematocyte |
| --- | --- | --- |
| `color=BLUE`, the name | Accepted | Error: "Input should be 'red' or 'blue'" |
| `color=Color.BLUE` | Accepted | Error |
| `color=blue`, the value | Accepted | Accepted |
| Printed by `--cfg`, saved with the config | `BLUE` | `blue` |

For an enum with integer values (`LOW = 1`), nematocyte accepts `1` and
not `LOW`.

nematocyte checks values with pydantic, which reads enums by value
([deserializer.md](design/deserializer.md)). `--help` lists the values that
a field accepts, and `nematocyte.to_dict` writes the value, so that what it
gives is what a config file would hold ([api.md](design/api.md)).

To keep config files that use the names, give the members their names as
values:

```python
class Color(enum.StrEnum):
    RED = "RED"
    BLUE = "BLUE"
```

For a fixed set of strings, a `Literal["red", "blue"]` works as well, and
needs no class.

### A number is not a string

A number given for a `str` field is an error, in a config file and on the
command line alike. OmegaConf converts it.

| Written, for `name: str \| None` | Hydra | nematocyte |
| --- | --- | --- |
| `wandb.name=123` | `'123'` | Error |
| `wandb.name='123'` | `'123'` | `'123'` |
| `wandb.name=1.50` | `'1.5'` | Error |

The shell removes the quotes, so on the command line they have to be
protected: `"wandb.name='123'"`. See [cli.md](design/cli.md).

### A string is not a number or a bool

A quoted number is a string, and a string is rejected for an `int`, a
`float` or a `bool` field, as is a number for a `bool`. OmegaConf 2.4
converts them, with a warning that this is deprecated.

| Written | Hydra (OmegaConf 2.4) | nematocyte |
| --- | --- | --- |
| `seed='123'`, for `int` | `123`, with a `FutureWarning` | Error |
| `lr='1e-3'`, for `float` | `0.001`, with a `FutureWarning` | Error |
| `flag='false'` or `flag=0`, for `bool` | `False`, with a `FutureWarning` | Error |

As in OmegaConf, `true` and `1.0` are rejected for an `int`. nematocyte
validates in pydantic's strict mode on JSON input; see
[deserializer.md](design/deserializer.md#strict-validation-through-json).

### Config files are read as YAML 1.2

Hydra reads config files with PyYAML, which implements YAML 1.1. A file
carried over from Hydra can change its meaning:

| Text | Hydra | nematocyte |
| --- | --- | --- |
| `yes`, `on`, `No` | bool | string |
| `1e-3` | string | float |
| `010` | `8` | `10` |
| `1:30` | `90` | string |
| A key given twice | The second one wins | Error |

On the command line nothing changes; there, `yes` was a string and `1e-3` a
float before too. See [composition.md](design/composition.md).

### Config files can also be JSON5

A config file can be written in JSON5, as `cnn.json5` in place of
`cnn.yaml`, with the extra `nematocyte[json5]` installed. Hydra reads only
YAML. A name that has both a YAML and a JSON5 file, such as `cnn.yaml` and
`cnn.json5` in the same directory, is an error, as are `cnn.yaml` and
`cnn.yml`. See [composition.md](design/composition.md#config-files-can-also-be-json5).

### `???` does not mark a required value

A dataclass field without a default is required, and a missing value is
reported with its path. In a config file, `???` is an ordinary string.
In order to have fields without defaults after fields with defaults,
use `@dataclass(kw_only=True)`.

See [deserializer.md](design/deserializer.md).

### A default has to keep its type in a config file

A default that would lose its type when written in a config file is an
error. This happens where the type of the field does not say what to build:

| Field and default | Hydra (OmegaConf 2.4) | nematocyte |
| --- | --- | --- |
| `color: Any = Color.RED` | `Color.RED` | Error |
| `root: Any = Path("/data")` | `Path("/data")` | Error |
| `opt: Any = field(default_factory=Adam)` | `Adam(lr=0.1)`, after `OmegaConf.to_object` | Error |
| `sizes: Any = (1, 2)` | `(1, 2)` | `[1, 2]` |

Give the field the type it holds (`color: Color`), or, for a choice between
config classes, make it a group ([unions.md](design/unions.md)). See
[deserializer.md](design/deserializer.md).

### Dict fields are open, and replaced on the command line

For a field typed `dict[...]`, such as
`optimizer_kwargs: dict[str, Any] | None`:

| Written | Hydra | nematocyte |
| --- | --- | --- |
| `opt.optimizer_kwargs={eps:1e-6}` | Error, asks for `+` | Sets the dict |
| ... when the field already holds `{amsgrad: true}` | Merges the two | Replaces it: only `eps` is left |
| `opt.optimizer_kwargs.eps=1e-6`, when the field holds a dict with `eps` | Sets `eps` | Error; give the whole dict |

As in Hydra, a mapping in a config file is merged deeply into a mapping,
whatever the type of the field. To replace it in a file instead, add
`_replace_: true` to the mapping
([below](#_replace_-true-replaces-instead-of-merging)).

If possible, use `TypedDict` instead. Then you can set
`opt.optimizer_kwargs.eps=1e-6` on the commandline.

See [cli.md](design/cli.md).

### Unknown keys inside a value are ignored

A config class inside a value, such as an item of `items: list[Point]`, is
not a node of the config, and a key that is not one of its fields is
dropped, so that the field takes its default:

| Written | Hydra | nematocyte |
| --- | --- | --- |
| `items=[{x:1,yy:2}]` | Error | `[Point(x=1, y=0)]` |

To make it an error, opt in through the pydantic config of the class:

```python
from pydantic import ConfigDict, with_config


@with_config(ConfigDict(extra="forbid"))
@dataclass
class Point:
    x: int = 0
    y: int = 0
```

A key that is not a field of a subconfig or an option of a group is always
an error, with the file or override that set it. See
[deserializer.md](design/deserializer.md#unknown-keys).

## Schema and config groups

### Groups are declared in the type of the field

There is no `ConfigStore`. A group is a field whose type is a union of
dataclasses, each named with pydantic's `Tag`:

```python
@dataclass(kw_only=True)
class Config:
    model: Annotated[
        Annotated[FcnFactory, Tag("fcn")] | Annotated[CnnFactory, Tag("cnn")],
        Discriminator(nematocyte.pick),
    ] = field(default_factory=FcnFactory)
```

- Every option is listed where the field is declared. Another module cannot
  add one later.
- The default option can be the field's default, as above. In Hydra a group
  field gets no default, and the primary config chooses one.
- With such a default, `model=cnn` works. In Hydra a default instance fixes
  the class, and selecting another option is an error.

YAML files in `configs/model/` add options, as in Hydra. See
[unions.md](design/unions.md).

### No `_target_` and no `instantiate`

The class of every node comes from the schema. A config file never names
Python code, and `_target_` is an unknown key like any other. A program that
wants to name a class in its config keeps the import path in a `str` field
and imports it itself. See [unions.md](design/unions.md).

### Switched off and on under `_defaults_` in a file

An optional subconfig, `ema: EmaCfg | None`, and an optional group,
`model: Model | None` with `None` outside the union of the group, are
switched off under `_defaults_` in a file, next to the selections:

```yaml
_defaults_:
  model: null
```

An optional subconfig is switched on in the same place, with `_some_`,
which Hydra does not have. On the command line it is `ema=_some_`. If the
subconfig is off, it starts from the defaults of its class; if it is on, it
stays as it is:

```yaml
_defaults_:
  ema: _some_

ema:
  decay: 0.5
```

In Hydra, `- model: null` in the defaults list means something else: it
selects nothing, and the field keeps the default of the dataclass, so a
default of `Fcn()` stays on. Hydra switches the group off with
`model: null` or `ema: null` in the body of a file, for a field typed
`Optional[...]`, and gives an error there for any other field ("field
'model' is not Optional").

In nematocyte, null in the body of a file is an error for anything that
holds a mapping and does not take `None`: any subconfig or group, and a
dict without `| None`. The error shows what may have been meant: the entry
under `_defaults_`, leaving out the key, or, for a TypedDict or a dict,
`{_replace_: true}`, which makes it empty.

| In the body of a file | Hydra | nematocyte |
| --- | --- | --- |
| `ema: null`, for `ema: EmaCfg \| None` | `None` | Error, which shows the entry under `_defaults_` |
| `ema:` with only comments under it | `None` | The same error |
| `plain: null`, for `plain: Plain` | Error | Error |
| `counts: null`, for `counts: dict[str, int]` | Error | Error, which shows `counts: {_replace_: true}` |
| `maybe: null`, for `maybe: dict[str, int] \| None` | `None` | `None` |
| `ema: {decay: 0.5}` or `ema: {}`, with `ema` off | Switches it on | Error, which shows `ema: _some_` under `_defaults_` |

The Hydra column was tested by merging into the structured config with
OmegaConf, as Hydra does.

On the command line, `model=null` switches it off, where Hydra crashes
("Config group override must be a string or a list"). A group is switched
on by selecting an option, as in `model=cnn`, and a subconfig by `_some_`
or a named config. A mapping among the values does not switch either on,
and neither does a dotted key: `model.kernel=2` while the group is off is
an error that points to `model=NAME`, and `ema.decay=0.5` while `ema` is
off one that points to `ema=_some_ ema.decay=0.5`. On the command line, a dataclass is not given a dict
([below](#a-dict-replaces-and-a-dataclass-takes-none)). See
[unions.md](design/unions.md#optional-subconfigs).

### A hand-written `__init__` is not called

pydantic builds the config objects: it sets the fields and calls
`__post_init__`, but not an `__init__` written in the class. OmegaConf's
`to_object` calls the constructor. Put that work in `__post_init__`. See
[deserializer.md](design/deserializer.md#details-of-building).

### Selecting an option or a preset starts the node again

When an option of a group, or a preset of a plain subconfig, is selected,
by a config file or on the command line, the node starts from the class
defaults: of the option, or of the subconfig's class. Whatever lower layers
set under it is dropped, and so is the default instance of the field.

With a primary config `good_run.yaml` that sets `model.activation: selu` and
`opt.weight_decay: 0.001`:

| Command | Hydra | nematocyte |
| --- | --- | --- |
| `--config-name good_run model=fcn` | An FCN with `activation: selu` | An FCN with its class defaults (`activation: gelu`) |
| `--config-name good_run model=cnn`, where `good_run` also sets `model.num_hidden`, which the CNN lacks | Error | A CNN with its class defaults |
| `--config-name good_run opt=only_lr`, a preset that sets only `lr` | `weight_decay: 0.001` | `weight_decay` at its class default |
| `opt=only_lr`, where the field is `opt: Opt = field(default_factory=lambda: Opt(wd=0.1))` | `wd: 0.1` | `wd` at its class default |

`opt=NAME` means the same whether `opt` is a group or a plain subconfig.
To keep a value, set it again after the selection. nematocyte warns, with a
`nematocyte.DroppedValueWarning`, when a selection drops a value that a
lower layer set and nothing sets it again. `opt=_some_`, which switches on
an optional subconfig, is not a selection, and keeps what was set. See
[composition.md](design/composition.md#layers).

## Config files

### `_replace_: true` replaces instead of merging

A mapping in a config file is merged into what was set before, in Hydra and
in nematocyte. In Hydra, a file cannot take out a key that a lower layer
set: `key: null` sets `None`, and `key: ???` leaves the value as it was. In
nematocyte, a mapping for a TypedDict, or a mapping in a value such as a
`dict[...]`, can replace what was there instead:

```yaml
opt:
  _replace_: true
  lr: 0.001
```

`opt` is then `{"lr": 0.001}`, whatever the field default, the primary
config or a preset set under it. On the command line, a dict replaces
anyway, so the same is `opt={lr:0.001}`
([below](#a-dict-replaces-and-a-dataclass-takes-none)); `~opt.wd` takes out
a single key. A
dataclass cannot be replaced this way. A dict that is to hold a key
`_replace_` of its own cannot be written. See
[composition.md](design/composition.md#_replace_-true-replaces-instead-of-merging).

### `_defaults_` is a mapping

The key is called `_defaults_`, not `defaults`, so that `defaults` can be the
name of a field. Each key says where a config is loaded; the order of the
entries has no meaning. `_self_` names the config that the file extends.

```yaml
# Hydra
defaults:
  - base_config
  - override dm: cmnist
  - override model: cnn
  - _self_
```

```yaml
# nematocyte
_defaults_:
  dm: cmnist
  model: cnn
```

```yaml
# configs/dm/celeba_male_smiling.yaml
_defaults_:
  _self_: celeba
```

- What is applied before what follows from the keys: first the config named
  by `_self_`, then the other entries, shortest path first, then the values
  in the file. Hydra applies the list in order.
- There is no `override` keyword.
- A key can be a dotted path: `model.norm: my_batchnorm`.
- `- opt: null`, which Hydra needs so that `opt=NAME` works without `+`, is
  not needed, and is an error. Under `_defaults_`, null only switches off
  an optional subconfig or group, `model: null`, which is not what it
  means in Hydra, and `_some_` switches on an optional subconfig
  ([above](#switched-off-and-on-under-_defaults_-in-a-file)).
- A key `defaults`, as in Hydra, is an error that suggests `_defaults_`,
  unless the node has a field of that name. A list is an error that shows
  the mapping form.
- A primary config can extend another one with `_self_: good_run`.

See [composition.md](design/composition.md).

### `_doc_` describes a file

A key `_doc_` at the top level of a config file describes the file.
`--help` shows the first paragraph next to the name of the file, and also
lists the primary configs with their docs. `_doc_` is not a config value,
and no config class can have a field of that name. In Hydra, `--help`
lists the files by name only, and `_doc_` would be an ordinary key. See
[helptext.md](design/helptext.md#docs-of-config-files).

### The schema is always applied

Every primary config is checked against the root dataclass that is given to
the entry point. There is no `- base_config` entry; in Hydra a primary config
without it is not checked at all. See [composition.md](design/composition.md).

### No `# @package` and no `group@package`

A file's content always goes to the place in the tree that matches where the
file is. A primary config in a subdirectory, such as
`configs/experiments/cmnist_cnn.yaml`, needs no `# @package _global_` header,
and its `_defaults_` need no leading slash (`dm: cmnist`, not `/dm: cmnist`).
See [composition.md](design/composition.md).

### Where a file is says what it is

- A directory named after a dataclass field or group, such as
  `configs/model/`, holds named configs for that field and nothing else.
  `--config-name model/small` is an error.
- A file at the top named after such a field, such as `configs/model.yaml`,
  is an error.
- A directory named after a field that is a plain value, such as
  `configs/seed/`, or `configs/counts/` for `counts: dict[str, int]`, is an
  error. In Hydra every directory is a config group, so `counts=heavy` puts
  the file's content into the dict and merges it with the default. In
  nematocyte, a primary config such as `configs/counts_heavy.yaml`, which
  writes `counts:` at the top, can be applied with `+counts_heavy`; it is
  merged into the dict, as in Hydra. For presets that go straight into the
  field, use a `TypedDict` instead; such a preset replaces what the field
  held ([above](#selecting-an-option-or-a-preset-starts-the-node-again)).
- In a group's directory, a subdirectory named after an option, such as
  `configs/dm/celeba/`, holds further configs for the group, selected as
  `dm=celeba/male_blond`. Each of them extends that option, without
  `_self_`; a `_self_` that leads to another option is an error. In Hydra
  a name with a slash loads a file from any subdirectory, and the file can
  extend anything; in nematocyte a subdirectory that is neither an option
  nor a field is an error, and a name with a slash is an error for a plain
  subconfig.
- Any other directory holds primary configs.

See [composition.md](design/composition.md).

### A primary config is optional

Without one, the config comes from the dataclass defaults. The entry point
does not have to name a default primary config, and `--config-name ''`
selects none. See [composition.md](design/composition.md).

## Command line

### `+` only adds a primary config

In Hydra, `+` adds a key, or a config group that is not in the defaults
list. In nematocyte that is not needed:

| Hydra | nematocyte |
| --- | --- |
| `+opt=weight_decay_adamw` | `opt=weight_decay_adamw`, without a `_defaults_` entry for `opt` |
| `+opt.optimizer_kwargs={eps:1e-6}` | `opt.optimizer_kwargs={eps:1e-6}` |

Whether `key=name` selects a config or sets a value follows from the type of
the field: a dataclass or a group is a selection. The one use of `+` that
is left is `+NAME`, without `=`, which applies a primary config on top of
the tree ([below](#a-preset-for-the-whole-config-is-presetsdebug)).
`+key=value` is an error that shows both forms that may have been meant.
See [cli.md](design/cli.md).

### A dict replaces, and a dataclass takes none

In Hydra, a dict on the command line is merged into what the key holds, as
a mapping in a file is. In nematocyte, a dict on the command line replaces
it, and is only for a TypedDict and for a value:

| Written | Hydra | nematocyte |
| --- | --- | --- |
| `wandb={name:foo}`, for a TypedDict `wandb` | Merges `name` into it | Replaces it: only `name` is set |
| `optimizer={beta1:0.5}`, for a group of TypedDicts | Merges into the selected option | Replaces what is set in the selected option, which stays selected |
| `opt={lr:0.1}`, for a dataclass `opt` | Merges into it | Error, which shows `opt.lr=0.1` |
| `model={num_hidden:2}`, for a group of dataclasses | Merges into the selected option | Error, which shows `model.num_hidden=2` |
| `ema={decay:0.5}`, for `ema: EmaCfg \| None = None`, a dataclass | Switches it on | Error, which shows `ema=_some_ ema.decay=0.5` |

To change one key of a TypedDict and keep the others, use a dotted key,
`wandb.name=foo`, as for a dataclass. A dict is the same key as `wandb=NAME`
and `wandb=null`, so the last one wins, and dotted keys under it are applied
after it, in either order. To switch on an optional dataclass with values
of your own, use `_some_` and dotted keys: `ema=_some_ ema.decay=0.5`. See [cli.md](design/cli.md#a-dict-for-a-typeddict-replaces-it).

### `~` only leaves out a TypedDict key

In Hydra, `~key` deletes any key, and `~key=value` deletes it only if it
has that value. In nematocyte, `~KEY` only leaves out a `NotRequired` key of
a TypedDict, so that it is not in the dict, even if a default or a config
file set it. Leaving out a key that nothing set is not an error.

`~key=value` is an error, and so is `~KEY` on a required key, on a field of
a dataclass, and on a subconfig or a group. Such a field is set to `null`
instead, if it takes `None`, and the error says so. This includes an
optional subconfig: `~ema` does not switch it off, but is an error that
points to `ema=null`. It does not switch it off in Hydra either, where the
key is deleted and the default of the dataclass comes back, without a
warning:

| Written, where the default is `EmaCfg()` | Hydra | nematocyte |
| --- | --- | --- |
| `~ema` | Still on | Error, which points to `ema=null` |
| `ema=null` | Off | Off |

See [cli.md](design/cli.md#key-leaves-out-an-optional-key-of-a-typeddict).

### No `++`

In Hydra, `++key=value` sets a key whether or not it is in the config. In
nematocyte, `key=value` sets any key that the schema has, and an unknown
key is an error, with or without a prefix. `++key=value` is an error that
says to write `key=value`.

### A preset for the whole config is `+presets/debug`

Hydra's `+presets=debug`, with `# @package _global_` in
`configs/presets/debug.yaml`, applies a file on top of the experiment. In
nematocyte the file needs no header, and is named as `--config-name` names
it, with a slash and no `=`:

```sh
python main.py --config-name good_run +presets/debug
```

This composes as if `debug.yaml` had `_self_: good_run`. The values on the
command line still win, as in Hydra. Several of them are applied in the
order given. `+NAME` only exists on the command line; in a file, `_self_`
says what it builds on. See [cli.md](design/cli.md).

### No slash in keys, and `hydra` is `nematocyte`

A selection is written with dots at every depth, and nematocyte's own
settings are under the key `nematocyte`:

| Hydra | nematocyte |
| --- | --- |
| `hydra/launcher=slurm_kyiv` | `nematocyte.launcher=slurm_kyiv` |
| `hydra/sweeper=fcn_params` | `nematocyte.sweeper=fcn_params` |
| `hydra.launcher.partition=general` | `nematocyte.launcher.partition=general` |
| `configs/hydra/launcher/` | `configs/nematocyte/launcher/` |

In Hydra, `hydra.launcher=slurm_kyiv` silently sets the string; in
nematocyte it is the selection. See [cli.md](design/cli.md).

### Less grammar

Not supported: `key@package`, list indices (`a[0]`), the casts `int()`,
`float()`, `str()`, `bool()` and `json_str()`, and `glob()`, `sort()`,
`shuffle()` and `extend_list()`. See [cli.md](design/cli.md).

### Fewer flags

`--help`, `--multirun`, `--config-name` and `--cfg {job,all}` are the only
flags, with Hydra's short forms. Among those that are missing:

- `--config-path` and `--config-dir`
- `--cfg hydra`, `--resolve` and `--package`
- `--info`, `--hydra-help` and `--version`
- `--shell-completion`
- `--experimental-rerun`

See [cli.md](design/cli.md).

## Interpolation

### `now` is the only resolver

`${a.b}` and `${now:...}` work. There are no custom resolvers, no `oc.env`
and the other `oc.*` resolvers, and no `${hydra:...}`. See
[interpolation.md](design/interpolation.md).

### `now` has the same value in every job

In a multirun, Hydra gives every job the same `${now:%Y-%m-%d}` and
`${now:%H-%M-%S}`, but computes any other pattern again in each job.
nematocyte computes the time once, for all jobs. See
[interpolation.md](design/interpolation.md).

### Not supported in references

- Relative references, `${.sibling}`
- Nested interpolations, `${a.${b}}`
- A reference into a list
- An escaped `\${`, for a literal `${`
- An interpolation as a dataclass default, `name: str = "${seed}"`, which is
  kept as written. Set it in a config file, or derive it in `__post_init__`.

See [interpolation.md](design/interpolation.md).

## Multirun

### Launchers and sweepers are built in

There are no plugins. submitit and Optuna are installed as extras of
nematocyte (`nematocyte[submitit]`, `nematocyte[optuna]`), and the options
keep Hydra's names: `basic` and `submitit_slurm` for the launcher, `basic` and
`optuna` for the sweeper. A launcher file extends one with `_self_`:

```yaml
# configs/nematocyte/launcher/slurm_kyiv.yaml
_defaults_:
  _self_: submitit_slurm

partition: general
```

Missing: the `submitit_local` launcher, `max_batch_size` of the basic
sweeper, `custom_search_space` of the Optuna sweeper, and every Optuna
sampler but `tpe` and `random`. See [multirun.md](design/multirun.md).

### A launcher or sweeper without `--multirun` is an error

Hydra ignores `hydra/launcher=...` without `--multirun`, and runs the job on
the machine it was started on. See [multirun.md](design/multirun.md).

### A job does not read the config files

The config of every job is composed and checked before anything runs. A job
that waits in the SLURM queue is not affected by edits to the config files
in the meantime; in Hydra it is. The config files are read once, so a later
batch of Optuna trials is not affected by such edits either. See
[multirun.md](design/multirun.md).

### nematocyte's settings cannot be swept

`nematocyte.launcher=a,b` is an error. See
[multirun.md](design/multirun.md).

## Output directories

### One root, and a numbered directory for every run

```text
outputs/2026-11-16/
    16-49-10/          a single run
        0/
    17-02-33/          a multirun
        0/
        1/
```

There is no `multirun/` root. A single run is a run with one job, `0`. Two
programs started in the same second get separate directories. See
[outputs.md](design/outputs.md).

### The directory is created when it is asked for

`nematocyte.output_dir()` returns it, and creates it. A run that never asks
leaves nothing behind. `${hydra:runtime.output_dir}` in a config file has no
counterpart; build the path in Python. See [outputs.md](design/outputs.md).

### One setting

`nematocyte.output_dir` replaces `hydra.run.dir`, `hydra.sweep.dir` and
`hydra.sweep.subdir`. The directory of a job is always its number. The
working directory is never changed; `hydra.job.chdir` has no counterpart.
See [outputs.md](design/outputs.md).

### What is saved

`.nematocyte/config.json` and `.nematocyte/overrides.json`, in place of
`.hydra/config.yaml`, `.hydra/overrides.yaml` and `.hydra/hydra.yaml`. The
saved config is resolved, and records the option of each group under
`_option_`. It cannot be passed back as a primary config, but
`nematocyte.from_run_config(Config, data)` turns it back into the config
object of the run, with the schema applied. Hydra's saved config only loads
as plain dicts. See [outputs.md](design/outputs.md).

## Logging

### `logging` is not set up unless asked for

Hydra configures `logging` before each job. nematocyte does so only when the
program calls `nematocyte.configure_logging()`, and then:

- the log file is `run.log`, not `NAME.log`;
- the level is set with `nematocyte.log=debug`, in place of
  `hydra.verbose=true`. There is no level for named loggers only
  (`hydra.verbose=[train]`).

The setup cannot be changed from the config, as `hydra/job_logging` does. A
program that sets up `logging` itself reads the level with
`nematocyte.log_level()`. See [logging.md](design/logging.md).

## Python API

### The function receives the dataclass

```python
@nematocyte.main(Config, config_path="configs")
def main(config: Config) -> float: ...
```

The function gets an instance of `Config`, not a `DictConfig`, so there is
no `OmegaConf.to_object`. The root dataclass is passed to the decorator, and
there is no `version_base`. `config_path` keeps Hydra's name and meaning.

The decorated function takes an optional list of arguments, in place of the
command line, and returns `None`. See [api.md](design/api.md).

### Composing in a test or notebook

```python
config = nematocyte.compose(Config, "configs", overrides=["model=cnn"])
```

There is no `initialize` and no `GlobalHydra`. `compose` returns the
dataclass instance. See [api.md](design/api.md).

### No `HydraConfig`

There is no `HydraConfig.get()`. In its place:

| Hydra | nematocyte |
| --- | --- |
| `HydraConfig.get().runtime.output_dir` | `nematocyte.output_dir()` |
| `HydraConfig.get().job.num` | `nematocyte.job_num()` |
| `HydraConfig.get().verbose` | `nematocyte.log_level()` |

The rest of Hydra's runtime information, such as the job's name, id and
overrides, has no counterpart. See [api.md](design/api.md).

### A plain dict for logging

`nematocyte.to_dict(config)` replaces `OmegaConf.to_container`. It records
the option of each group under `_option_`, and writes enums as their values
and paths as strings. Inside a run, `nematocyte.run_config()` returns the
same from the job's resolved tree, as it is saved in `config.json`. See
[api.md](design/api.md).
