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

### Dict fields are open, and replaced as a whole

For a field typed `dict[...]`, such as
`optimizer_kwargs: dict[str, Any] | None`:

| Written | Hydra | nematocyte |
| --- | --- | --- |
| `opt.optimizer_kwargs={eps:1e-6}` | Error, asks for `+` | Sets the dict |
| ... when the field already holds `{amsgrad: true}` | Merges the two | Replaces it: only `eps` is left |
| `opt.optimizer_kwargs.eps=1e-6` | Error | Error; give the whole dict |

If possible, use `TypedDict` instead. Then you can set
`opt.optimizer_kwargs.eps=1e-6` on the commandline.

See [cli.md](design/cli.md).

### A subconfig cannot be set to a dict

`opt={lr:0.1}` is an error. Hydra merges the dict into the subconfig. Set the
fields one by one: `opt.lr=0.1`. See [cli.md](design/cli.md).

## Schema and config groups

### Groups are declared in the type of the field

There is no `ConfigStore`. A group is a field whose type is a union of named
dataclasses:

```python
@dataclass(kw_only=True)
class Config:
    model: (
        Annotated[FcnFactory, nematocyte.option("fcn")]
        | Annotated[CnnFactory, nematocyte.option("cnn")]
    ) = field(default_factory=FcnFactory)
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

### No optional subconfigs

A dataclass field cannot be `X | None`, and a group cannot have `None` as an
option. See [unions.md](design/unions.md).

### Selecting an option starts the node again

When an option of a group is selected, by a config file or on the command
line, the node starts from that option's class defaults. Whatever lower
layers set under it is dropped.

With a primary config `good_run.yaml` that sets `model.activation: selu`:

| Command | Hydra | nematocyte |
| --- | --- | --- |
| `--config-name good_run model=fcn` | An FCN with `activation: selu` | An FCN with its class defaults (`activation: gelu`) |
| `--config-name good_run model=cnn`, where `good_run` also sets `model.num_hidden`, which the CNN lacks | Error | A CNN with its class defaults |

To keep a value, set it again after the selection. nematocyte warns, with a
`nematocyte.DroppedValueWarning`, when a selection drops a value that a
lower layer set and nothing sets it again. A preset for a plain
subconfig, such as `opt=weight_decay_adamw`, is applied on top and keeps the
other values. See [composition.md](design/composition.md#layers).

## Config files

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
- `- opt: null` is not needed, and is an error.
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
- Any other directory holds primary configs. This includes a directory
  named after a field typed `dict[...]`, such as `configs/counts/` for
  `counts: dict[str, int]`: its files describe the root and have to write
  `counts:` at the top. In Hydra every directory is a config group, so
  `counts=heavy` puts the file's content into the dict and merges it with
  the default. In nematocyte, `+counts/heavy` applies the file on top of the
  primary config and replaces the dict as a whole. For presets that go
  straight into the field, use a `TypedDict` instead.

See [composition.md](design/composition.md).

### A primary config is optional

Without one, the config comes from the dataclass defaults. The entry point
does not have to name a default primary config, and `--config-name ''`
selects none. See [composition.md](design/composition.md).

## Command line

### No `++` or `~`, and `+` takes no `=`

Besides `key=value`, only `+NAME` remains. An unknown key under a dataclass
is still an error. `+` is not needed where Hydra asks for it to add a key:

| Hydra | nematocyte |
| --- | --- |
| `+opt=weight_decay_adamw` | `opt=weight_decay_adamw`, without a `_defaults_` entry for `opt` |
| `+opt.optimizer_kwargs={eps:1e-6}` | `opt.optimizer_kwargs={eps:1e-6}` |

Whether `key=name` selects a config or sets a value follows from the type of
the field: a dataclass or a group is a selection. See
[cli.md](design/cli.md).

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
in the meantime; in Hydra it is. See [multirun.md](design/multirun.md).

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
