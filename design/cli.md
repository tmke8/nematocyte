# Command line

How command-line arguments are parsed and applied to the config.

## Goal

Keep Hydra's command-line syntax as it is, with one removal: the `+key=value`,
`++key=value` and `~key` overrides. They are confusing, and apart from one
use of `+` they are not used in `hydra-example`. That one use, adding a
preset such as `+presets=debug` on top of an experiment, keeps the `+`, in a
simpler form: `+presets/debug`.

`~key` came back later, in a narrow form: it leaves out an optional key of a
TypedDict ([below](#key-leaves-out-an-optional-key-of-a-typeddict)).

## How Hydra does it

Read from the source of Hydra 1.4.0rc3 and confirmed by running about 60
sample overrides through its parser.

### 1. Flags

`--help`, `--multirun`, `--config-name` and the other flags are ordinary
`argparse` options (`hydra/_internal/utils.py`). Every positional argument is
collected, unparsed, into one list of overrides.

### 2. Overrides

Each override string is parsed by an ANTLR grammar
(`hydra/grammar/OverrideLexer.g4` and `OverrideParser.g4`), compiled to
Python. The ANTLR runtime is a copy vendored inside OmegaConf. The lexer has
two modes: key mode up to the `=`, then value mode.

The value is typed by the grammar alone, without looking at the schema:

| Argument | Parsed value |
| --- | --- |
| `seed=123`, `seed=1_000` | `int` |
| `opt.lr=1e-3`, `opt.lr=.5`, `opt.lr=inf` | `float` |
| `flag=true`, `flag=TRUE` | `bool` (`yes` and `on` stay strings) |
| `wandb.name=null` | `None` |
| `wandb.name='null'`, `wandb.name="123"` | string; quotes keep it a string |
| `wandb.name=my run`, `dm.root=/srv/data`, `seed=007` | string: anything unquoted that is none of the above |
| `wandb.name=` | empty string |
| `wandb.tags=[a,b]`, `dm.colors=[1,[2,3]]` | list, nestable |
| `dm.label_map={1:2}`, `x={a:{b:c}}` | dict, nestable |
| `wandb.group=${now:%Y-%m-%d}-x` | string, left for interpolation |
| `seed=42,43` | choice sweep |
| `seed=range(1,4)`, `seed=choice(1,2)`, `opt.lr=interval(1e-5,1e-3)`, `opt.lr=tag(log, interval(1e-5, 1e-3))` | sweeps, as function calls from a fixed registry |

Unlike YAML 1.1, `1e-3` is a float here.

### 3. Routing and conversion

- **Group or value.** This is not decided by syntax. Hydra asks its config
  repository whether a group with that name exists. If so, `model=cnn` goes
  into the defaults list; otherwise it is a value override.
- **Conversion.** Value overrides are applied after composition with
  `OmegaConf.update(cfg, key, value, merge=True)`. OmegaConf converts the
  value to the field type at that point.
- **Unknown keys.** The composed config is put into struct mode, which forbids
  new keys everywhere. `+` is the way around it.
- **Sweeps.** A sweep value outside `--multirun` is an error ("Ambiguous
  value for argument ...").

## Decided

### No `++`, `+` only adds a primary config, and `~` only leaves out a key

The grammar's top-level rule keeps `key=value` and `key=`, and adds
`+NAME`, with no `=` (next section), and `~KEY`, with no `=`
([below](#key-leaves-out-an-optional-key-of-a-typeddict)). `+key=value`,
`++key=value` and `~key=value` are errors. For `+presets=debug`, the error
shows both forms that may have been meant: `+presets/debug` and
`presets=debug`.

### `+NAME` applies a primary config on top

`+NAME` names a primary config, as `--config-name` takes it, and applies it
on top of the primary config. `--config-name good_run +presets/debug`
composes as if `configs/presets/debug.yaml` had `_self_: good_run`. The
command line's selections and values come after that, as always.

In Hydra this is `+presets=debug`, which appends the group `presets` to the
defaults list, and the file needs `# @package _global_` so that its content
lands at the root. In nematocyte `configs/presets/` matches no field, so its
files are primary configs already ([composition.md](composition.md)), and
the `=` would only split the name of the file in two. Observed in
`hydra-example`, with a `presets/debug.yaml` that sets `opt.epochs: 1`:

| Command | Hydra |
| --- | --- |
| `+presets=debug` | `epochs: 1` |
| `+presets=debug opt.epochs=5`, or the other way round | `epochs: 5`; the command line wins |
| `presets=debug` | "Could not override 'presets'. No match in the defaults list." |

The rules:

- The primary config goes at the bottom of the `_self_` chain of the added
  file. If `debug.yaml` has `_self_: presets/base`, the chain is `good_run`,
  then `presets/base`, then `presets/debug`. `_self_` means the same in
  every primary config, and presets can extend each other.
- Values on the command line win over an added config, whatever the order
  of the arguments. A selection on the command line replaces the node, as
  always ([composition.md](composition.md#layers)).
- Several added configs are applied in the order in which they are given,
  so the later one wins. A config that is given twice counts once, at its
  last place.
- `+NAME` has to name a primary config. `+model/single_linear_layer` is an
  error that points to `model=single_linear_layer`.
- `+NAME` exists only on the command line (and in the overrides of
  `nematocyte.compose`). A file says what it builds on with `_self_`.
- Without a primary config (`--config-name ''`), the added config applies on
  top of the dataclass defaults.

### `~KEY` leaves out an optional key of a TypedDict

A `NotRequired` key of a TypedDict has a state that no `key=value` reaches:
not being in the dict at all ([deserializer.md](deserializer.md#typeddict-configs)).
`wandb.name=null` puts `None` into the dict, which is not the same for a
program that passes the dict on as keyword arguments. Once the default of
the field, a primary config or a preset has set the key, `~KEY` is the way
to take it out again:

```sh
python main.py +presets/debug ~wandb.name
```

In Hydra, `~key` deletes any key from the config, and `~key=value` deletes
it only if it has that value.

The rules:

- `~KEY` leaves the key out of the dict that the program gets, as if nothing
  had set it, whatever set it before: the default instance of the field, the
  primary config, an added config, or a selected config file.
- KEY has to be a `NotRequired` key of a TypedDict (`total=False` and
  inheritance count, as everywhere). Anything else is an error that names
  the argument, the key and the class: a `Required` key; a field of a
  dataclass, where the error points to `KEY=null` if the field takes
  `None`; a subconfig or a group, which cannot be `NotRequired`
  ([deserializer.md](deserializer.md#typeddict-configs)), where the error
  points to `KEY=null` if the subconfig is optional
  ([unions.md](unions.md#optional-subconfigs)); and an unknown key, as for
  `KEY=value`.
- `~KEY=value` is an error, which shows the two forms that may have been
  meant, `~KEY` and `KEY=value`. Hydra's conditional delete is not
  supported.
- Leaving out a key that nothing set is allowed, and does nothing. Hydra
  makes it an error; here the schema already says that the key exists, and
  whether a preset or a job of a sweep sets it is not always known.
- `~KEY` and `KEY=value` are the same key: the last one wins, as for a key
  that is given twice. `~KEY` is applied with the values, after all
  selections, so `optimizer=adam ~optimizer.eps` works in either order.
- `~KEY` cannot be swept: there is no sweep between "left out" and values.
  With `--multirun` it holds for every job, and it takes the key out of the
  `params` of the sweeper, as any value on the command line does
  ([multirun.md](multirun.md)).
- `~KEY` exists only on the command line and in the overrides of
  `nematocyte.compose`, as `+NAME` does. A config file cannot leave out a
  single key: `key: null` is a value. It can replace the whole TypedDict
  with `_replace_: true`, and set again the keys that are to stay
  ([composition.md](composition.md#_replace_-true-replaces-instead-of-merging)).
- An interpolation that refers to a key that was left out is an error, as
  for a key that nothing set
  ([interpolation.md](interpolation.md)).
- `--help` shows the key as not set (`key=` with `(not set)`), and `--cfg`
  as the comment `# key: not set` ([helptext.md](helptext.md)).

### Dict-typed fields are open

Hydra's struct mode also closes fields whose type allows any key. With
`optimizer_kwargs: dict[str, Any] | None = None` from `hydra-example`:

| Override | Hydra |
| --- | --- |
| `opt.optimizer_kwargs={eps:1e-6}` | Error, suggests `+` |
| `+opt.optimizer_kwargs={eps:1e-6}` | Works |
| `opt.optimizer_kwargs.eps=1e-6` | Error |
| `+opt.optimizer_kwargs.eps=1e-6` | Error (the field is `None`) |

Without `+`, such a field could only be set from a YAML file. nematocyte uses
the schema instead:

- Under a dataclass, an unknown key is an error. `model.hiden_dim=10` is still
  caught.
- A field typed `dict[...]` accepts any key.

### A dict-typed field is given as a mapping, and merged

`opt.optimizer_kwargs={eps:1e-6}` is merged into the dict that the field
holds, as in a config file and as in Hydra
([composition.md](composition.md#a-mapping-is-merged-into-a-mapping-also-in-a-value)).
`opt.optimizer_kwargs={_replace_:true,eps:1e-6}` replaces it instead
([composition.md](composition.md#_replace_-true-replaces-instead-of-merging)).

Dotted paths end at the dict-typed field. `opt.optimizer_kwargs.eps=1e-6` is
not supported, unlike in Hydra; a key inside the dict is set with a mapping,
which the error shows: `opt.optimizer_kwargs={eps:1e-6}`. For dotted keys,
use a `TypedDict`.

### A dict for a subconfig or a group stands for its dotted keys

`opt={lr:0.1}` does what `opt: {lr: 0.1}` does in a config file: the values
are merged into the subconfig, and the values that it held before stay. It
is the same as `opt.lr=0.1`, and its keys are the same keys as the dotted
ones. A `dict[...]` field is merged in the same way
([above](#a-dict-typed-field-is-given-as-a-mapping-and-merged)), so `=`
means "as in a file" for both. Hydra merges both as well.

- The dict is applied with the values, in the order of the arguments, so
  that the last one wins for each key: `opt.lr=0.2 opt={lr:0.1}` sets
  `0.1`, and `opt={lr:0.1} opt.lr=0.2` sets `0.2`. Two dicts for the same
  subconfig both apply: `opt={lr:0.1} opt={momentum:0.9}` sets both.
- The dict stands for its dotted keys down to the first field that holds a
  value. A dict there is the value, which is merged into the value that
  the field holds: `opt={optimizer_kwargs:{eps:1e-6}}` is
  `opt.optimizer_kwargs={eps:1e-6}`. Inside a group, which fields hold a
  value depends on the option, so the dict is read after the selections.
- A selection and a dict for the same key both apply, the selection first,
  as with dotted keys: `opt=fast opt={lr:0.1}` applies the preset `fast`
  and then sets `lr`, in either order.
- For a group, the dict goes into the option that is selected, as in Hydra:
  `model=cnn model={kernel_size:3}` gives `Cnn(kernel_size=3)` in either
  order. If no option is selected, it is an error that lists the options.
- A string for a nested group or subconfig is an error that points to
  `KEY=NAME`, as a string among the values of a file points to
  `_defaults_`.
- For an optional subconfig that is off, a dict switches it on, and
  `ema={}` does so with the class defaults
  ([unions.md](unions.md#optional-subconfigs)). That happens with the
  selections, also for an optional subconfig inside the dict, so that
  `ema.every=4 ema={decay:0.5}` works in either order; the values of the
  dict follow in order. Together with `ema=null`, in either order, the dict
  is an error, since the order of the arguments does not decide between
  them.
- For an optional group that is off, a dict is an error: it names no
  option. The error points to `model=NAME`
  ([unions.md](unions.md#optional-groups)).
- For a TypedDict, and in a dict value, `_replace_:true` in the dict drops
  what the config files set at its place, and keeps every selection and
  value on the command line: `opt={_replace_:true,lr:0.1}`
  ([composition.md](composition.md#_replace_-true-replaces-instead-of-merging)).
  It is not a key of its own: `opt._replace_=true` is an error that points
  to the dict.
- A key can be swept only once: `-m opt={lr:1},{lr:2} opt=fast,slow` is an
  error.
- With `--multirun`, a swept value keeps the place of its sweep among the
  arguments. As in Hydra, `opt.lr=0.1,0.2 opt={lr:0.5}` gives two jobs that
  both have `lr=0.5`, and `opt={lr:0.5} opt.lr=0.1,0.2` gives `0.1` and
  `0.2`.

### A hand-written parser

The grammar is about 15 rules, less what is removed here. ANTLR would add a
runtime dependency and a code-generation step; Hydra only avoids the
dependency because OmegaConf vendors the runtime. A hand-written lexer with
the same two modes and a recursive-descent parser is a few hundred lines.

### A small, strict grammar

Grammar that `hydra-example` does not use is left out:

- `key@package`
- `a[0]` and escaped keys
- the casts `int()`, `float()`, `str()`, `bool()`, `json_str()`
- `glob()`, `sort()`, `shuffle()`
- `extend_list()`, which Hydra itself has deprecated

Input that Hydra rejects stays rejected. Unquoted, these characters are an
error and have to be quoted: `~` (`dm.root=~/data`), non-ASCII letters, `#`,
`&`, `!`, and a second `=`.

### Values are typed by the grammar, as in a YAML file

The value the grammar produces is what pydantic receives. `123` is an `int`
and `1.50` is a `float`; to get a string, quote it. This is the same rule as
in a config file, so there is one thing to remember.

Tested against pydantic's strict mode on JSON input
([deserializer.md](deserializer.md#strict-validation-through-json)):

| Override | Field | Result |
| --- | --- | --- |
| `seed=123` | `int` | `123` |
| `opt.lr=1e-3`, `opt.lr=1` | `float` | `0.001`, `1.0` |
| `flag=true` | `bool` | `True` |
| `wandb.name=run1` | `str \| None` | `'run1'` |
| `wandb.name=null` | `str \| None` | `None` |
| `wandb.name=123` | `str \| None` | rejected: "Input should be a valid string" |
| `wandb.name='123'` | `str \| None` | `'123'` |
| `wandb.tags=[a,1]` | `list[str] \| None` | rejected; `[a,'1']` works |
| `dm.root=/srv/data` | `Path` | `Path('/srv/data')` |
| `dm.label_map={1:2}` | `dict[int, int] \| None` | `{1: 2}` |
| `opt.optimizer_kwargs={eps:1e-8,amsgrad:true}` | `dict[str, Any] \| None` | `{'eps': 1e-08, 'amsgrad': True}` |

Two things to know:

- **The shell removes quotes.** `wandb.name='123'` reaches the program as
  `wandb.name=123`. The quotes have to be protected: `"wandb.name='123'"` or
  `wandb.name=\'123\'`. The error for `wandb.name=123` should say so.
- **The grammar is not YAML.** It agrees with YAML 1.2, which the config
  files use ([composition.md](composition.md)), on the cases above, and on
  `yes`, `on` and `1e-3`. It differs in a few places:

  | Text | On the command line | In a config file |
  | --- | --- | --- |
  | `007` | string | int `7` |
  | `inf` | float | string; the float is written `.inf` |
  | `2026-11-16` | string | date |
  | `~` | An error unless quoted | null |

This differs from Hydra in one place. OmegaConf converts a number to a string
for a `str` field, so `wandb.name=123` works there without quotes (and
`s=1.50` becomes `'1.5'`).

### Group or value from the field's type

The question is only how to read `key=name`: as "select the config called
`name` for this field" or as "set this field to the value `name`".

- If the field's type is a dataclass, or a group of dataclasses (see
  [unions.md](unions.md)), it is a selection.
- Otherwise it is a value.

`Config.model` is a group, so `model=cnn` is a selection. `Config.opt` is a
dataclass, so `opt=weight_decay_adamw` is one too. `Config.seed` is an `int`,
so `seed=123` is a value.

The order of the arguments has no meaning: `model.kernel_size=3 model=cnn`
is the same as `model=cnn model.kernel_size=3`. Selections are applied
first, shortest path first, and then the values. There are two exceptions.
A key that is given more than once: the last one wins (`seed=1 seed=2` sets
`2`). An earlier selection or value for the key is dropped before anything
is applied, so that with `--multirun`, `seed=1,2 seed=3` is one job. A dict
for a subconfig or a group is not dropped: it stands for its dotted keys,
and is applied with the values, in the order of the arguments
([below](#a-dict-for-a-subconfig-or-a-group-stands-for-its-dotted-keys)).
And the added configs (`+NAME`) are applied in the order given, before all
selections and values. What a selection does to
the values that the config files set is described in
[composition.md](composition.md#layers).

Which names can be selected does not change from Hydra. For `model` they are
the registered classes (`fcn`, `cnn`) plus the YAML files in `configs/model/`
(`single_linear_layer`). `model=single_linear_layer` works as before.

What changes is where the answer to "selection or value?" comes from. Hydra
looks at the config directories and the defaults list. Observed in
`hydra-example`:

| Command | Hydra |
| --- | --- |
| `opt=weight_decay_adamw` | Works, but only because `configs/opt/` exists and the primary config contains `- opt: null` |
| The same, with `- opt: null` removed | "Could not override 'opt'. No match in the defaults list. To append to your default list use +opt=weight_decay_adamw" |
| `wandb=foo` (there is no `configs/wandb/`) | Treated as a value: "Invalid type assigned: str is not a subclass of WandbCfg" |
| `model=123` | An unhandled traceback |

With the field's type as the source:

- `opt=weight_decay_adamw` works without the `- opt: null` line, because
  `Config.opt` is a dataclass. This matters now that `+` is gone.
- `wandb=foo` is read as a selection, and the error can say that `wandb` has
  no config called `foo`.

### nematocyte's own settings under the key `nematocyte`

Launcher and sweeper settings live under the reserved top-level key
`nematocyte`, next to the user's config. Hydra does the same with the key
`hydra`: its own config is composed into the same tree as the user's.

The user's root dataclass has no such field, so the reserved key has its own
schema, owned by nematocyte. The rule from the previous section then applies
unchanged:

- `nematocyte.launcher=slurm_kyiv` is a selection, because the launcher field
  is a dataclass.
- `nematocyte.launcher.partition=general` sets a value.

The user's root dataclass must not have a field called `nematocyte`.

The output directory is set under the same key, as `nematocyte.output_dir`
([outputs.md](outputs.md)), and so is the level of log messages, as
`nematocyte.log` ([logging.md](logging.md)).

### No slash in keys

In Hydra a key has two spellings, for two different namespaces:

- `hydra/launcher=slurm_kyiv` names a **config group**, which is a directory
  of config files. The slash is a path separator: load `slurm_kyiv.yaml` from
  `hydra/launcher/`.
- `hydra.launcher.partition=general` names a **node in the composed tree**.

The two usually line up, because a group's content lands at the tree path
that matches its directory. Hydra keeps them apart because they need not: a
`# @package` header or `group@package` can put a group's content somewhere
else. For a top-level group such as `model` there is no slash to see, which
is why `model=cnn` has to be disambiguated by checking whether a group
exists.

Using the wrong spelling fails silently. Observed in `hydra-example`:

| Command | The `hydra.launcher` node afterwards |
| --- | --- |
| `hydra/launcher=slurm_kyiv` | The settings from `slurm_kyiv.yaml` |
| `hydra.launcher=slurm_kyiv` | The string `slurm_kyiv`; no error |
| `hydra.launcher=nope` | The string `nope`; no error |

nematocyte has one namespace, the tree (see
[composition.md](composition.md)), and the field's type says whether a key is
a selection. The slash form leaves the grammar, and a selection is
written with dots at every depth.

### Four flags to start with

| Flag | Purpose |
| --- | --- |
| `--help`, `-h` | Prints the groups with their options and the composed config |
| `--multirun`, `-m` | Runs one job per combination of swept values |
| `--config-name`, `-cn` | Selects the primary config |
| `--cfg`, `-c` `{job,all}` | Prints the composed config instead of running |

`--cfg` is taken from Hydra, with its required argument:

- `--cfg job` prints the user's config, without the `nematocyte` key.
- `--cfg all` prints the user's config and nematocyte's own settings.

The output is the whole config: the values from files and overrides, and the
defaults of the dataclasses for everything else.

As in Hydra, interpolations are printed as written, not resolved. Hydra's
`--resolve`, which prints the resolved values instead, is a likely addition
but not in the first version.

A tree with unresolved interpolations cannot be validated in full:
`hidden_dim: ${dm.default_res}` is a string where an `int` is expected. So
after printing the tree, `--cfg` resolves and builds the config as a normal
run would, and reports any error.

Hydra's third choice, `--cfg hydra`, which prints only Hydra's own settings,
has no counterpart for now.

More flags can be added later.

What `--help` shows, and where the descriptions of the values come from, is
covered in [helptext.md](helptext.md).

For reference, the flags Hydra has beyond these (`hydra/_internal/utils.py`):

| Flag | What it does in Hydra |
| --- | --- |
| `--resolve` | With `--cfg`: resolves interpolations before printing |
| `--package`, `-p` | With `--cfg`: prints only one subtree |
| `--info`, `-i` `{all,config,defaults,defaults-tree,plugins,searchpath,aliases}` | Prints debugging information about composition |
| `--run`, `-r` | Runs a single job; this is the default |
| `--config-path`, `-cp` | Overrides the config directory given in the entry point |
| `--config-dir`, `-cd` | Adds a second config directory to the search path |
| `--shell-completion`, `-sc` | Installs or removes shell tab completion |
| `--experimental-rerun` | Reruns a job from its saved output directory |
| `--hydra-help` | Prints help for Hydra's flags, as opposed to the application's |
| `--version` | Prints Hydra's version |

### Details of the grammar and the flags

- A quoted key in a dict is rejected, as in Hydra (`{'a b':1}`).
- Function arguments can be named, as in Hydra (`range(start=1, stop=4)`).
- `interval(1, 12)` keeps its integers, so that Optuna gets an integer
  distribution ([multirun.md](multirun.md)).
- `--help` and `--cfg` with `--multirun` leave the swept values out, since
  there is one config per job. Without `--multirun`, a sweep is the error
  that it always is.
- Exit codes: 1 for an error in the configuration, 2 for a wrong flag (from
  `argparse`). A config error is printed without a traceback; an exception
  of the program itself is not caught.

## Not in the first version

- **Shell tab completion**, which Hydra installs with `--shell-completion`.
  Perhaps later.
- **`--resolve`**, to print the resolved values with `--cfg`.
- **Sweeping over added configs**, as in `+presets/a,presets/b`. With
  `--multirun`, the added configs are the same for every job.

## Rejected ideas

### An added config as a list of command-line arguments

Read `+presets/debug` as if the content of the file had been typed on the
command line: `opt.lr: 0.1` as `opt.lr=0.1`, an entry `dm: cmnist` in
`_defaults_` as `dm=cmnist`, and `_self_: base` as `+base`.

**Rejected because** the arguments of the command line are applied as one
set, selections first and then values, and a repeated key goes by order.
The order of the arguments would then decide between the file and an
explicit `opt.lr=3.0`, where Hydra lets the command line win either way.
And a selection would no longer replace what the layer below set: with a
`_self_` whose config sets `model.activation: selu`, and a file that selects
`model: cnn`, the value would be applied to the CNN after the selection,
instead of being dropped with the old model.

### Converting numbers to strings for `str` fields

pydantic's `coerce_numbers_to_str` setting makes lax mode accept a number for
a `str` field, which is what OmegaConf does. `wandb.name=123` would then work
without quotes.

**Rejected because** the setting covers floats as well as ints, and a float
does not survive the round trip: `3.10` becomes `'3.1'`. The setting cannot
be limited to ints.

To revisit if quoting turns out to be needed often. An int-only conversion
would have to be written by hand, before the tree goes to pydantic.

### Scalars reach the validator as text

Hand every unquoted scalar to pydantic as its original text and let lax mode
convert it to the field type. This works for typed fields (`int`, `float`,
`bool`, `str`, `Path`, `Literal`, `X | None`, `list[int]`, `dict[int, int]`),
keeps `s=1.50` as `'1.50'`, and needs no quoting for `wandb.name=123`.

**Rejected because** it cannot be the only rule. `null` still has to be a
token of the grammar, since pydantic does not turn the text `null` into
`None`. And under `Any`, such as the values of `dict[str, Any]`, there is no
type to convert to: `opt.optimizer_kwargs={eps:1e-8}` would yield the string
`'1e-8'`, so the grammar's typing would be needed there after all. Two
mechanisms are harder to explain than one.
