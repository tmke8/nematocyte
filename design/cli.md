# Command line

How command-line arguments are parsed and applied to the config.

## Goal

Keep Hydra's command-line syntax as it is, with one removal: the `+key=value`,
`++key=value` and `~key` overrides. They are not used in `hydra-example` and
are confusing.

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

### No `+`, `++` or `~`

The grammar's top-level rule keeps only `key=value` and `key=`.

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

### A dict-typed field is set as a whole, and replaced

`opt.optimizer_kwargs={eps:1e-6}` replaces whatever the field held before.
Hydra merges instead, which is not what `=` suggests.

Dotted paths end at the dict-typed field. `opt.optimizer_kwargs.eps=1e-6` is
not supported; to change one key, the whole dict is given again.

### A dataclass field cannot be given a dict value

`opt={lr:0.1}` is an error. Hydra accepts it and merges the dict into the
subconfig. In nematocyte the fields are set one by one (`opt.lr=0.1`).

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

Tested against pydantic's lax mode:

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
([outputs.md](outputs.md)).

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

## Not in the first version

- **Shell tab completion**, which Hydra installs with `--shell-completion`.
  Perhaps later.
- **`--resolve`**, to print the resolved values with `--cfg`.

## Rejected ideas

### Converting numbers to strings for `str` fields

pydantic's `coerce_numbers_to_str` setting makes lax mode accept a number for
a `str` field, which is what OmegaConf does. `wandb.name=123` would then work
without quotes.

**Rejected because** the setting covers floats as well as ints, and a float
does not survive the round trip: `3.10` becomes `'3.1'`. The setting cannot
be limited to ints.

To revisit if quoting turns out to be needed often. An int-only conversion
would have to be written by hand in the walk over dataclass fields.

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
