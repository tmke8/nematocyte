# Composition

How config files are found and combined into one tree.

## How Hydra does it

Hydra keeps two namespaces apart:

- A **config group** is a directory of config files. Its name is a path, with
  slashes: `dm`, `hydra/launcher`.
- A **package** is where a file's content goes in the composed tree. Its name
  is a tree path, with dots: `dm`, `hydra.launcher`.

By default the package follows from the file's directory. A file can override
that with a `# @package` header, and a defaults entry can override it with
`group@package`.

This is why `configs/experiments/cmnist_cnn.yaml` in `hydra-example` needs two
pieces of boilerplate. Observed by editing a temporary copy:

| Change to the file | Result of `--config-name experiments/cmnist_cnn` |
| --- | --- |
| None | The config, at the root of the tree |
| `# @package _global_` removed | The whole config lands under a key called `experiments` |
| `- /dm: cmnist` written as `- dm: cmnist` | "Could not find 'experiments/dm/cmnist'" |

The header moves the content from `experiments` to the root. The leading
slash makes the group path absolute; a relative `dm` is looked up next to the
file, as `experiments/dm`.

## Decided

### One namespace

There is no `# @package` header and no `group@package`. A file's content
always goes to the tree path that matches where the file is.

With only one namespace, the slash form of a key has nothing left to express
and is gone from the command line too; see [cli.md](cli.md).

### A file's role comes from how it is used

- A file named by `--config-name`, or named as the default in the entry
  point, is a **primary config**. It describes the root of the tree, wherever
  the file is. So is a file named by `+NAME` on the command line, which is
  applied on top of the primary config ([cli.md](cli.md)).
- A file in a directory whose path matches a field whose type is a dataclass
  or a group (see [unions.md](unions.md)) is a named config for that field:
  `configs/dm/` holds the options of the group `dm`, and `configs/opt/` the
  presets for the subconfig `opt`.

`configs/experiments/cmnist_cnn.yaml` is therefore a primary config with no
header.

What a file builds on is said under its key `_defaults_`.

### `_defaults_` is a mapping

Hydra calls the key `defaults`, and it is a list, for reasons that no
longer apply:

- The list is a merge order, and the entry `_self_` marks where the file's
  own values go in it.
- Some entries have no key, such as `- base_config` and, in a group file,
  `- celeba`.
- Some entries carry a keyword, such as `- override model: cnn`.

In nematocyte `_defaults_` is a mapping. Each key says where: a field of the
node that the file describes, or `_self_` for that node itself. Each value
names the config to load there, or is a mapping that selects for the fields
of the field ([below](#an-entry-in-_defaults_-can-be-nested)).

```yaml
# configs/experiments/cmnist_cnn.yaml, a primary config
_defaults_:
  dm: cmnist
  model: cnn
  opt: weight_decay_adamw

dm:
  num_colors: 10
```

```yaml
# configs/dm/celeba_male_smiling.yaml, a named config for `dm`
_defaults_:
  _self_: celeba

superclass: Smiling
```

- The order of the entries has no meaning. What is applied before what
  follows from the keys; see [Layers](#layers).
- A place cannot be given twice: the parser reports a repeated key, in
  YAML and in JSON5 files.
- Besides the name of a config, a value can be `null`, which switches off
  an optional subconfig or group, or `_some_`, which switches on an
  optional subconfig ([unions.md](unions.md#optional-subconfigs)).
- A primary config describes the root, so `dm: cmnist` means the field `dm`
  of the root dataclass. There is no leading slash, and a primary config
  reads the same whichever directory it is in.
- `_defaults_` and `_self_` are reserved: no config class can have a field
  with either name ([deserializer.md](deserializer.md#reserved-names)). In
  Hydra `_self_` means something else, as described above.
- The key is not called `defaults` as in Hydra, because a name that is
  reserved below the root takes underscores, and `defaults` is a natural
  name for a field.

### `_self_` names the config that a file extends

`_self_` names another config for the same node. That config is loaded
first, and the `_defaults_` and values of this file apply on top.

| File | What `_self_` can name | Required |
| --- | --- | --- |
| A named config for a group | An option of the group, or another named config in the same directory | Yes, except in the directory of an option; the chain has to end at an option ([unions.md](unions.md#defaults)) |
| A preset for a plain subconfig | Another preset in the same directory | No; without it, the chain ends at the class defaults of the subconfig |
| A primary config | Another primary config, by the name that `--config-name` takes | No |

A chain that comes back to itself is an error.

A primary config that extends another one:

```yaml
# configs/experiments/good_run_blond.yaml
_defaults_:
  _self_: good_run
  dm: celeba/male_blond

seed: 2
```

`hydra-example` has neither a primary config that extends another, nor a
preset that extends another. Both were tested in Hydra on a copy of it.

### An entry in `_defaults_` can be nested

Suppose the option `cnn` of `model` has a field `norm`, which is itself a
group. A primary config can select for both:

```yaml
_defaults_:
  model:
    _self_: my_cnn
    norm: my_batchnorm

model:
  norm:
    momentum: 0.2
```

`norm: my_batchnorm` loads `configs/model/norm/my_batchnorm.yaml` (see
below for the directory). It is an error if the option selected for
`model` has no field `norm`.

- In a nested mapping, `_self_` is the entry for the field that holds it,
  as `_self_` at the top of `_defaults_` is the entry for the node of the
  file. `model: my_cnn` and `model: {_self_: my_cnn}` mean the same.
- The other keys of a nested mapping are fields of that field, and can be
  nested further. A nested mapping without `_self_` selects only for the
  fields: `model: {norm: my_batchnorm}` keeps the option that a lower layer
  selected for `model`.
- A nested `_self_` takes what an entry takes, `null` and `_some_`
  included: `ema: {_self_: _some_, inner: big}` switches `ema` on and
  selects `big` for its field `inner`. Only `_self_` at the top of
  `_defaults_` has to name a config.
- A key with a dot, `model.norm: my_batchnorm`, is an error that names the
  file and shows the entry in the nested form, and says that `_self_`
  selects for the field itself. The dot in a key would be a
  second way to write the same entry; it also has to be quoted in JSON5,
  where the nested form does not. On the command line, a selection is still
  written with dots, `model.norm=my_batchnorm` ([cli.md](cli.md)).
- An empty mapping selects nothing, and is an error that names the file
  and the entry.
- A key that is not the name of a field, such as `model/norm`, and a value
  that is neither a name, `null`, nor a mapping, are errors that name the
  file and the key.
- In error messages, an entry is named by its path with dots,
  `'model.norm'`.

A named config can do the same for its own fields. The keys other than
`_self_` are relative to the node that the file describes:

```yaml
# configs/model/my_cnn.yaml
_defaults_:
  _self_: cnn
  norm: layernorm
```

### Layers

Everything that contributes to the config is a layer, and a higher layer
always wins. For one file, from lowest to highest:

1. The defaults of the dataclasses.
2. The config that the file names with `_self_`, composed completely,
   including its own `_self_`.
3. The mappings in the file with `_replace_: true`, which drop what the
   layers below set at their place
   ([below](#_replace_-true-replaces-instead-of-merging)).
4. The other entries in the file's `_defaults_`, shortest path first.
5. The values in the file.

The command line comes on top of the primary config, in the same way: first
the added configs (`+NAME`), then its selections and its dicts for
TypedDicts, which replace them ([cli.md](cli.md#a-dict-for-a-typeddict-replaces-it)),
shortest path first, then its values. An added config is
composed as if it extended the primary config with `_self_`
([cli.md](cli.md)): its own `_self_` chain, its `_defaults_` and its values
apply on top of the tree so far, without going back to the defaults of the
dataclasses.

The order in which things are written, in a file or on the command line,
never matters, with one exception: several added configs are applied in the
order given.

**Shortest path first.** `norm` under `model` is inside `model`, so it is
the more specific entry and is applied later, whether it is written next to
`_self_` or in another layer's file. In the example above, `my_batchnorm`
wins over what `configs/model/my_cnn.yaml` selects for `norm`, and
`momentum: 0.2` is then set on it. Two paths of which neither contains the
other cannot conflict.

**What a selection does to the node.** Selecting a config replaces the
node: an option of a group, a named config for one, and a preset for a
plain subconfig alike. The node starts again from the class defaults: of
the option, for a group, and of the subconfig's class, for a preset
([unions.md](unions.md#defaults)). Everything that lower layers selected or
set under it is dropped, the default instance of the field included. A
chain of presets ends at the class defaults, as a chain of named configs
for a group ends at an option, so `opt=p` means the same whether `opt` is a
group or a plain subconfig.

`_some_` is not a selection: it switches an optional subconfig on, and
leaves one that is on as it is
([unions.md](unions.md#optional-subconfigs)).

With `good_run.yaml`, which sets `model.activation: selu`,
`dm.download: false`, `opt.lr: 0.001` and `opt.weight_decay: 0.001`:

| Applied on top of `good_run` | Result |
| --- | --- |
| `opt: weight_decay_adamw`, in a file that extends it or as `opt=weight_decay_adamw` | `opt` starts from the class defaults. `weight_decay` is `0.01`, from the preset; `0.001` is dropped |
| `model: fcn`, or `model=fcn` | An FCN with its class defaults. `activation: selu` is dropped, and `activation` is `gelu` |
| `dm: celeba/male_blond` | `download` is back at its default. To keep `false`, it is set again |

Hydra has no such rule. Tested on a copy of `hydra-example`, with the same
`good_run` and the same preset:

| How the preset is selected | `opt.weight_decay` |
| --- | --- |
| Child lists `- good_run`, then `- opt: weight_decay_adamw` | `0.01`, the preset wins |
| Child lists `- opt: weight_decay_adamw`, then `- good_run` | `0.001`, the parent wins |
| The parent has an entry for `opt`; child lists `- override opt: weight_decay_adamw` | `0.001`, the parent wins |
| Command line `+opt=weight_decay_adamw`; the parent has no entry for `opt` | `0.01`, the preset wins |
| Command line `opt=weight_decay_adamw`; the parent has an entry for `opt` | `0.001`, the parent wins |

- The result depends on the order of the list, and on whether the extended
  config happens to have an entry for the field.
- Values that the parent sets under a group stay when another option is
  selected. `--config-name good_run model=fcn` gives an FCN with
  `activation: selu`. If the parent sets a value that the new class does not
  have, such as `model.num_hidden` with `model=cnn`, the result is an error,
  and the child has no way to remove the value.

### A value that a selection drops is warned about

Selecting a config drops values without a word. `+good_run model=fcn`
gives `activation: gelu`, where `good_run.yaml` says `selu`, and
`+good_run model=single_linear_layer` changes the result although it selects
what `good_run` already selects. So a selection that drops such a value
gives a warning, for an option or named config of a group and for a preset
of a plain subconfig alike:

```
The override 'model=fcn' resets the config values in 'model' to the values given by 'fcn'. This drops a value that was set for 'model' before:
    'model.activation=selu', set in `configs/good_run.yaml`
To keep the value, set it again after the selection; in the file that makes the selection, or on the command line.
```

**What counts.** A value or a selection under the node that a lower layer
set explicitly: in the values of a file, or in its `_defaults_`. What the
replaced config itself brings is not counted, since it goes along with the
config. With `good_run` and `model=fcn`, `num_hidden: 0` and `norm: none`
from `single_linear_layer.yaml` are dropped with it, and are not shown.

**When there is no warning.** A dropped value or selection is left out of
the warning, so that it does not fire on configs that are fine:

- when a layer above the selection sets it again, as
  `experiments/good_run_blond.yaml` does with `dm.download`. The value that
  the new config itself brings does not count as set again: with
  `model=single_linear_layer`, its `activation: relu` replaces `selu`, and
  that is warned about;
- when the composed config ends up with the same value anyway, such as a
  dropped `download: false` where the new class has `download: bool = False`;
- when the new config has no such field. A CNN has no `num_hidden`, so
  dropping it is what the selection is for.

The check runs once the whole tree is composed, so that "set again" counts
every layer, the command line included. There is one warning for each
selection that drops something, which lists everything it drops.

**It is a Python warning**, of the class `nematocyte.DroppedValueWarning`, a
subclass of `UserWarning`. It points at the caller's code, not at
nematocyte. On the command line, Python prints it to standard error; in a
test, pytest lists it, and

```toml
[tool.pytest.ini_options]
filterwarnings = ["error::nematocyte.DroppedValueWarning"]
```

makes it fail the test. The usual filters of `warnings` silence it. It is
not a message of `logging`: composition runs before
`nematocyte.configure_logging()` can be called ([logging.md](logging.md)),
and `nematocyte.compose` and the validator have no logging set up at all.

**Where it is given**: wherever a config is composed, with
`nematocyte.compose`, `--help`, `--cfg` and the jobs of a multirun.
The validator warns for a primary config, which is composed as it would be
in a run. It does not warn for the selections it makes itself to check a
named config ([validation.md](validation.md)), because nobody wrote those.
Python shows a warning with the same text only once from the same place,
so the jobs of a multirun that drop the same values give one warning.

### A declared key is merged, any other key is replaced

In a config file, whether a mapping is merged into what the layers below set,
or replaces it, is decided per key:

1. A key that is declared in a config class, a field of a dataclass or a key
   of a TypedDict, is merged: the keys that the mapping under it gives are
   set, and the others stay. For a TypedDict, `_replace_: true` makes it
   replace instead
   ([below](#_replace_-true-replaces-instead-of-merging)).
2. A key that is not declared, such as any key of a field typed `dict[...]`,
   is replaced: when a higher layer writes a mapping for the dict, what a
   lower layer set in it is gone. `_replace_: false` makes a mapping in a
   dict merge one level instead
   ([below](#_replace_-false-merges-a-dict-one-level)).

So a dataclass or a TypedDict is merged, and a dict is replaced as a whole,
deeply: nothing inside it is merged, unless a mapping says so.

```python
@dataclass
class Opt:
    lr: float = 0.1
    kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "sched": {"gamma": 0.1, "step": 5},
            "milestones": [1, 2],
        }
    )
```

| Written in a config file on top | `opt.kwargs` in Hydra | `opt.kwargs` in nematocyte |
| --- | --- | --- |
| `opt: {kwargs: {sched: {gamma: 0.5}, extra: 1}}` | `{sched: {gamma: 0.5, step: 5}, milestones: [1, 2], extra: 1}` | `{sched: {gamma: 0.5}, extra: 1}` |
| `opt: {kwargs: {milestones: [9]}}` | `{sched: {gamma: 0.1, step: 5}, milestones: [9]}` | `{milestones: [9]}` |
| `opt: {kwargs: {sched: null}}` | `{sched: null, milestones: [1, 2]}` | `{sched: null}` |
| `opt: {kwargs: {}}` | `{sched: {gamma: 0.1, step: 5}, milestones: [1, 2]}` | `{}` |

In all four, `opt.lr` stays `0.1`: `opt` is a dataclass, and is merged.
The Hydra column was tested by merging into the structured config with
OmegaConf, as Hydra 1.4 does; it merges every mapping into a mapping.

A dict usually holds keyword arguments for a function, and a file that
writes some of them means those, not those on top of whatever a lower layer
set. A key that is to be set on its own belongs in a TypedDict, where it
also has a dotted key on the command line
([cli.md](cli.md#on-the-command-line-a-dict-replaces)).

- What counts is the type of the field, not the value: a field typed `Any`
  or `dict[str, int] | None` that holds a mapping is replaced, since it
  declares no keys. A config class inside a value, such as the items of
  `dict[str, Point]`, is replaced with the dict. A mapping over something
  that is not one, such as `null` or a string, replaces it too.
- A mapping for a subconfig, a group or a TypedDict is merged deeply: a
  mapping for a subconfig inside it is merged into that subconfig, and a
  dict inside it is replaced.
- The default of the field is the first layer, as for a subconfig.
- An interpolation (`${...}`) is resolved after composition, so it is a
  string while the layers are applied: a mapping written over it replaces
  it ([interpolation.md](interpolation.md)).
- On the command line, a dict replaces too
  ([cli.md](cli.md#on-the-command-line-a-dict-replaces)), also for a
  TypedDict ([cli.md](cli.md#a-dict-for-a-typeddict-replaces-it)), and a
  key inside a dict is set on its own with `+`, as in
  `+opt.kwargs.eps=1e-6`, which keeps the other keys of the dict
  ([cli.md](cli.md#keyvalue-sets-a-key-inside-a-dict)).
- Every TypedDict is treated as closed: a key that it does not declare is an
  error. Once an open TypedDict (`closed=False`, or with `extra_items`) is
  supported, its extra keys follow rule 2, and are replaced while its
  declared keys are merged.

### `_replace_: true` replaces instead of merging

Since a mapping for a TypedDict is merged, a file has no way to take out a
key that a layer below set: `wd: null` is a value, and `~KEY` exists only on the command
line and only for a TypedDict
([cli.md](cli.md#key-leaves-out-an-optional-key-of-a-typeddict)). A mapping
can say instead that it replaces what the layers below set at its place:

```yaml
# configs/presets/plain_adam.yaml
opt:
  _replace_: true
  lr: 0.001
```

With `opt: Opt = field(default_factory=lambda: Opt(lr=0.1, wd=0.01))`,
`+presets/plain_adam` gives `opt == {"lr": 0.001}`. Without `_replace_`, it
gives `{"lr": 0.001, "wd": 0.01}`.

Hydra has no way to do this. Tested with Hydra 1.4, on top of a primary
config with `opt: {lr: 0.1, wd: 0.01}`:

| Written in a config file on top | Result in Hydra |
| --- | --- |
| `opt: {wd: null}` | `{lr: 0.1, wd: None}` |
| `opt: {wd: ???}` | `{lr: 0.1, wd: 0.01}`: merging `???` does nothing |
| `opt: {wd: {_delete_: true}}` | A dict with the key `_delete_` |

Only `~opt.wd` on the command line takes the key out.

A dict is replaced anyway
([above](#a-declared-key-is-merged-any-other-key-is-replaced)), so in a
value, `_replace_: true` is allowed in any mapping, at any depth, and
changes nothing. `_replace_: false` merges a mapping in a value instead
([below](#_replace_-false-merges-a-dict-one-level)).

The rules:

- `_replace_: true` drops everything that the layers below set or selected
  at its place, the default of the field included, as `~KEY` does for
  one key. For a TypedDict, the node is then as if nothing had set anything
  in it. A TypedDict has no defaults, so it is an empty dict, and each
  `Required` key has to be set again ("a value is required" otherwise). A
  subconfig or a group inside it is as if the TypedDict had no default: a
  group has no option selected, and a subconfig starts from the defaults of
  its class.
- The layers below are what was applied before the file: the default of
  the field, the files below it, and the file's own chain of `_self_`.
  What the file selects under `_defaults_` and sets among its values stays,
  wherever it is written in the file: `_replace_` is applied before them
  ([layers](#layers)).
- `_replace_` exists only in config files. On the command line, a dict
  for a TypedDict or a value replaces anyway
  ([cli.md](cli.md#a-dict-for-a-typeddict-replaces-it)), so `_replace_` in
  it is an error, true or false, and so is `opt._replace_=true`.
- A TypedDict replaced in this way can be a subconfig, the selected option
  of a group of TypedDicts, or an optional subconfig that is on. A mapping
  does not switch an optional subconfig on, with `_replace_` or without:
  `_some_` under `_defaults_` does
  ([unions.md](unions.md#optional-subconfigs)).
  The option of a group stays selected, and only what was set in it is
  dropped. `_replace_` is also allowed at the top level of a named config
  for such a field, where it drops what was set for the field before the
  file, including what the file's chain of `_self_` set.
- In a value, `_replace_` is allowed in any mapping, but not in the items
  of a list; there it is a key of the item.
- For a dataclass, `_replace_` is an error that names the class, true or
  false. Every field of a dataclass that has a default can be set back to
  it, so a dataclass needs it less; it may be allowed later. At the top
  level of a primary config, where it would replace the whole config, it is
  an error too.
- The value is `true` or `false`. Anything else is an error. For a
  TypedDict, `false` is allowed and changes nothing, since its mapping is
  merged anyway: a mapping that is to be merged can leave out the key.
- An empty mapping for a TypedDict, `opt: {}`, is merged, and so changes
  nothing; `opt: {_replace_: true}` makes it empty. A dict is made empty
  with `kwargs: {}`. Null is not an empty dict: for a dict that does not
  take `None`, it is an error that shows `kwargs: {}`
  ([below](#details-of-finding-and-reading-files)).
- `_replace_` is reserved on every config class
  ([deserializer.md](deserializer.md#reserved-names)), and in the mappings
  of a value: a dict with a key `_replace_` of its own cannot be written in
  a config file or on the command line.
- What `_replace_` drops is not warned about: the file says that it is to
  be dropped. A warning that a selection below recorded for values under
  the node is dropped with them.

### `_replace_: false` merges a dict one level

A mapping in a dict replaces what was there
([above](#a-declared-key-is-merged-any-other-key-is-replaced)). With
`_replace_: false`, it is merged into it instead, one level: the keys that
it gives are set, the others stay, and the value under a key that it gives
replaces what was under that key, a mapping included. With `Opt` from
[above](#a-declared-key-is-merged-any-other-key-is-replaced):

| Written in a config file on top | `opt.kwargs` |
| --- | --- |
| `opt: {kwargs: {_replace_: false, extra: 1}}` | `{sched: {gamma: 0.1, step: 5}, milestones: [1, 2], extra: 1}` |
| `opt: {kwargs: {_replace_: false, sched: {gamma: 0.5}}}` | `{sched: {gamma: 0.5}, milestones: [1, 2]}` |
| `opt: {kwargs: {_replace_: false, sched: {_replace_: false, gamma: 0.5}}}` | `{sched: {gamma: 0.5, step: 5}, milestones: [1, 2]}` |
| `opt: {kwargs: {sched: {_replace_: false, gamma: 0.5}}}` | `{sched: {gamma: 0.5}}` |
| `opt: {kwargs: {_replace_: false, sched: null}}` | `{sched: null, milestones: [1, 2]}` |
| `opt: {kwargs: {_replace_: false}}` | Unchanged |

A file that wants to add a keyword argument to those that a lower layer
gave writes it this way, without repeating them. The merge is shallow, so
that the file still says the whole of what it gives under each key; a deeper
mapping is merged only if it says so too.

The rules:

- `_replace_: false` is allowed in any mapping inside a value, at any depth:
  a field typed `dict[...]`, a field typed `Any` that holds a mapping, a dict
  inside a TypedDict, and the items of `dict[str, Point]`.
- Each mapping is merged into what is at its place once the mapping around
  it has been applied. A mapping inside one that replaces has nothing to be
  merged into, so the fourth row of the table gives the same as without
  `_replace_`.
- What is there is what the layers below set, in the order of the
  [layers](#layers): the default of the field, the files below, the file's
  own chain of `_self_`, and what the file selects under `_defaults_`,
  which is applied before its values. If a `_replace_: true` around the
  dict dropped it, nothing is there.
- Over a value that is not a mapping, such as `None`, a string or a list, or
  a key that is not set, the mapping replaces it, as without `_replace_`.
- Over an interpolation (`${...}`), it is an error that names the key, the
  file, the class and the interpolation. The interpolation is resolved
  after composition ([interpolation.md](interpolation.md)), so there is no
  dict to merge into yet, and replacing it silently would drop what the file
  meant to keep.
- The items of a list are not looked into: there `_replace_` is a key of
  the item, as for `true`.
- On the command line, `_replace_` is an error, true or false. A dict there
  replaces, and a single key inside a dict is set with
  `+opt.kwargs.eps=1e-6` ([cli.md](cli.md#keyvalue-sets-a-key-inside-a-dict)).
  A dict on the command line comes after the files, so it replaces what a
  file merged.

### What a directory holds follows from the schema

`configs/model/` holds named configs because `Config.model` is a dataclass
field. `configs/experiments/` matches no field, so it can only hold primary
configs.

The test that validates every config file (`tests/test_configs.py` in
`hydra-example`) no longer has to be told which directories hold primary
configs.

The same applies to nematocyte's own settings under the reserved key:
`configs/nematocyte/launcher/` holds named launcher configs.

### A directory named after a field holds no primary configs

`--config-name model/single_linear_layer` is an error. Files in
`configs/model/` are named configs for `model` and nothing else.

This holds for a field that is a plain value too, such as `seed` or a field
typed `dict[...]`. Such a field has no named configs, so a file in
`configs/seed/` belongs nowhere, and is an error. A directory with that name
almost always means that the user expects `seed=one` to select
`configs/seed/one.yaml`, but `seed=one` sets the value to the string `"one"`;
the error catches the mistake at the directory. A primary config that sets
such a field goes at the top of the config directory or in a directory that
matches no field, and can be applied with `+NAME`: a dict in it replaces
the dict that the field holds
([above](#a-declared-key-is-merged-any-other-key-is-replaced)), as `=` on
the command line does.

A file at the top of the config directory that is named after a subconfig
or a group of the root dataclass, such as `configs/model.yaml`, is an error,
whatever is being composed. Its name says "a config for `model`", but at the
top it would be a primary config; a config for `model` belongs in
`configs/model/`. The same goes for `configs/nematocyte.yaml`. A file named
after a plain value, such as `configs/seed.yaml`, is an ordinary primary
config.

### Named configs for a nested field live in a nested directory

The directory path follows the field path at every depth. If an option of
`model` had a dataclass field `encoder`, its named configs would be in
`configs/model/encoder/`, next to the files that are options of `model`.
`configs/nematocyte/launcher/` is the same thing under the reserved key.

A file and a directory with the same stem do not clash: `encoder.yaml` is an
option of `model`, `encoder/` holds named configs for the field `encoder`.

A subdirectory of `configs/model/` has to match a dataclass field of at least
one option of `model`, or be the directory of an option (below). Anything
else is an error, because its files could be neither named configs nor
primary configs. The two cannot be confused: no option of a group can have a
field named after an option of the group (below).

### The configs of a group can be grouped by option

A subdirectory of a group's directory that is named after an option of the
group holds further named configs for the group, as if they were in the
group's directory itself. Their name has the subdirectory in front:

```yaml
# configs/dm/celeba/male_blond.yaml
superclass: Blond_Hair
subclass: Male
```

is selected with `dm=celeba/male_blond`, on the command line or in
`_defaults_`.

- The file is a named config for `dm` in every way. Its `_self_` names are
  looked up from `configs/dm/`, as for any named config of the group:
  `_self_: celeba` is the option, and `_self_: celeba/male_blond` another
  file in `configs/dm/celeba/`.
- Without `_self_`, the file extends the option that the directory is named
  after, as if it had `_self_: celeba`. The directory already says which
  option it is, so the file does not have to repeat it. Writing
  `_self_: celeba` is allowed.
- The chain of `_self_` of such a file has to end at the option that the
  directory is named after. A file in `configs/dm/celeba/` whose chain ends
  at `cmnist` is an error that names the file and both options.
- Only a built-in option of the group, one from the annotations, has a
  directory; a config from a file has none. The directory holds only config
  files, no further directories.
- A name with a slash is only the directory of an option of a group. For a
  plain subconfig, which has no options, it is an error, as is a name whose
  first part is not an option, such as `dm=other/x`.
- No option of a group can have a field whose name is an option of the
  group: if `model` has an option `cnn`, no option of `model` can have a
  field `cnn`, of any kind. `configs/model/cnn/` would be both the
  option's directory and the field's
  ([above](#named-configs-for-a-nested-field-live-in-a-nested-directory)).
  For a subconfig or a group, both would have their configs there. For a
  value, the directory would be an error that
  [catches](#a-directory-named-after-a-field-holds-no-primary-configs) a
  user who expects `model.cnn=x` to load `configs/model/cnn/x.yaml`, and
  the option's directory would hide that mistake. It is an error in the
  schema, which names the group, the option and the field, whether or not
  the directory exists.
- `--help` lists the configs in the directories of options under
  `From files:`, with the others, as `celeba/male_blond`.

In Hydra, a name with a slash loads the file from any subdirectory of the
group's directory, and nothing checks what the file extends. Tested on a
copy of `hydra-example` with `dm/celeba/male_blond.yaml` and
`dm/other/thing.yaml`: `dm=celeba/male_blond` and `dm=other/thing` both
compose, with the content under `dm`.

### The schema is always enforced

The root dataclass is given to the entry point, and every primary config is
validated against it. There is no schema entry in `_defaults_`.

In Hydra the schema is opt-in. `hydra-example` registers the root dataclass
under the name `base_config`, and a primary config is only validated if its
`defaults` list names it. Observed with a primary config containing
`seed: not-a-number` and `made_up_key: 1`:

| `defaults` | Result |
| --- | --- |
| Starts with `- base_config` | "Value 'not-a-number' of type 'str' could not be converted to Integer" |
| No `- base_config` | Composes without complaint, with both values in the tree |

### A primary config is optional

Without a primary config, the tree comes from the root dataclass alone:
field defaults, including defaults on group fields
([unions.md](unions.md#defaults)).

`configs/default_config.yaml` in `hydra-example` then has nothing left to
say:

| Entry | Why it is not needed |
| --- | --- |
| `- base_config` | The schema is always enforced |
| `- dm: cmnist`, `- model: fcn` | A group field can have a default in the dataclass |
| `- opt: null` | Selecting a preset no longer needs the entry ([cli.md](cli.md)) |

An empty string for `--config-name` means no primary config. The entry point
does not have to name a default one either.

### Config files are YAML 1.2, read with ruamel.yaml

Hydra reads config files with PyYAML, which implements YAML 1.1. The same
text read with PyYAML 6.0.3 and with ruamel.yaml 0.19.1:

| Text in a file | PyYAML (YAML 1.1) | ruamel.yaml (YAML 1.2) |
| --- | --- | --- |
| `1e-3` | string | float |
| `yes`, `on`, `No` | bool | string |
| `010` | int `8` | int `10` |
| `1:30` | int `90` | string |
| A key given twice | The second one wins, silently | Error |
| `1_000` | int | int |
| `3.10` | float `3.1` | float `3.1` |
| `2026-11-16` | date | date |

Reasons for YAML 1.2:

- The values in a file are typed almost as on the command line
  ([cli.md](cli.md)). With YAML 1.1, `yes`, `on` and `1e-3` would mean one
  thing in a file and another on the command line.
- A key that is given twice is a mistake, and is reported.

ruamel.yaml is pure Python and has no required dependencies. It is slower
than PyYAML's C loader (38 ms against 3 ms for a file of 1,200 lines), which
does not matter for config files.

A config file carried over from Hydra can change its meaning where it relied
on YAML 1.1, such as `on` for a boolean.

### Config files can also be JSON5

A file ending in `.json5` is read as [JSON5](https://json5.org/), with the
[`json5`](https://pypi.org/project/json5/) package. It holds what a YAML file
holds, with the same keys: `_defaults_`, `_doc_`, the values and `${...}`
strings.

```json5
// configs/experiments/cmnist_cnn.json5
{
  _defaults_: {dm: 'cmnist', model: 'cnn', opt: 'weight_decay_adamw'},
  dm: {num_colors: 10},
}
```

- Whether a file is YAML or JSON5 makes no difference anywhere else. A
  JSON5 file can extend a YAML file with `_self_` and the other way round,
  and `--config-name`, `+NAME`, `--help` and the validator treat both alike.
- `json5` is an optional extra: `pip install 'nematocyte[json5]'`. It is
  imported only when a `.json5` file is read; finding and listing files
  does not need it. Reading a `.json5` file without it is an error that
  names the file and the extra.
- A key given twice is an error, as in YAML.
- An empty `.json5` file is not valid JSON5, and is an error. (An empty YAML
  file is an empty mapping.) The top level has to be an object.
- `Infinity` and `NaN` are floats, and `0x10` is an int. pydantic then
  converts the values for the fields, as for YAML.
- JSON5 has no dates, so `'2026-11-16'` is a string. A field typed
  `datetime.date` gets the same date from either format, since pydantic
  converts the string. A field typed `Any` gets the string `'2026-11-16'`
  from either format, as every value reaches pydantic in JSON form
  ([deserializer.md](deserializer.md#details-of-building)). A field typed
  `str` takes the JSON5 string, while in YAML the date has to be quoted.
- `--cfg` prints YAML whichever format the files are in.

`json5` was chosen over `pyjson5`, which is faster, because `pyjson5` cannot
report a key given twice and is compiled, so it needs a wheel for each new
Python version. Reading config files does not need to be fast (see above).

### Details of finding and reading files

- The config directory has to exist. One that does not is an error straight
  away.
- A config file is read once per start of the program: every config that
  is composed from then on, such as each job of a multirun, uses what the
  file held when it was first read. `nematocyte.compose` reads the files
  again on each call, and a `Validator` once for all the files it checks.
- Config files end in `.yaml`, `.yml` or `.json5`. `--config-name` accepts a
  name with or without the suffix. Two files whose names differ only in the
  suffix, such as `cnn.yaml` and `cnn.yml`, or `cnn.yaml` and `cnn.json5`, in
  the same directory, are an error, since the name is ambiguous.
- The name of a config is the path of its file, without the suffix:
  relative to the config directory for a primary config, and to the
  field's directory for a named config. Each part between slashes has to be
  a plain file name on every platform: not empty, not `.` or `..`, and
  without a backslash or a drive (`C:`). So `dm=../blond`, `dm='..\blond'`
  and `--config-name /etc/x` are errors, and a name never reaches a file
  outside the directory it is looked up in, on Linux, macOS or Windows. Any
  other character is allowed, since a quoted override or a name in a file
  can be any string: `dm='größe 2'` selects `configs/dm/größe 2.yaml`.
  A file whose name breaks the rule, such as `configs/x:run.yaml`, cannot
  be selected. `--help` leaves it out, and the validator reports it.
- A key with nothing under it (`model:` and then only comments) is null to
  YAML, the same as `model: null`. Among the values, null is an error for
  anything that holds a mapping and does not take `None`: a subconfig or a
  group, optional or not, and a value whose type is a dict without
  `| None`. So commenting out every line under a key never changes the
  config silently. The error names the file, the key and the class, and
  shows what may have been meant: leaving out the key; for an optional
  subconfig or group, switching it off under `_defaults_`
  ([unions.md](unions.md#optional-subconfigs)); for a TypedDict, making it
  empty with `{_replace_: true}`
  ([above](#_replace_-true-replaces-instead-of-merging)); for a dict, making
  it empty with `{}`. For a value whose
  type takes `None`, such as `dict[int, int] | None` or `Any`, null is
  `None`, as in Hydra.
- In the file that switches an optional subconfig or group off under
  `_defaults_`, a value for it among the values is an error that names the
  file, the key and the entry.
- Among the values, a mapping for an optional subconfig that is off is an
  error that names the file, the key and the class, and shows the entry
  with `_some_` under `_defaults_`
  ([unions.md](unions.md#optional-subconfigs)).
- A selection written among the values (`model: cnn` outside `_defaults_`)
  is an error that points to `_defaults_`.
- A key `defaults`, as in Hydra, is an error that suggests `_defaults_`,
  unless the node has a field of that name.
- `_defaults_` as a list, as in Hydra, is an error that shows the mapping
  form. An entry with `null` switches off an optional subconfig or group
  (`ema: null`, `model: null`, [unions.md](unions.md#optional-subconfigs)),
  which Hydra's `- model: null` does not do. For anything else, it is an
  error that says it is not needed: Hydra needs `- opt: null` only to list
  the group, so that `opt=NAME` works without `+` ([cli.md](cli.md)), and
  nematocyte reads that from the type of the field. `_self_: null` is an
  error.
- A key `_doc_` at the top level of a file describes the file, for
  `--help` ([helptext.md](helptext.md#docs-of-config-files)). It is not a
  value. It has to be a string; anything else is an error that names the
  file. A `_doc_` below the top level is an error that says it belongs at
  the top. A file does not take the `_doc_` of the config it extends with
  `_self_`.
- `nematocyte=NAME` selects a preset for all of nematocyte's settings, from a
  file directly in `configs/nematocyte/`. This follows from the general rule.
