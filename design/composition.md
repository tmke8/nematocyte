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
  the file is.
- A file in a directory whose path matches a field whose type is a dataclass
  or a group (see [unions.md](unions.md)) is a named config for that field:
  `configs/dm/` holds the options of the group `dm`, and `configs/opt/` the
  presets for the subconfig `opt`.

`configs/experiments/cmnist_cnn.yaml` is therefore a primary config with no
header.

What a file builds on is said in its `defaults`.

### `defaults` is a mapping

In Hydra `defaults` is a list, for reasons that no longer apply:

- The list is a merge order, and the entry `_self_` marks where the file's
  own values go in it.
- Some entries have no key, such as `- base_config` and, in a group file,
  `- celeba`.
- Some entries carry a keyword, such as `- override model: cnn`.

In nematocyte `defaults` is a mapping. Each key says where: a field of the
node that the file describes, a dotted path to a nested field, or `_self_`
for that node itself. Each value names the config to load there.

```yaml
# configs/experiments/cmnist_cnn.yaml, a primary config
defaults:
  dm: cmnist
  model: cnn
  opt: weight_decay_adamw

dm:
  num_colors: 10
```

```yaml
# configs/dm/celeba_male_smiling.yaml, a named config for `dm`
defaults:
  _self_: celeba

superclass: Smiling
```

- The order of the entries has no meaning. What is applied before what
  follows from the keys; see [Layers](#layers).
- A place cannot be given twice: the YAML parser reports a repeated key.
- A primary config describes the root, so `dm: cmnist` means the field `dm`
  of the root dataclass. There is no leading slash, and a primary config
  reads the same whichever directory it is in.
- `_self_` is reserved: no dataclass field can have that name. In Hydra the
  word means something else, as described above.

### `_self_` names the config that a file extends

`_self_` names another config for the same node. That config is loaded
first, and the `defaults` and values of this file apply on top.

| File | What `_self_` can name | Required |
| --- | --- | --- |
| A named config for a group | An option of the group, or another named config in the same directory | Yes; the chain has to end at an option ([unions.md](unions.md#defaults)) |
| A preset for a plain subconfig | Another preset in the same directory | No |
| A primary config | Another primary config, by the name that `--config-name` takes | No |

A chain that comes back to itself is an error.

A primary config that extends another one:

```yaml
# configs/experiments/good_run_blond.yaml
defaults:
  _self_: good_run
  dm: celeba_male_blond

seed: 2
```

`hydra-example` has neither a primary config that extends another, nor a
preset that extends another. Both were tested in Hydra on a copy of it.

### A key in `defaults` can be a path

Suppose the option `cnn` of `model` has a field `norm`, which is itself a
group. A primary config can select for both:

```yaml
defaults:
  model: my_cnn
  model.norm: my_batchnorm

model:
  norm:
    momentum: 0.2
```

`model.norm: my_batchnorm` loads `configs/model/norm/my_batchnorm.yaml`
(see below for the directory). It is an error if the option selected for
`model` has no field `norm`.

A named config can do the same for its own fields. The keys other than
`_self_` are relative to the node that the file describes:

```yaml
# configs/model/my_cnn.yaml
defaults:
  _self_: cnn
  norm: layernorm
```

### Layers

Everything that contributes to the config is a layer, and a higher layer
always wins. For one file, from lowest to highest:

1. The defaults of the dataclasses.
2. The config that the file names with `_self_`, composed completely,
   including its own `_self_`.
3. The other entries in the file's `defaults`, shortest path first.
4. The values in the file.

The command line comes on top of the primary config, in the same way: first
its selections, shortest path first, then its values.

The order in which things are written, in a file or on the command line,
never matters.

**Shortest path first.** `model.norm` is inside `model`, so it is the more
specific entry and is applied later. In the example above, `my_batchnorm`
wins over what `configs/model/my_cnn.yaml` selects for `norm`, and
`momentum: 0.2` is then set on it. Two paths of which neither contains the
other cannot conflict.

**What a selection does to the node** depends on what is selected:

- An option of a group, or a named config for one, replaces the node. It
  starts again from the class defaults of the option
  ([unions.md](unions.md#defaults)), and everything that lower layers
  selected or set under it is dropped.
- A preset for a plain subconfig sets its values on top. The other values
  that lower layers set at that node stay.

With `good_run.yaml`, which sets `model.activation: selu`,
`dm.download: false` and `opt.weight_decay: 0.001`:

| Applied on top of `good_run` | Result |
| --- | --- |
| `opt: weight_decay_adamw`, in a file that extends it or as `opt=weight_decay_adamw` | `opt.weight_decay` is `0.01`, from the preset |
| `model: cnn`, or `model=cnn` | A CNN with its class defaults. `activation: selu` is dropped |
| `dm: celeba_male_blond` | `download` is back at its default. To keep `false`, it is set again |

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
  selected. `--config-name good_run model=cnn` gives a CNN with
  `activation: selu`. If the parent sets a value that the new class does not
  have, such as `model.num_hidden`, the result is an error, and the child
  has no way to remove the value.

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
one option of `model`. Anything else is an error, because its files could be
neither named configs nor primary configs.

### The schema is always enforced

The root dataclass is given to the entry point, and every primary config is
validated against it. There is no schema entry in `defaults`.

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

### Details of finding and reading files

- The config directory has to exist. One that does not is an error straight
  away.
- Config files end in `.yaml` or `.yml`. `--config-name` accepts a name with
  or without the suffix. Two files whose names differ only in the suffix,
  such as `cnn.yaml` and `cnn.yml` in the same directory, are an error,
  since the name is ambiguous.
- A key of a subconfig or group with nothing under it (`model:` and then
  only comments, which YAML reads as null) is ignored.
- A selection written among the values (`model: cnn` outside `defaults`) is
  an error that points to `defaults`.
- `defaults` as a list, as in Hydra, is an error that shows the mapping
  form. An entry with `null` (Hydra's `- opt: null`) is an error that says
  it is not needed.
- `nematocyte=NAME` selects a preset for all of nematocyte's settings, from a
  file directly in `configs/nematocyte/`. This follows from the general rule.
