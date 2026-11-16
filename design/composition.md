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

A named config for a group has to say in its `defaults` which option it
extends. A preset for a plain subconfig does not. What each of them builds
on is described in [unions.md](unions.md#defaults).

### Entries in a primary config's `defaults` are relative to the root

A primary config describes the root, so `- dm: cmnist` means the field `dm`
of the root dataclass. There is no leading slash, and a primary config reads
the same whichever directory it is in.

`experiments/cmnist_cnn.yaml` becomes:

```yaml
defaults:
  - dm: cmnist
  - model: cnn
  - opt: weight_decay_adamw

dm:
  num_colors: 10
```

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
