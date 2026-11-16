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
  is a named config for that field: `configs/dm/` holds the options of the
  group `dm`, and `configs/opt/` the presets for the subconfig `opt`.

`configs/experiments/cmnist_cnn.yaml` is therefore a primary config with no
header.

### Entries in a primary config's `defaults` are relative to the root

A primary config describes the root, so `- dm: cmnist` means the field `dm`
of the root dataclass. There is no leading slash, and a primary config reads
the same whichever directory it is in.

`experiments/cmnist_cnn.yaml` becomes:

```yaml
defaults:
  - base_config
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

## Open questions

- **The schema entry.** In `hydra-example` every primary config starts its
  `defaults` with `- base_config` (or `- /base_config`), the name under which
  the root dataclass was registered. If the root dataclass is given to the
  entry point, this entry may not be needed.
- **A subdirectory with the name of a field.** A primary config cannot live
  in `configs/model/`. Whether that is an error or merely a consequence to
  document.
- **Named configs below the first level.** For a dataclass field nested
  inside `model`, the directory would be `configs/model/NAME/`, next to the
  files that are options of `model` itself. `hydra-example` only has this
  under `hydra/`.
