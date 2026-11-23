# Help text

How `--help` can explain what each config value is for.

## The problem

Hydra's `--help` prints the groups and the config with its default values,
but nothing about what the values mean. There was no agreed place to put a
description of a dataclass field.

Python 3.14 adds one: `dataclasses.field(doc=...)`. This is why nematocyte
requires Python 3.14.

```python
@dataclass(kw_only=True)
class Config:
    seed: int = field(default=42, doc="Random seed.")
    gpu: int = field(
        default=-1, doc="Index of the GPU to use. Set to -1 to use the CPU."
    )
```

## What was tested

Experiments on Python 3.15 and 3.14, with throwaway scripts.

| Question | Result |
| --- | --- |
| Is the doc available at runtime? | Yes, as `f.doc` for each `f` in `dataclasses.fields(cls)`; `None` if not given |
| Does it combine with `default`, `default_factory`, `kw_only`, and a field with no default? | Yes |
| Is it inherited? | Yes, as long as the subclass does not declare the field again |
| What if a subclass declares the field again, to change the default? | The doc is lost: `default_res: int = 28` in a subclass has `doc=None` |
| `InitVar` fields | Not in `dataclasses.fields()`, but the doc is in `cls.__dataclass_fields__[name].doc` |
| Does Python's own `help()` show it? | Only with `@dataclass(slots=True)`, where `__slots__` becomes a dict of the docs |
| Do type checkers accept it? | ty and pyright do, when checking for Python 3.14. ty rejects `doc` when checking for 3.13 |
| Does pydantic see it? | Yes. pydantic 2.14 puts it into the JSON schema as the field's `description`, and the class docstring as the description of the class |
| Does tyro use it? | Yes, for its own `--help` |

## Where each piece of help text comes from

| Text | Source |
| --- | --- |
| What a value is for | `field(doc=...)` on the field |
| What a group is for | `field(doc=...)` on the group field |
| What an option of a group is | The docstring of the option's class |
| What a subconfig is for | `field(doc=...)` on the field, else the docstring of its class |
| What the program does | The docstring of the root dataclass |
| What a config file is | The `_doc_` key of the file ([below](#docs-of-config-files)) |
| The type and the default | The dataclass itself; nothing to write |

Every one of these is optional. A field without a doc is shown without one.

nematocyte's own settings under the reserved key are dataclasses too, so the
launcher and sweeper settings get help text the same way.

## Things to handle

- **A field declared again in a subclass.** To avoid losing the doc, look
  the field up along the class's MRO and take the first doc found. In the
  prototype this gave `ColoredMNISTDataModule.default_res` the doc from its
  base class.
- **Generated class docstrings.** A dataclass without a docstring gets one
  generated from its signature, such as `NoDocstring(x: int = 1)`. A subclass
  does not inherit its base's docstring either, so it gets a generated one
  too. These have to be recognised and ignored; they start with the class
  name followed by `(`.
- **Long docs.** Only the first paragraph belongs in an overview.
- **The cost in the code.** A field with a plain default has to be rewritten
  to carry a doc: `gpu: int = -1  # Set to -1 to use CPU.` becomes
  `gpu: int = field(default=-1, doc="Set to -1 to use the CPU.")`.

## What the help could look like

Two layouts were prototyped on a documented copy of the example's `Config`,
both built from the same walk over the config that the deserializer needs
anyway ([deserializer.md](deserializer.md)).

### One override per line

```text
Train a model on a dataset.

dm=cmnist                The dataset to train on.
                           celeba   The CelebA dataset of celebrity faces.
                         * cmnist   MNIST with digits coloured according to their label.
dm.default_res=28        [int] Resolution to which the images are resized.
dm.colors=null           [list[int] | null] Indices of the colours to use.
dm.val_prop=0.2          [float] Proportion of the training data that is held out
                         for validation.

model=fcn                The model architecture.
                         * fcn      Fully-connected network.
                           cnn
model.num_hidden=1       [int] Number of hidden layers.
model.activation=gelu    [relu|gelu|selu|silu] Activation function.

opt:  Settings for the optimiser.
opt.lr=5e-05             [float] Learning rate.
seed=42                  [int] Random seed.
gpu=-1                   [int] Index of the GPU to use. Set to -1 to use the CPU.
```

The left column is exactly what is typed on the command line, with the
current value. A group shows its options, with the selected one marked. The
type column shows the allowed values of a `Literal`.

Its weakness is width: one long default, such as a path, widens the left
column for every line.

### YAML with comments

```yaml
# Train a model on a dataset.

# The dataset to train on. Options: celeba, cmnist. Selected: cmnist.
dm:
  # Resolution to which the images are resized. (int)
  default_res: 28
  # Proportion of the training data that is held out for validation. (float)
  val_prop: 0.2
# Random seed. (int)
seed: 42
```

This is close to what Hydra prints, and it is valid YAML, so it can be pasted
into a config file as a starting point. The docs have the full width. It does
not show what to type for a nested key, and a group's options do not fit
well.

### Decided

The first layout is used for `--help`, which answers "what can I type?". The
second is used for `--cfg`, which prints the config ([cli.md](cli.md)), with
the docs as comments.

- When `key=value` is longer than the left column, the description starts on
  the next line, as in argparse. One long default then does not widen the
  column for every line.
- As in Hydra, `--help` reflects the other arguments: `model=cnn --help`
  shows the fields of the CNN.
- The config files of a group, such as `single_linear_layer`, are listed
  after its options, one per line under `From files:`, with their docs
  ([below](#docs-of-config-files)). The configs in the directory of an
  option are among them, as `celeba/male_blond`. The presets of a plain
  subconfig are listed the same way, under `Presets:`.
- `--help` also shows nematocyte's own settings and the flags (with
  `+NAME` and `~KEY`), after the
  program's keys, as a block of their own, so that their keys do not widen
  the left column. The left column is at most 40 characters wide, and the
  text is wrapped at the width of the terminal, between 60 and 100 columns
  (88 when there is no terminal).
- `--cfg` always prints the docs as comments; there is no switch. A group is
  shown with its options and the selected one.
- A value that is required and missing is shown as `???`, in `--help` and in
  `--cfg`.
- A key of a `TypedDict` has no doc: there is no `field(doc=...)` for it,
  and no other way to attach one. Its type is still shown. The section of a
  TypedDict subconfig uses the class docstring, as for a dataclass.
- A `NotRequired` key of a TypedDict that nothing set, or that `~KEY` left
  out, is shown as `key=` with `(not set)` in `--help`, and as a comment
  `# key: not set` in `--cfg`. In `--cfg`, a subconfig with nothing but such
  keys is `{}`.
- Python 3.15 generates dataclass docstrings lazily, through an object that
  is not a string. The check for generated docstrings handles both.

## Other ways to attach a doc

tyro reads three sources. All three showed up in its `--help` in the test:

```python
num_hidden: int = field(default=1, doc="From field(doc=...).")
activation: Activation = "gelu"  # From a trailing comment.
dropout: float = 0.0
"""From a string after the field."""
```

The last two are not available at runtime. tyro gets them by reading the
source code of the class, which only works where the source can be found.

**Decided:** nematocyte reads `field(doc=...)` only. It is one documented
mechanism, it needs no source code, and pydantic and type checkers understand
it.

## Docs of config files

A config file describes itself in a key `_doc_` at its top level:

```yaml
_doc_: A single linear layer.
_defaults_:
  _self_: fcn

num_hidden: 0
```

### Decided

- `_doc_` is not a value of the config. It has to be a string; anything else
  is an error that names the file. It is allowed only at the top level of a
  file, in every kind of config file: a primary config, a named config of a
  group, a preset, and a file under `configs/nematocyte/`. A `_doc_` below
  the top level is an error that says where it belongs.
- `_doc_` is reserved: no config class can have a field of that name
  ([deserializer.md](deserializer.md#reserved-names)).
- A file does not take the doc of the config it extends with `_self_`; each
  file describes itself.
- Only the first paragraph is shown, as for docstrings.
- The docs are shown in `--help` only. `--cfg` prints the composed config,
  which is not one file.
- `--help` lists the files of a group or a subconfig one per line, each
  with its doc; a file without a `_doc_` is listed by its name:

  ```text
  model=fcn                The model architecture.
                           * fcn      Fully-connected network.
                             cnn
                           From files:
                             single_linear_layer   A single linear layer.
  ```

- `--help` also lists the primary configs, which `--config-name` and `+NAME`
  take, with their docs, in a block of their own after the program's keys
  and before nematocyte's own settings:

  ```text
  Primary configs, for --config-name NAME or +NAME:
    experiments/cmnist_cnn   A CNN on ColoredMNIST.
    good_run
    presets/debug            Log locally and verbosely, for debugging.
  ```

- `--help` reads every file it lists. A file that cannot be read, or whose
  `_doc_` is not a string, makes `--help` an error, as it would make a run
  that selects the file. `validate_configs` finds such a file in a test.

A comment at the top of the file was considered instead of a key, but
ruamel.yaml's safe loader drops comments, and the comments in a config file
often explain the YAML below them rather than describe the file.

## Docs in error messages

Decided:

- For a value that is not valid (`opt.lr=abc`), the error includes the doc of
  that field.
- For an unknown key (`model.hiden_dim=10`), the error lists the valid names
  and the closest match, without docs.

The wording of the messages is left to the implementation.

## Not in the first version

- **Checking that fields are documented.** A helper for tests, next to
  `validate_configs`, that reports fields without a doc.
