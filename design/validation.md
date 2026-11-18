# Validation

How to check that every YAML file is still valid, without running an
experiment.

## Goal

YAML files are not seen by the type checker. If a field of a dataclass is
renamed, a file that still uses the old name only fails when someone uses it.
A test should catch that.

Hydra has nothing for this. `hydra-example` uses `hydra-qol`
(`tests/test_configs.py`):

```python
validate_configs(
    main_cls,
    groups,
    *,
    schema_name,
    config_dir,
    config_with_defaults=None,
    primary_config_dirs=(),
) -> list[CheckResult]
```

It composes and instantiates every YAML file in `config_dir`.

## Design

### Run the real loader on every file

```python
nematocyte.validate_configs(Config, config_dir)
```

Each file goes through the same steps as at runtime: composition,
interpolation, and building the dataclass instances. The verdict is then the
one a real run would give. That includes `__post_init__` checks and
pydantic's conversions, which no separate description of the rules would
reproduce exactly.

Nothing is run beyond building the config object.

### Fewer arguments than `hydra-qol`

| `hydra-qol` argument | Why nematocyte does not need it |
| --- | --- |
| `groups` | The options are in the annotations of the root dataclass ([unions.md](unions.md)) |
| `schema_name` | The schema is always enforced; there is no schema entry ([composition.md](composition.md)) |
| `primary_config_dirs` | A directory that matches no field can only hold primary configs ([composition.md](composition.md)) |
| `config_with_defaults` | The rest of the tree comes from the defaults of the root dataclass; see below |

### What is checked for each kind of file

The kind follows from where the file is ([composition.md](composition.md)).

- **A primary config** is composed as if it had been passed to
  `--config-name`.
- **A named config for a group** (`configs/dm/celeba_male_smiling.yaml`) is
  selected for its field, and its chain of `_self_` entries has to end at
  an option from the annotations.
- **A preset for a plain subconfig** (`configs/opt/weight_decay_adamw.yaml`)
  is selected for its field.
- **Files under `configs/nematocyte/`**, the launcher and sweeper configs,
  are checked the same way against nematocyte's own schema.

Files that belong nowhere are reported: a file in a directory named after a
field that is not a named config for it, or a subdirectory of a group's
directory that matches no field.

### A named config is validated in the whole tree

A file such as `celeba_male_smiling.yaml` describes one node. To check it,
the whole tree is built with that one selection made, and everything else
left at its default.

The whole tree is needed because an interpolation can refer to any value in
it. A file in `configs/dm/` may use `${seed}` or `${opt.lr}`, which only
resolve in a complete tree. The `__post_init__` of the root dataclass also
only runs on a complete tree.

Since a primary config is optional ([composition.md](composition.md)), the
rest of the tree normally comes from the defaults of the root dataclass. If
some field has no default, a primary config that supplies it has to be named,
as with `config_with_defaults` in `hydra-qol`.

Building the tree once per file is more work than checking the node alone,
but the tree is small and nothing is run.

A named config for a nested field of a group, such as a file in
`configs/model/norm/`, is checked once for each option of the group that
has the field: with the option that is selected, if it has it, and with
every other one, selected by name. Two options can have a field of the same
name with different classes, and the file can be selected under either, so
it has to fit each of them. The error says which selection it failed with.

### `__post_init__` runs, and checks the config only

Validation builds the full objects by default, so every `__post_init__`
runs. A check such as "`val_prop` is between 0 and 1" depends only on the
config, and a file that violates it is broken on every machine.

This relies on a convention:

- `__post_init__` checks the config only. Anything it tests follows from the
  field values.
- Checks against the environment go where the work starts, such as the
  beginning of `run()`.

`Config` in `hydra-example` breaks the convention. It checks the GPU index
against `torch.cuda.device_count()`, which gives a different answer on the
test machine and fails in CI. The `gpu < -1` half belongs in
`__post_init__`; the device count belongs in `run()`.

For code that cannot follow the convention yet, there is an opt-in switch
that stops before the constructors:

```python
nematocyte.validate_configs(Config, config_dir, construct=False)
```

Composition, interpolation,
the checks for unknown and missing keys, and pydantic's check of every field
value still happen; only the final `cls(**kwargs)` is skipped, and with it
every `__post_init__`. This is possible because nematocyte walks the
dataclass nodes itself ([deserializer.md](deserializer.md)).

### One result per file

The function returns one result per file, with the path and the error if
there is one, so that a test can report each broken file separately.

For parametrised tests such as the one in `hydra-example`, with one test case
per file, a validator checks one file at a time:

```python
validator = nematocyte.Validator(Config, config_dir)
result = validator.validate(path)
```

The work that does not depend on the file is done once, when the validator is
made: the check of the config classes and the check of the files at the top
of the config directory. A test makes the validator once, in a fixture with
`scope="module"`, and calls `validate` in each test case. `validate_configs`
uses one validator for all the files.

Errors in that shared work are raised by the constructor, and by
`validate_configs`, instead of being returned as a result: a config class
declared in an unsupported way, a config directory that does not exist, or a
file at the top named after a field. They are not the fault of any one file.

### Sweeper parameters are checked

The values under `params` in a sweeper config are override expressions:

```yaml
params:
  opt.lr: tag(log, interval(1.e-5, 1.e-3))
  model.num_hidden: range(1, 4)
```

Validation checks two things:

- Each value is parsed with the override parser ([cli.md](cli.md)).
- Each key has to name a value in the schema. Where the path goes through a
  group, it is enough that one option of the group has the field. The file
  does not say which option of `model` will be selected, and
  `model.num_hidden` exists only for `fcn`.

The exact check, against the options that are selected, is made at the start
of a multirun, before any job is submitted ([multirun.md](multirun.md)).

### Details

- `validate_configs` and `Validator` take `config_name`, for the primary
  config that supplies the rest of the tree, and `construct`.

## Rejected ideas

### Building only the node a file describes

For a named config, build just the dataclass the file describes
(`CelebADataModule` for a file in `configs/dm/`) and leave the rest of the
tree out. This needs no defaults for the other fields.

**Rejected because** an interpolation that points outside the node, such as
`${seed}`, cannot be resolved, and the `__post_init__` of the root dataclass
does not run.

### Validating against JSON schemas from pydantic

pydantic can generate a JSON schema for a dataclass, and a YAML file can be
checked against it. A prototype with the `hydra-example` dataclasses:

| YAML | JSON schema | Real loader |
| --- | --- | --- |
| A valid `dm` file | ok | accepts |
| `superclas: Smiling` | "Additional properties are not allowed ('superclas' was unexpected)" | rejects |
| `superclass: Smilin` | "'Smilin' is not one of ['Blond_Hair', 'Male', 'Smiling']" | rejects |
| `lr: 1e-3` | "'1e-3' is not of type 'number'" | accepts |
| `lr: ${some.ref}` | "is not of type 'number'" | depends on the target |

**Rejected because** a schema is a second, approximate statement of the
rules, and it disagrees with the loader:

- PyYAML, which the prototype used, reads `1e-3` as a string. pydantic's lax
  mode converts it; the schema rejects it. This point no longer applies now
  that config files are read as YAML 1.2
  ([composition.md](composition.md)); the others do.
- An interpolation in a non-string field fails, unless every field is
  loosened to allow strings as well.
- `__post_init__` checks cannot be expressed.
- Unknown keys are not caught by default. pydantic leaves
  `additionalProperties` unset for a stdlib dataclass, and it had to be set
  to `false` by hand.
- Which schema applies depends on composition: the class a file's `defaults`
  chain ends at, or the option selected for a group in a primary config.
  nematocyte has to work that out first anyway.

Generated schemas may still be useful for editors, where the YAML language
server uses them for completion and inline errors. That would be a separate
feature and is not planned.
