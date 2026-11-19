# Python API

What a program imports from nematocyte.

## Decided

### The entry point

```python
@nematocyte.main(Config, config_path="configs")
def main(config: Config) -> float:
    return config.run()


if __name__ == "__main__":
    main()
```

- The root dataclass is passed to the decorator. It is the schema that every
  config is checked against ([composition.md](composition.md)), and a type
  checker can verify that the function takes that class. The root can also
  be a `TypedDict` ([deserializer.md](deserializer.md#typeddict-configs)).
  `main` and `compose` take the class as a `TypeForm[C]`
  (`typing_extensions`), which every checker accepts for a TypedDict as
  well as a dataclass; anything else is a `SchemaError` at run time.
- `config_path` is required, and is relative to the file that defines the
  function. Its name and its meaning are Hydra's. The directory has to
  exist, even if it is empty.
- `config_name` is optional and names the default primary config. Without
  it there is none, and the tree comes from the root dataclass alone
  ([composition.md](composition.md)).
- The return value is the objective for a sweeper
  ([multirun.md](multirun.md)). Otherwise it is ignored.

The decorated function takes an optional list of arguments, which replaces
`sys.argv[1:]`. It is for tests, which call the entry point directly:
`main(["seed=1", "model=cnn"])`. Without it, the arguments come from the
command line. The decorated function returns `None`: the return value of
the function inside is the objective, and in a multirun it is handed to the
sweeper inside nematocyte, one value per job. A test that needs the values
reads them from the function it decorates.

### Composing without the command line

```python
config = nematocyte.compose(Config, config_dir, overrides=["model=cnn", "seed=1"])
```

Returns the config object, for tests and notebooks. It takes the same
`config_name` as the entry point. `overrides` takes the arguments as they
are written on the command line, `+NAME` included
([cli.md](cli.md#name-applies-a-primary-config-on-top)).

Unlike `config_path` in the entry point, `config_dir` is optional here;
without it there are no config files. A relative path is relative to the working directory.

### A dict for logging

```python
nematocyte.to_dict(config)
```

Returns the config as plain dicts, for logging to a tool such as W&B.

A dataclass instance does not say which option of a group it is, so the dict
for a group field records it under the key `_option_`:

```python
{
    "model": {"_option_": "fcn", "num_hidden": 1, "activation": "gelu"},
    "opt": {"lr": 5e-05},
    "seed": 42,
}
```

- The name is the option of the instance's class. If the config was selected
  with `model=single_linear_layer`, the dict still says `fcn`: the object
  does not know which file its values came from.
- A plain subconfig such as `opt` gets no `_option_` key.
- No config class can have a field called `_option_`
  ([unions.md](unions.md#reserved-names)).
- The root has to be a dataclass; a TypedDict root is a `TypeError`.
- A TypedDict below it, whether a plain subconfig or an option of a group,
  is kept as it is, apart from converting its values as in the table below.
  An option of a group already holds its `_option_`
  ([unions.md](unions.md#typeddicts-as-options)), and a `NotRequired` key
  that was not set is not there.
- So every TypedDict that can be reached from the root may hold only
  values. One with a subconfig or a group is a `TypeError`, which names the
  TypedDict and the key. The schema is checked before anything is
  converted.

Values that are not plain YAML types are converted, so that the dict can
always be dumped as YAML or JSON:

| Value | In the dict |
| --- | --- |
| `Path` | `str` |
| `tuple`, `set` | `list` |
| `frozendict` | `dict` |
| Enum member | Its value |

The rule is that the dict holds what a config file would hold. For an enum
that is the value, because pydantic reads enums by value
([deserializer.md](deserializer.md)). `hydra-qol` uses the name instead,
because Hydra reads enums by name.

### The config of a run, as plain dicts

```python
wandb.init(config=nematocyte.run_config())
```

Returns the config of the current job as plain dicts, lists and scalars:
exactly what its `.nematocyte/config.json` holds
([outputs.md](outputs.md#what-nematocyte-writes)), whether or not the file
has been written yet.

- It is the resolved tree that the job's config object was built from, not
  a conversion of that object. So unlike `to_dict`, it works for a
  TypedDict root and for TypedDicts that hold subconfigs or groups, it has
  the values of `InitVar` fields, and the option of a group is the one that
  was selected.
- The option of each group is under `_option_`, as in `to_dict`.
- It holds only JSON values: a value that JSON does not have, such as a
  date, is its text, and so is a dict key that is not a string. Enums are
  their values and paths are strings, as in `to_dict`.
- It is typed as `dict[str, JSONValue]`, which says just that. A nested
  value has to be narrowed before it is used, or the result cast to
  `dict[str, Any]`.
- nematocyte's own settings, the key `nematocyte`, are not part of it.
- It returns a new copy on every call, which can be changed freely. Changing
  it changes neither the config of the job nor `config.json`, even if the
  file is written later.
- Outside a run it is an error, as for `output_dir()`. `to_dict` is for a
  config outside a run, such as one from `compose`.

### Reading the config of a run back

```python
data = json.loads((run_dir / ".nematocyte" / "config.json").read_text())
config = nematocyte.from_run_config(Config, data)
```

Returns the config object that the function of the run received, from the
`config.json` that the run saved, or from what `run_config()` returned, for
example to load a checkpoint with the model it was trained with. It also
takes the output of `to_dict`, which leaves out the values of `InitVar`
fields; those then take the default in the class.

- `data` is a `Mapping[str, JSONValue]`, where
  `type JSONValue = Mapping[str, JSONValue] | list[JSONValue] | str | int |
  float | bool | None` is `nematocyte.JSONValue`. A mapping need not be a
  `dict`, so a `frozendict` or a `MappingProxyType` is accepted.
- It takes the parsed data, not the path of the file.
- The mapping of a group names its option under `_option_`, next to the
  fields, as `to_dict` writes it. A mapping without `_option_`, or with a
  name that is not an option of the group, is an error that names the key,
  the class and the options. In a plain subconfig or at the root,
  `_option_` is an unknown key.
- A key that the mapping leaves out takes the default in the class, as it
  would without any config file: a value its class default, and a subconfig
  or a group the default of the field. A group that is left out and has no
  default is an error. An unknown key is an error, as in a
  config file, and so is the key `nematocyte`: the settings are not part of
  the saved config.
- Every value is checked and converted as a value from a config file is, and
  every `__post_init__` runs. The values that the JSON file holds as text,
  such as a path, a date, or a key of `dict[int, int]`, are read back by
  pydantic. Nothing is resolved: a `${...}` in a value is kept as it is.
- Errors in the structure (an unknown key, a group without an option, a
  subconfig that is not a mapping) are collected and reported together, and
  before any value is checked; the errors in the values are then collected
  as in `compose`.
- The root can be a dataclass or a TypedDict, as for `compose`. Unlike
  `to_dict`, nothing has to be flat.
- The job of a multirun is rebuilt from its resolved tree in the same way,
  with the same checks; that tree always has every key.
- A field whose type pydantic cannot read, such as `torch.device`
  ([deserializer.md](deserializer.md#details-of-the-walk)), cannot be read
  back: a value for it is an error, as in a config file. `config.json`
  holds such a value as its text, so its key has to be removed first, which
  gives the default in the class.

### The output directory

```python
checkpoint_dir = nematocyte.output_dir() / "checkpoints"
```

Returns the directory of the current job as a `Path`, and creates it if it
does not exist yet; see [outputs.md](outputs.md). Outside a run, such as
after `nematocyte.compose(...)`, it is an error.

### The number of the job

```python
number = nematocyte.job_num()
```

Returns the number of the current job as an `int`. The jobs of a multirun
are numbered from 0, in the order described in
[multirun.md](multirun.md), and a single run is job 0. The number is the
name of the job's output directory ([outputs.md](outputs.md)). It plays the
part of Hydra's `HydraConfig.get().job.num`. Outside a run it is an error.

### Logging

```python
nematocyte.configure_logging()
level = nematocyte.log_level()
```

`configure_logging(*, file=True)` sets up `logging` for the current job,
with the level of `nematocyte.log`; `log_level()` returns that level as a
level of `logging`. Both are only for use inside a run; see
[logging.md](logging.md).

### Declaring a group

`nematocyte.option("name")` names a member of an annotated union; see
[unions.md](unions.md).

### Validating config files

```python
nematocyte.validate_configs(Config, config_dir)
nematocyte.Validator(Config, config_dir).validate(path)
```

See [validation.md](validation.md).

### Details

- The names that the other documents do not give: `nematocyte.ConfigError`
  for every error in a configuration, `nematocyte.SchemaError` (a subclass)
  for a config class that is declared in an unsupported way,
  `nematocyte.DroppedValueWarning` for a value that a selection drops
  ([composition.md](composition.md)),
  `nematocyte.ValidationResult` with `path`, `error` and `ok`,
  `nematocyte.EntryPoint` for the type of the decorated function, and
  `nematocyte.Option` for what `option()` returns.
- `to_dict`: a value of an unknown type is left as it is; a set is sorted if
  it can be; a dataclass inside a list or dict becomes a dict without
  `_option_`; an instance of a subclass of an option gets the name of that
  option.
