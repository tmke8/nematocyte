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
`config_name` as the entry point.

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
- A class that is an option of a group cannot have a field called
  `_option_`.
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
| Enum member | Its value |

The rule is that the dict holds what a config file would hold. For an enum
that is the value, because pydantic reads enums by value
([deserializer.md](deserializer.md)). `hydra-qol` uses the name instead,
because Hydra reads enums by name.

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
nematocyte.validate_config(Config, config_dir, path)
```

See [validation.md](validation.md).

### Details

- The names that the other documents do not give: `nematocyte.ConfigError`
  for every error in a configuration, `nematocyte.SchemaError` (a subclass)
  for a config class that is declared in an unsupported way,
  `nematocyte.ValidationResult` with `path`, `error` and `ok`,
  `nematocyte.EntryPoint` for the type of the decorated function, and
  `nematocyte.Option` for what `option()` returns.
- `to_dict`: a value of an unknown type is left as it is; a set is sorted if
  it can be; a dataclass inside a list or dict becomes a dict without
  `_option_`; an instance of a subclass of an option gets the name of that
  option.
