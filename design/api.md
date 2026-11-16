# Python API

What a program imports from nematocyte.

## Decided

### The entry point

```python
@nematocyte.main(Config, config_dir="configs")
def main(config: Config) -> float:
    return config.run()


if __name__ == "__main__":
    main()
```

- The root dataclass is passed to the decorator. It is the schema that every
  config is checked against ([composition.md](composition.md)), and a type
  checker can verify that the function takes that class.
- `config_dir` is relative to the file that defines the function, as in
  Hydra.
- `config_name` is optional and names the default primary config. Without
  it there is none, and the tree comes from the root dataclass alone
  ([composition.md](composition.md)).
- The return value is the objective for a sweeper
  ([multirun.md](multirun.md)). Otherwise it is ignored.

### Composing without the command line

```python
config = nematocyte.compose(Config, config_dir, overrides=["model=cnn", "seed=1"])
```

Returns the config object, for tests and notebooks. It takes the same
`config_name` as the entry point.

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

### Declaring a group

`nematocyte.option("name")` names a member of an annotated union; see
[unions.md](unions.md).

### Validating config files

```python
nematocyte.validate_configs(Config, config_dir)
nematocyte.validate_config(Config, config_dir, path)
```

See [validation.md](validation.md).
