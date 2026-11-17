# Differences from Hydra

What works differently in nematocyte, for someone who knows Hydra. Each
entry says what changes in a config file or on the command line, and links
to the design document that explains why.

## Values

### Enums are read by value, not by name

With an enum such as

```python
class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"
```

a field of type `Color` is set to `BLUE` like this:

| Written | Hydra (OmegaConf 2.4) | nematocyte |
| --- | --- | --- |
| `color=BLUE`, the name | Accepted | Error: "Input should be 'red' or 'blue'" |
| `color=Color.BLUE` | Accepted | Error |
| `color=blue`, the value | Accepted | Accepted |
| Printed by `--cfg`, saved with the config | `BLUE` | `blue` |

For an enum with integer values (`LOW = 1`), nematocyte accepts `1` and
not `LOW`.

nematocyte checks values with pydantic, which reads enums by value
([deserializer.md](design/deserializer.md)). `--help` lists the values that
a field accepts, and `nematocyte.to_dict` writes the value, so that what it
gives is what a config file would hold ([api.md](design/api.md)).

To keep config files that use the names, give the members their names as
values:

```python
class Color(enum.StrEnum):
    RED = "RED"
    BLUE = "BLUE"
```

For a fixed set of strings, a `Literal["red", "blue"]` works as well, and
needs no class.
