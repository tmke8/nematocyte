# nematocyte

A configuration library for Python programs: a replacement for Hydra with
fewer features and more focus on usability. The config is a tree of stdlib
dataclasses (or `TypedDict`s), filled from YAML files and command-line
overrides.

```python
from dataclasses import dataclass, field
from typing import Annotated

import nematocyte
from nematocyte import option


@dataclass(kw_only=True)
class Fcn:
    """Fully-connected network."""

    num_hidden: int = field(default=1, doc="Number of hidden layers.")


@dataclass(kw_only=True)
class Cnn:
    """Convolutional network."""

    kernel_size: int = field(default=5, doc="Kernel size of the convolutions.")


@dataclass(kw_only=True)
class Config:
    """Train a model."""

    model: Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")] = field(
        default_factory=Fcn, doc="The model architecture."
    )
    seed: int = field(default=42, doc="Random seed.")


@nematocyte.main(Config, config_path="configs")
def main(config: Config) -> None:
    print(config)


if __name__ == "__main__":
    main()
```

```bash
python main.py model=cnn model.kernel_size=3 seed=1
python main.py --help
python main.py --multirun seed=1,2,3
```

- [`example/`](example/) is a complete project with an annotated README.
- [`design/`](design/) holds the design documents, which are the
  specification.
- [`DIFFERENCES.md`](DIFFERENCES.md) lists what works differently from
  Hydra, for someone who knows it.

## Installing

nematocyte needs Python 3.14 or later.

```bash
uv add nematocyte
uv add 'nematocyte[submitit]'  # for the SLURM launcher
uv add 'nematocyte[optuna]'  # for the Optuna sweeper
```

## Developing

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
```
