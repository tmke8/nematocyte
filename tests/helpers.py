"""Config classes and config files that the tests share."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from nematocyte import option
from nematocyte._build import build
from nematocyte._compose import Composer
from nematocyte._interpolation import resolve
from nematocyte._overrides import split_arguments
from nematocyte._settings import OptunaSweeper, Settings, SubmititSlurmLauncher
from nematocyte._tree import Node

type Activation = Literal["relu", "gelu", "selu", "silu"]


@dataclass(kw_only=True)
class BatchNorm:
    """Batch normalisation."""

    momentum: float = 0.1


@dataclass(kw_only=True)
class LayerNorm:
    eps: float = 1e-5


type Norm = (
    Annotated[BatchNorm, option("batchnorm")]
    | Annotated[LayerNorm, option("layernorm")]
)


@dataclass(kw_only=True)
class Fcn:
    """Fully-connected network.

    This paragraph is not part of the overview.
    """

    num_hidden: int = field(default=1, doc="Number of hidden layers.")
    hidden_dim: int | None = None
    activation: Activation = field(default="gelu", doc="Activation function.")
    dropout_prob: float = 0.0


@dataclass(kw_only=True)
class Cnn:
    kernel_size: int = field(default=5, doc="Kernel size of the convolutions.")
    stride: int = 2
    activation: Activation = "relu"
    norm: Norm = field(default_factory=BatchNorm, doc="The normalisation layer.")


type Model = Annotated[Fcn, option("fcn")] | Annotated[Cnn, option("cnn")]


@dataclass(kw_only=True)
class DataModule(ABC):
    """Base class for data modules."""

    root: Path = field(default=Path("/data"), doc="Where the datasets are stored.")
    default_res: int = field(default=1, doc="Resolution of the images.")

    @property
    @abstractmethod
    def in_dim(self) -> int: ...


@dataclass(kw_only=True)
class Celeba(DataModule):
    """The CelebA dataset."""

    default_res: int = 224
    superclass: Literal["Blond_Hair", "Male", "Smiling"] = "Blond_Hair"
    download: bool = False

    @property
    def in_dim(self) -> int:
        return self.default_res


@dataclass(kw_only=True)
class Cmnist(DataModule):
    """Coloured MNIST."""

    default_res: int = 28
    label_map: dict[int, int] | None = None
    colors: list[int] | None = None
    val_prop: float = 0.2

    @property
    def in_dim(self) -> int:
        return self.default_res


@dataclass
class Opt:
    """Settings for the optimiser."""

    lr: float = field(default=5e-5, doc="Learning rate.")
    weight_decay: float = 0.0
    optimizer_kwargs: dict[str, Any] | None = None


@dataclass
class Wandb:
    name: str | None = None
    group: str | None = None
    tags: list[str] | None = None


@dataclass(kw_only=True)
class Config:
    """Train a model on a dataset."""

    dm: Annotated[Celeba, option("celeba")] | Annotated[Cmnist, option("cmnist")] = (
        field(default_factory=Cmnist, doc="The dataset to train on.")
    )
    model: Model = field(default_factory=Fcn, doc="The model architecture.")
    opt: Opt = field(default_factory=Opt)
    wandb: Wandb = field(default_factory=Wandb)
    seed: int = field(default=42, doc="Random seed.")
    gpu: int = field(default=-1, doc="Index of the GPU to use.")
    flag: bool = False

    def __post_init__(self) -> None:
        if self.gpu < -1:
            raise ValueError(f"Invalid GPU index {self.gpu}.")


CONFIG_FILES = {
    "good_run.yaml": """\
_doc_: |
  A good run, on small images.

  More text that does not belong in an overview.
_defaults_:
  model: single_linear_layer
  dm: celeba_small

seed: 1

model:
  activation: selu

dm:
  download: true

opt:
  lr: 0.001
  weight_decay: 0.001
""",
    "experiments/cmnist_cnn.yaml": """\
_defaults_:
  dm: cmnist
  model: cnn
  opt: weight_decay_adamw

dm:
  val_prop: 0.3

model:
  kernel_size: 3
""",
    "experiments/good_run_blond.yaml": """\
_defaults_:
  _self_: good_run
  dm: celeba_blond

seed: 2

# The selection above drops what `good_run` set under `dm`.
dm:
  download: true
""",
    "presets/debug.yaml": """\
# Applied on top of any primary config with +presets/debug.
_doc_: Settings for debugging.
seed: 0

opt:
  lr: 0.1

nematocyte:
  log: debug
""",
    "presets/debug_cnn.yaml": """\
_defaults_:
  _self_: presets/debug
  model: cnn

seed: 5

model:
  stride: 1
""",
    "dm/celeba_smiling.yaml": """\
_defaults_:
  _self_: celeba

superclass: Smiling
""",
    "dm/celeba_small.yaml": """\
_defaults_:
  _self_: celeba_smiling

default_res: 64
""",
    "dm/celeba_blond.yaml": """\
_defaults_:
  _self_: celeba

superclass: Blond_Hair
""",
    "model/single_linear_layer.yaml": """\
_doc_: An fcn without hidden layers.
_defaults_:
  _self_: fcn

num_hidden: 0
activation: relu
""",
    "model/my_cnn.yaml": """\
_defaults_:
  _self_: cnn
  norm: layernorm

stride: 1
""",
    "model/norm/my_batchnorm.yaml": """\
_defaults_:
  _self_: batchnorm

momentum: 0.5
""",
    "opt/weight_decay_adamw.yaml": """\
_doc_: AdamW with weight decay.
lr: 1.0e-3
weight_decay: 1.e-2
""",
    "opt/strong_weight_decay.yaml": """\
_defaults_:
  _self_: weight_decay_adamw

weight_decay: 0.1
""",
    "nematocyte/launcher/slurm_test.yaml": """\
_doc_: The test partition, with one GPU.
_defaults_:
  _self_: submitit_slurm

partition: test
cpus_per_task: 4
mem_gb: 8
gres: "gpu:1"
setup:
  - "source .venv/bin/activate"
""",
    "nematocyte/sweeper/search.yaml": """\
_defaults_:
  _self_: optuna

sampler:
  seed: 0

n_trials: 6
n_jobs: 3
direction:
  - maximize

params:
  opt.lr: tag(log, interval(1.e-5, 1.e-3))
  model.num_hidden: range(1, 4)
  model.dropout_prob: choice(0.0, 0.1)
""",
}


def empty_config_dir() -> Path:
    """An empty config directory, in the working directory of the test."""
    directory = Path.cwd() / "no_configs"
    directory.mkdir(exist_ok=True)
    return directory


def write_files(directory: Path, files: dict[str, str]) -> Path:
    """Write `files`, which maps relative paths to their content."""
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return directory


def compose_tree(
    config_cls: type,
    config_dir: Path | None = None,
    overrides: Sequence[str] = (),
    config_name: str | None = None,
) -> Node:
    """The composed tree, before interpolations are resolved."""
    composer = Composer(config_cls, config_dir, Settings)
    added, parsed = split_arguments(overrides)
    return composer.compose(config_name, parsed, added)


def compose_settings(
    config_dir: Path | None = None,
    overrides: Sequence[str] = (),
    config_name: str | None = None,
) -> Settings:
    """nematocyte's own settings for the test config."""
    root = compose_tree(Config, config_dir, overrides, config_name)
    resolve(root, datetime(2026, 11, 16, 16, 49, 10))
    settings = build(root.children["nematocyte"])
    assert isinstance(settings, Settings)
    return settings


def slurm_launcher(settings: Settings) -> SubmititSlurmLauncher:
    assert isinstance(settings.launcher, SubmititSlurmLauncher)
    return settings.launcher


def optuna_sweeper(settings: Settings) -> OptunaSweeper:
    assert isinstance(settings.sweeper, OptunaSweeper)
    return settings.sweeper


def fcn(config: Config) -> Fcn:
    assert isinstance(config.model, Fcn)
    return config.model
