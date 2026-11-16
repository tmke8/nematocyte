"""Data loading and preprocessing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, override

from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["Dataset", "DataModule", "CelebADataModule", "ColoredMNISTDataModule"]


@dataclass(kw_only=True)
class DataModule(ABC):
    """Base class for data modules."""

    root: Path = field(
        default=Path("/srv/galene0/shared/data"),
        doc="Directory in which the datasets are stored.",
    )

    @property
    @abstractmethod
    def train_data(self) -> Dataset[tuple[Tensor, Tensor]]: ...

    @property
    @abstractmethod
    def in_dim(self) -> int: ...

    @property
    @abstractmethod
    def out_dim(self) -> int: ...

    @abstractmethod
    def prepare(self, seed: int) -> None: ...


type CelebAttr = Literal["Blond_Hair", "Male", "Smiling"]


@dataclass(kw_only=True)
class CelebADataModule(DataModule):
    """Data-module for the CelebA dataset."""

    default_res: int = field(default=224, doc="Resolution of the images.")
    superclass: CelebAttr = field(
        default="Blond_Hair", doc="Attribute that defines the superclass."
    )
    subclass: CelebAttr = field(
        default="Male", doc="Attribute that defines the subclass."
    )
    download: bool = field(
        default=False, doc="Whether to download the dataset if necessary."
    )
    split_seed: int | None = field(default=None, doc="Seed for splitting the dataset.")

    @property
    @override
    def train_data(self) -> Dataset[tuple[Tensor, Tensor]]:
        # Load data from disk.
        # Preprocess data.
        raise NotImplementedError()

    @property
    @override
    def in_dim(self) -> int:
        return self.default_res

    @property
    @override
    def out_dim(self) -> int:
        return 3

    @override
    def prepare(self, seed: int) -> None:
        # Download data if necessary.
        pass


@dataclass(kw_only=True)
class ColoredMNISTDataModule(DataModule):
    """Data-module for the ColoredMNIST dataset."""

    default_res: int = field(default=28, doc="Resolution of the images.")

    # Colorization settings
    label_map: dict[int, int] | None = field(
        default=None, doc="Mapping from the original labels to new labels."
    )
    colors: list[int] | None = field(default=None, doc="Indices of the colors to use.")
    num_colors: int = field(default=10, doc="Number of colors.")
    val_prop: float = field(
        default=0.2, doc="Proportion of the data that is used for validation."
    )

    @property
    @override
    def train_data(self) -> Dataset[tuple[Tensor, Tensor]]:
        # Load data from disk.
        # Preprocess data.
        raise NotImplementedError()

    @property
    @override
    def in_dim(self) -> int:
        return self.default_res

    @property
    @override
    def out_dim(self) -> int:
        return 10

    @override
    def prepare(self, seed: int) -> None:
        # Download data if necessary.
        pass
