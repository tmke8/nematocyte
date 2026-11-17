"""Main run function and config class."""

from dataclasses import dataclass, field
import logging
from typing import Annotated, Any

from nematocyte import option
import torch

from src.datasets import CelebADataModule, ColoredMNISTDataModule
from src.logging import WandbCfg
from src.models import FcnFactory, SimpleCNNFactory
from src.optimisation import OptimisationCfg

__all__ = ["Config"]

# Each module gets its own logger, named after the module (here `src.run`). The
# name appears in every message, so you can tell where a message comes from.
log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Config:
    """Main configuration class for the code base."""

    # The first two fields are config groups: each can hold one of several config
    # classes, which need different keys and values to be configured.
    # The options of a group are listed right here in the type of the field, each
    # with the name under which it is selected (`dm=celeba`, `model=cnn`).
    # The default says which option is used if nothing is selected.
    dm: (
        Annotated[CelebADataModule, option("celeba")]
        | Annotated[ColoredMNISTDataModule, option("cmnist")]
    ) = field(default_factory=ColoredMNISTDataModule, doc="The dataset to train on.")
    model: (
        Annotated[FcnFactory, option("fcn")]
        | Annotated[SimpleCNNFactory, option("cnn")]
    ) = field(default_factory=FcnFactory, doc="The model architecture.")

    # These are normal subconfigs. Note that in dataclasses, the default must not be
    # mutable, so we use `default_factory`.
    opt: OptimisationCfg = field(default_factory=OptimisationCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)

    # These are normal fields. The `doc` of a field is shown by `--help`.
    seed: int = field(default=42, doc="Random seed.")
    gpu: int = field(
        default=-1, doc="Index of the GPU to use. Set to -1 to use the CPU."
    )

    def __post_init__(self) -> None:
        """Post-initialization checks.

        Only check things here that follow from the config values alone. This method
        also runs when the config files are validated in the tests, on a machine that
        may look nothing like the one the experiment runs on.
        """
        if self.gpu < -1:
            raise ValueError(
                f"Invalid GPU index {self.gpu}. "
                "Must be -1 (to use the CPU) or the index of a GPU."
            )

    def run(self, config_for_logging: dict[str, Any]) -> float:
        """Run the experiment."""
        # Checks that depend on the machine we are running on belong here, and not
        # in `__post_init__`.
        if self.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid GPU index {self.gpu}. "
                f"This machine has {torch.cuda.device_count()} GPU(s)."
            )

        device = f"GPU {gpu}" if (gpu := self.gpu) >= 0 else "CPU"
        log.info("Starting a run with seed %d on %s.", self.seed, device)
        # Debug messages are only shown with `nematocyte.log=debug`.
        log.debug("The config of this run: %s", config_for_logging)
        # Set the seed for reproducibility.
        torch.manual_seed(self.seed)

        # Initialize the logger.
        wandb_run = self.wandb.init(config_for_logging, reinit=True)
        if wandb_run is not None:
            wandb_run.log({"accuracy": 0.5})

        # Prepare the data module.
        self.dm.prepare(seed=self.seed)

        # Build the model.
        model = self.model.build(in_dim=self.dm.in_dim, out_dim=self.dm.out_dim)

        log.info("Model architecture:\n%s", model)

        # At the end, we return a value representing how well the model performed on the
        # validation set. That can be the validation loss or validation accuracy, for
        # example. This value is used for hyperparameter optimization.
        # If you don't intend to perform hyperparameter optimization, you don't have to
        # return anything.
        return 0.5
