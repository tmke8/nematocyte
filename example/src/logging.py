from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import wandb
from wandb import sdk
from wandb.sdk import lib

__all__ = ["WandbCfg"]

type Run = sdk.wandb_run.Run | lib.disabled.RunDisabled | None


@dataclass
class WandbCfg:
    """Dataclass wrapper around the parameters for `wandb.init()`.

    This allows us to directly configure `wandb.init()` from the config file and the cmd
    line.
    """

    name: str | None = field(default=None, doc="Name of the run.")
    dir: str = field(
        default="./local_logging", doc="Directory in which W&B stores its files."
    )
    id: str | None = field(default=None, doc="ID of the run.")
    project: str = field(default="nematocyte-example", doc="W&B project to log to.")
    group: str | None = field(default=None, doc="Group to put the run in.")
    entity: str = field(
        default="predictive-analytics-lab", doc="W&B user or team to log to."
    )
    tags: list[str] | None = field(default=None, doc="Tags for the run.")
    job_type: str | None = field(default=None, doc="Type of the run.")
    mode: Literal["online", "offline", "disabled"] = field(
        default="disabled", doc="Whether to log online, offline or not at all."
    )
    resume: str | None = field(
        default=None, doc="Whether and how to resume a previous run."
    )

    def init(self, cfg: dict[str, Any], *, reinit: bool) -> Run:
        """Call `wandb.init()` with the parameters from the config."""
        kwargs = asdict(self)
        return wandb.init(config=cfg, reinit=reinit, **kwargs)
