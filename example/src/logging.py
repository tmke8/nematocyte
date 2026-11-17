from typing import Any, Literal, NotRequired, TypedDict

import wandb
from wandb import sdk
from wandb.sdk import lib

__all__ = ["WandbKwargs", "init_wandb"]

type Run = sdk.wandb_run.Run | lib.disabled.RunDisabled | None


class WandbKwargs(TypedDict):
    """Dictionary of the parameters for `wandb.init()`.

    This allows us to directly configure `wandb.init()` from the config file and the cmd
    line.
    """

    name: NotRequired[str | None]
    dir: str
    id: NotRequired[str | None]
    project: str
    group: NotRequired[str | None]
    entity: str
    tags: NotRequired[list[str] | None]
    job_type: NotRequired[str | None]
    mode: Literal["online", "offline", "disabled"]
    resume: NotRequired[bool | Literal["allow", "never", "must", "auto"] | None]


def init_wandb(wandb_kwargs: WandbKwargs, cfg: dict[str, Any], *, reinit: bool) -> Run:
    """Call `wandb.init()` with the parameters from the config."""
    return wandb.init(config=cfg, reinit=reinit, **wandb_kwargs)
