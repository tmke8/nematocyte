"""nematocyte's own settings, which live under the reserved key `nematocyte`."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Discriminator, Field, Tag

from ._schema import pick

__all__ = [
    "BasicLauncher",
    "BasicSweeper",
    "Direction",
    "LogLevel",
    "OptunaSweeper",
    "RandomSampler",
    "Settings",
    "SubmititSlurmLauncher",
    "TpeSampler",
]


@dataclass(kw_only=True)
class BasicLauncher:
    """Runs the jobs one after another, in the process that was started."""


@dataclass(kw_only=True)
class SubmititSlurmLauncher:
    """Submits the jobs to a SLURM cluster as one job array, through submitit."""

    submitit_folder: str | None = field(
        default=None,
        doc="Directory for the files of submitit, where %j stands for the id "
        "of the job. If not set, it is .submitit/%j in the directory of the "
        "multirun.",
    )
    timeout_min: int = field(default=60, doc="Time limit of a job, in minutes.")
    cpus_per_task: int | None = field(default=None, doc="Number of CPUs per task.")
    gpus_per_node: int | None = field(default=None, doc="Number of GPUs per node.")
    tasks_per_node: int | None = field(default=1, doc="Number of tasks per node.")
    mem_gb: float | None = field(
        default=None, doc="Memory to reserve on each node, in GB."
    )
    nodes: int = field(default=1, doc="Number of nodes per job.")
    name: str | None = field(
        default=None,
        doc="Name of the SLURM job. If not set, it is the name of the script.",
    )
    stderr_to_stdout: bool = field(
        default=False, doc="Whether standard error goes to the file of standard output."
    )
    partition: str | None = field(default=None, doc="SLURM partition to submit to.")
    qos: str | None = field(default=None, doc="Quality of service to request.")
    comment: str | None = field(default=None, doc="Comment attached to the job.")
    constraint: str | None = field(
        default=None, doc="Features that the nodes must have."
    )
    exclude: str | None = field(default=None, doc="Nodes that must not be used.")
    gres: str | None = field(
        default=None, doc="Generic resources to request, such as gpu:1."
    )
    cpus_per_gpu: int | None = field(default=None, doc="Number of CPUs per GPU.")
    gpus_per_task: int | None = field(default=None, doc="Number of GPUs per task.")
    mem_per_gpu: str | None = field(default=None, doc="Memory per GPU, such as 16GB.")
    mem_per_cpu: str | None = field(default=None, doc="Memory per CPU, such as 4GB.")
    account: str | None = field(default=None, doc="Account to charge the job to.")
    signal_delay_s: int = field(
        default=120,
        doc="Seconds between the warning signal and the end of the time limit.",
    )
    max_num_timeout: int = field(
        default=0, doc="How often a job is submitted again after it timed out."
    )
    python: str | None = field(
        default=None,
        doc="Python executable for the jobs. If not set, it is the one that "
        "runs the launching process.",
    )
    additional_parameters: dict[str, Any] = field(
        default_factory=dict,
        doc="Further arguments of sbatch that have no field here, by name.",
    )
    array_parallelism: int = field(
        default=256, doc="Maximum number of jobs of the array that run at once."
    )
    setup: list[str] | None = field(
        default=None, doc="Shell commands to run in the batch script before the job."
    )
    srun_args: list[str] | None = field(default=None, doc="Further arguments for srun.")
    use_srun: bool = field(
        default=True,
        doc="Whether the job is started through srun, and not directly in the "
        "batch script.",
    )


@dataclass(kw_only=True)
class BasicSweeper:
    """Runs one job for every combination of the swept values."""

    params: dict[str, Any] | None = field(
        default=None,
        doc="Sweeps by key, written as on the command line: {seed: '42,43'}.",
    )


@dataclass(kw_only=True)
class TpeSampler:
    """Optuna's TPESampler, the tree-structured Parzen estimator.

    The fields are the arguments of `optuna.samplers.TPESampler`. A field
    that is null is left at Optuna's default.
    """

    seed: int | None = field(default=None, doc="Seed of the random number generator.")
    consider_prior: bool | None = field(
        default=None,
        doc="Whether to enhance the stability of the estimator with a prior.",
    )
    prior_weight: float | None = field(default=None, doc="Weight of the prior.")
    consider_magic_clip: bool | None = field(
        default=None, doc="Whether to enable the 'magic clip' heuristic."
    )
    consider_endpoints: bool | None = field(
        default=None,
        doc="Whether to take the ends of the domains into account for the variances.",
    )
    n_startup_trials: int | None = field(
        default=None, doc="Number of trials that are sampled at random at the start."
    )
    n_ei_candidates: int | None = field(
        default=None,
        doc="Number of candidates from which the expected improvement is computed.",
    )
    multivariate: bool | None = field(
        default=None, doc="Whether to model the parameters jointly."
    )
    group: bool | None = field(
        default=None,
        doc="Whether to group the parameters by search space, with multivariate.",
    )
    warn_independent_sampling: bool | None = field(
        default=None,
        doc="Whether to warn when a parameter is sampled independently of the others.",
    )
    constant_liar: bool | None = field(
        default=None,
        doc="Whether running trials are counted as bad, to avoid sampling close to them.",
    )


@dataclass(kw_only=True)
class RandomSampler:
    """Optuna's RandomSampler, which samples every parameter independently."""

    seed: int | None = field(default=None, doc="Seed of the random number generator.")


type Direction = Literal["minimize", "maximize"]


@dataclass(kw_only=True)
class OptunaSweeper:
    """Searches for the best values with Optuna."""

    sampler: Annotated[
        Annotated[TpeSampler, Tag("tpe")] | Annotated[RandomSampler, Tag("random")],
        Discriminator(pick),
    ] = field(default_factory=TpeSampler, doc="The algorithm that proposes the values.")
    direction: Direction | list[Direction] = field(
        default="minimize",
        doc="Whether the return value is minimized or maximized. A list, for "
        "a function that returns several values.",
    )
    study_name: str | None = field(default=None, doc="Name of the study.")
    storage: str | None = field(
        default=None, doc="URL of a database in which Optuna keeps the study."
    )
    n_trials: int = field(default=20, doc="Total number of trials.")
    n_jobs: int = field(default=2, doc="Number of trials that are run as one batch.")
    max_failure_rate: Annotated[float, Field(ge=0.0, le=1.0)] = field(
        default=0.0, doc="Share of failed trials in a batch that is tolerated."
    )
    params: dict[str, Any] | None = field(
        default=None,
        doc="The search space by key, written as on the command line: "
        "{opt.lr: 'interval(1e-5, 1e-3)'}.",
    )


type LogLevel = Literal["debug", "info", "warn"]


@dataclass(kw_only=True)
class Settings:
    """nematocyte's own settings."""

    launcher: Annotated[
        Annotated[BasicLauncher, Tag("basic")]
        | Annotated[SubmititSlurmLauncher, Tag("submitit_slurm")],
        Discriminator(pick),
    ] = field(default_factory=BasicLauncher, doc="How the jobs of a multirun are run.")
    sweeper: Annotated[
        Annotated[BasicSweeper, Tag("basic")] | Annotated[OptunaSweeper, Tag("optuna")],
        Discriminator(pick),
    ] = field(
        default_factory=BasicSweeper,
        doc="How a multirun turns the swept values into jobs.",
    )
    output_dir: Path | None = field(
        default=None,
        doc="Directory of this start of the program, which holds one "
        "directory per job. If not set, it is outputs/DATE/TIME.",
    )
    log: LogLevel = field(
        default="info",
        doc="Level of log messages, for nematocyte.configure_logging() and "
        "nematocyte.log_level().",
    )
