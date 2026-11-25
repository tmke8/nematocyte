"""`--multirun`: sweepers turn the arguments into jobs, launchers run them."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import itertools
from pathlib import Path
import sys
import traceback
from typing import Any

from ruamel.yaml import YAML

from ._build import checked_json
from ._compose import Composer
from ._errors import ConfigError
from ._interpolation import resolve
from ._overrides import (
    AddedConfig,
    ChoiceSweep,
    IntervalSweep,
    Override,
    RangeSweep,
    Sweep,
    format_value,
    is_sweep,
    iter_sweep,
    parse_value,
)
from ._run import ENTRY_ATTRIBUTE, Entry, Job, JobResult, StartDir, run_job
from ._schema import RESERVED_KEY, Kind, class_fields
from ._settings import (
    BasicSweeper,
    Direction,
    OptunaSweeper,
    RandomSampler,
    Settings,
    SubmititSlurmLauncher,
    TpeSampler,
)
from ._tree import Node, switched_off_message
from ._tree import Path as KeyPath

__all__ = ["Multirun", "check_params", "parse_params", "run_multirun"]


def log(message: str) -> None:
    print(f"[nematocyte] {message}", file=sys.stderr, flush=True)


def _key_path(key: Any, where: str) -> KeyPath:
    path = tuple(str(key).split("."))
    if not isinstance(key, str) or not all(part.isidentifier() for part in path):
        raise ConfigError(
            f"{where}: the key {key!r} under 'params' has to be the dotted path "
            "of a config value, such as 'opt.lr'."
        )
    return path


def parse_params(params: dict[Any, Any] | None, where: str) -> dict[KeyPath, Any]:
    """Parse the `params` of a sweeper: override expressions by key."""
    result: dict[KeyPath, Any] = {}
    for key, text in (params or {}).items():
        path = _key_path(key, where)
        if isinstance(text, str):
            try:
                result[path] = parse_value(text)
            except ConfigError as exc:
                raise ConfigError(
                    f"{where}, in the entry {key!r} of 'params':\n{exc}"
                ) from None
        else:
            # A YAML value that is not a string, such as `seed: 1`, is a fixed value.
            result[path] = text
    return result


def schema_has_key(config_cls: type, path: KeyPath) -> bool:
    """Whether `path` names a field for at least one choice of options."""
    info = class_fields(config_cls).get(path[0])
    if info is None:
        return False
    if len(path) == 1:
        return True
    if info.kind is Kind.VALUE:
        return False
    classes = list(info.options.values()) or [info.config_cls]
    return any(cls is not None and schema_has_key(cls, path[1:]) for cls in classes)


def check_params(params: dict[KeyPath, Any], config_cls: type, where: str) -> None:
    """Check that each key under `params` can name a value in the schema.

    Where a path goes through a group, it is enough that one option of the
    group has the field: which option is selected is only known in a run.
    """
    for path in params:
        key = ".".join(path)
        if path[0] == RESERVED_KEY:
            raise ConfigError(
                f"{where}: the key {key!r} under 'params' is one of "
                "nematocyte's own settings, which cannot be swept."
            )
        if not schema_has_key(config_cls, path):
            raise ConfigError(
                f"{where}: the key {key!r} under 'params' does not name a "
                f"value of {config_cls.__name__}, whichever options are selected."
            )


@dataclass
class Multirun:
    """Everything that a sweeper and a launcher need to know about the run."""

    entry: Entry
    wrapper: Callable[..., Any]
    composer: Composer
    config_name: str | None
    added: list[AddedConfig]
    """The primary configs from `+NAME` arguments, in the order given."""
    fixed: list[Override]
    """The overrides from the command line that are not sweeps."""
    sweeps: dict[KeyPath, Sweep]
    """The sweeps from the command line."""
    undeclared: frozenset[KeyPath]
    """The swept keys that are given as `+key=value`: keys in a dict."""
    settings: Settings
    now: datetime
    start: StartDir
    program: str
    """The name of the script, without `.py`."""
    results: list[JobResult] = field(default_factory=list)

    def make_job(self, number: int, assignments: Sequence[tuple[KeyPath, Any]]) -> Job:
        """Compose, resolve and check the config of one job.

        The config object is built to check it, and then thrown away: the job
        gets the JSON text that was validated and builds the object again.
        """
        overrides = self.overrides_with(assignments)
        try:
            root = self.composer.compose(self.config_name, overrides, self.added)
            resolve(root, self.now)
            config = checked_json(root, skip=RESERVED_KEY)
        except ConfigError as exc:
            shown = " ".join(
                [a.text for a in self.added]
                + [o.text for o in overrides if o.key[0] != RESERVED_KEY]
            )
            raise ConfigError(f"In job #{number} ({shown}):\n{exc}") from None
        return Job(
            number,
            [a.text for a in self.added] + [o.text for o in overrides],
            config,
            self.settings.log,
        )

    def overrides_with(
        self, assignments: Sequence[tuple[KeyPath, Any]]
    ) -> list[Override]:
        """The fixed overrides, with `assignments` on top of them."""
        overrides = {override.key: override for override in self.fixed}
        for path, value in assignments:
            plus = "+" if path in self.undeclared else ""
            text = f"{plus}{'.'.join(path)}={format_value(value)}"
            overrides[path] = Override(
                path, value, text, undeclared=path in self.undeclared
            )
        return list(overrides.values())

    def launch(self, jobs: list[Job]) -> list[JobResult]:
        """Run a batch of jobs with the launcher that is selected."""
        launcher = self.settings.launcher
        if isinstance(launcher, SubmititSlurmLauncher):
            results = _launch_slurm(self, launcher, jobs)
        else:
            results = _launch_basic(self, jobs)
        self.results.extend(results)
        return results


def job_label(job: Job) -> str:
    """The overrides of a job, without nematocyte's own settings."""
    shown = [o for o in job.overrides if not o.startswith(f"{RESERVED_KEY}.")]
    return " ".join(shown)


# -- launchers ------------------------------------------------------------


def _launch_basic(run: Multirun, jobs: list[Job]) -> list[JobResult]:
    log(f"Launching {len(jobs)} job{'s' if len(jobs) != 1 else ''} locally")
    results: list[JobResult] = []
    for job in jobs:
        log(f"       #{job.number} : {job_label(job)}")
        try:
            value = run_job(run.entry, job, run.start)
        except Exception as exc:
            results.append(JobResult(job, error=exc))
        else:
            results.append(JobResult(job, value=value))
    return results


def run_remote_job(wrapper: Callable[..., Any], job: Job, start: StartDir) -> Any:
    """What a SLURM job runs. Module-level, so that it can be pickled."""
    entry = vars(wrapper)[ENTRY_ATTRIBUTE]
    assert isinstance(entry, Entry)
    return run_job(entry, job, start)


def slurm_parameters(launcher: SubmititSlurmLauncher, program: str) -> dict[str, Any]:
    """The arguments for `SlurmExecutor.update_parameters`.

    submitit shares a few names between its executors and translates them
    for SLURM; the translation is done here, since the SLURM executor is
    used directly.
    """
    parameters: dict[str, Any] = {
        "job_name": launcher.name if launcher.name is not None else program,
        "time": launcher.timeout_min,
        "nodes": launcher.nodes,
        "cpus_per_task": launcher.cpus_per_task,
        "gpus_per_node": launcher.gpus_per_node,
        "ntasks_per_node": launcher.tasks_per_node,
        "stderr_to_stdout": launcher.stderr_to_stdout,
        "partition": launcher.partition,
        "qos": launcher.qos,
        "comment": launcher.comment,
        "constraint": launcher.constraint,
        "exclude": launcher.exclude,
        "gres": launcher.gres,
        "cpus_per_gpu": launcher.cpus_per_gpu,
        "gpus_per_task": launcher.gpus_per_task,
        "mem_per_gpu": launcher.mem_per_gpu,
        "mem_per_cpu": launcher.mem_per_cpu,
        "account": launcher.account,
        "signal_delay_s": launcher.signal_delay_s,
        "additional_parameters": launcher.additional_parameters,
        "array_parallelism": launcher.array_parallelism,
        "setup": launcher.setup,
        "srun_args": launcher.srun_args,
        "use_srun": launcher.use_srun,
    }
    if launcher.mem_gb is not None:
        gb = launcher.mem_gb
        parameters["mem"] = f"{int(gb)}GB" if gb == int(gb) else f"{int(gb * 1024)}MB"
    return parameters


def _launch_slurm(
    run: Multirun, launcher: SubmititSlurmLauncher, jobs: list[Job]
) -> list[JobResult]:
    try:
        import submitit  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "The launcher 'submitit_slurm' needs the package submitit. "
            "Install it with the extra: nematocyte[submitit]."
        ) from exc

    start = run.start.create()
    folder = launcher.submitit_folder
    if folder is None:
        folder = str(start / ".submitit" / "%j")
    try:
        executor = submitit.SlurmExecutor(
            folder=folder,
            max_num_timeout=launcher.max_num_timeout,
            python=launcher.python,
        )
        executor.update_parameters(**slurm_parameters(launcher, run.program))
    except (RuntimeError, ValueError) as exc:
        raise ConfigError(f"The SLURM launcher cannot be set up: {exc}") from exc

    log(f"Submitting {len(jobs)} job{'s' if len(jobs) != 1 else ''} to SLURM")
    log(f"The files of submitit are in {folder}")
    for job in jobs:
        log(f"       #{job.number} : {job_label(job)}")
    submitted = executor.map_array(
        run_remote_job,
        [run.wrapper] * len(jobs),
        jobs,
        [run.start] * len(jobs),
    )
    results: list[JobResult] = []
    # The launching process waits for the whole array.
    for job, handle in zip(jobs, submitted, strict=True):
        try:
            value = handle.result()
        except Exception as exc:
            results.append(JobResult(job, error=exc))
        else:
            results.append(JobResult(job, value=value))
    return results


# -- sweepers -------------------------------------------------------------


def _merged_params(run: Multirun, where: str) -> dict[KeyPath, Any]:
    """The `params` of the sweeper, with the command line on top."""
    params = parse_params(run.settings.sweeper.params, where)
    check_params(params, run.entry.config_cls, where)
    overridden = {override.key for override in run.fixed}
    merged = {path: value for path, value in params.items() if path not in overridden}
    merged.update(run.sweeps)
    return merged


def _sweep_basic(run: Multirun) -> None:
    params = _merged_params(run, "nematocyte.sweeper")
    fixed: list[tuple[KeyPath, Any]] = []
    keys: list[KeyPath] = []
    choices: list[list[Any]] = []
    for path, value in params.items():
        if isinstance(value, IntervalSweep):
            raise ConfigError(
                f"{'.'.join(path)}: interval() is a continuous range, which the "
                "'basic' sweeper cannot enumerate. Use the 'optuna' sweeper "
                "(nematocyte.sweeper=optuna), or a discrete sweep such as "
                "range() or choice()."
            )
        if is_sweep(value):
            values = list(iter_sweep(value))
            if not values:
                raise ConfigError(f"{'.'.join(path)}: the sweep has no values.")
            keys.append(path)
            choices.append(values)
        else:
            fixed.append((path, value))
    # Every job is composed and checked before the first one runs.
    jobs = [
        run.make_job(number, [*fixed, *zip(keys, combination, strict=True)])
        for number, combination in enumerate(itertools.product(*choices))
    ]
    run.start.create()
    run.launch(jobs)


def _make_sampler(sampler: TpeSampler | RandomSampler) -> Any:
    import optuna  # noqa: PLC0415

    if isinstance(sampler, RandomSampler):
        return optuna.samplers.RandomSampler(seed=sampler.seed)
    arguments = {
        name: getattr(sampler, name)
        for name in class_fields(TpeSampler)
        if getattr(sampler, name) is not None
    }
    return optuna.samplers.TPESampler(**arguments)


def _distribution(key: str, sweep: Sweep) -> Any:
    """The Optuna distribution for a sweep; the mapping is Hydra's."""
    from optuna.distributions import (  # noqa: PLC0415
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    def categorical(values: Sequence[Any]) -> Any:
        for value in values:
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ConfigError(
                    f"{key}: Optuna can only choose between strings, numbers, "
                    f"booleans and null, not {value!r}."
                )
        return CategoricalDistribution(list(values))

    if isinstance(sweep, ChoiceSweep):
        return categorical(sweep.values)
    if isinstance(sweep, RangeSweep):
        values = sweep.values()
        if not values:
            raise ConfigError(f"{key}: the range has no values.")
        low, high = min(values[0], values[-1]), max(values[0], values[-1])
        if all(isinstance(value, int) for value in values):
            return IntDistribution(int(low), int(high), step=abs(int(sweep.step)))
        return FloatDistribution(float(low), float(high), step=abs(float(sweep.step)))
    log_scale = "log" in sweep.tags
    if isinstance(sweep.start, int) and isinstance(sweep.end, int):
        return IntDistribution(sweep.start, sweep.end, log=log_scale)
    return FloatDistribution(float(sweep.start), float(sweep.end), log=log_scale)


def _check_exact(
    run: Multirun, keys: Sequence[KeyPath], fixed: list[tuple[KeyPath, Any]]
) -> None:
    """Check the swept keys against the options that are selected.

    A key under a group that is itself swept cannot be checked this way; for
    those, the check of `check_params` has to do.
    """
    overrides = run.overrides_with(fixed)
    root = run.composer.compose(run.config_name, overrides, run.added)
    swept_groups: list[KeyPath] = []
    for path in sorted(keys, key=len):
        if any(path[: len(group)] == group for group in swept_groups):
            continue
        if path in run.undeclared:
            _check_undeclared(run, path, fixed)
            continue
        node: Node = root
        for index, segment in enumerate(path):
            info = node.fields.get(segment)
            last = index == len(path) - 1
            child = node.children.get(segment)
            if info is not None and not last and node.is_off(info):
                raise ConfigError(
                    f"The swept key {'.'.join(path)!r} does not exist in the "
                    f"config: {switched_off_message(node, info, path[index + 1 :])}"
                )
            if info is None or (
                not last and (info.kind is Kind.VALUE or child is None)
            ):
                raise ConfigError(
                    f"The swept key {'.'.join(path)!r} does not exist in the "
                    f"config: {segment!r} is not a field of {node.describe()}."
                )
            if last:
                if info.kind is not Kind.VALUE:
                    swept_groups.append(path)
                break
            assert child is not None
            node = child


def _check_undeclared(
    run: Multirun, path: KeyPath, fixed: list[tuple[KeyPath, Any]]
) -> None:
    """Check a swept `+key`, a key in a dict, by composing with its first value."""
    sweep = run.sweeps[path]
    if isinstance(sweep, IntervalSweep):
        first: Any = sweep.start
    else:
        first = next(iter_sweep(sweep), None)
    try:
        run.composer.compose(
            run.config_name, run.overrides_with([*fixed, (path, first)]), run.added
        )
    except ConfigError as exc:
        raise ConfigError(
            f"The swept key {'+' + '.'.join(path)!r} cannot be set:\n{exc}"
        ) from None


def _sweep_optuna(run: Multirun, sweeper: OptunaSweeper) -> None:
    try:
        import optuna  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "The sweeper 'optuna' needs the package optuna. "
            "Install it with the extra: nematocyte[optuna]."
        ) from exc

    params = _merged_params(run, "nematocyte.sweeper")
    fixed = [(path, value) for path, value in params.items() if not is_sweep(value)]
    space = {
        ".".join(path): _distribution(".".join(path), value)
        for path, value in params.items()
        if is_sweep(value)
    }
    paths = {".".join(path): path for path in params}
    _check_exact(run, [paths[key] for key in space], fixed)

    directions: list[Direction] = (
        [sweeper.direction]
        if isinstance(sweeper.direction, str)
        else list(sweeper.direction)
    )
    if not directions:
        raise ConfigError("nematocyte.sweeper.direction is an empty list.")
    study = optuna.create_study(
        study_name=sweeper.study_name,
        storage=sweeper.storage,
        sampler=_make_sampler(sweeper.sampler),
        directions=directions,
        load_if_exists=True,
    )
    log(
        f"Optuna study {study.study_name!r}: {sweeper.n_trials} trials, "
        f"{sweeper.n_jobs} per batch, directions: {', '.join(directions)}"
    )
    start = run.start.create()

    number = 0
    remaining = sweeper.n_trials
    while remaining > 0:
        batch = min(remaining, sweeper.n_jobs)
        trials = [study.ask(space) for _ in range(batch)]
        jobs = []
        for offset, trial in enumerate(trials):
            for path, value in fixed:
                trial.set_user_attr(".".join(path), value)
            assignments = [
                *fixed,
                *((paths[key], v) for key, v in trial.params.items()),
            ]
            jobs.append(run.make_job(number + offset, assignments))
        results = run.launch(jobs)
        number += batch
        remaining -= batch

        failures = 0
        for trial, result in zip(trials, results, strict=True):
            try:
                if result.error is not None:
                    raise result.error
                values = _objective(result.value, len(directions))
                study.tell(trial, values)
            except Exception as exc:
                if result.error is None:
                    # The job ran, but what it returned is no objective.
                    index = run.results.index(result)
                    run.results[index] = JobResult(result.job, error=exc)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                failures += 1
        if failures / len(results) > sweeper.max_failure_rate:
            raise _TooManyFailuresError(
                f"{failures} of the {len(results)} trials of a batch failed, "
                f"which is more than max_failure_rate={sweeper.max_failure_rate} "
                "allows. The search is stopped."
            )

    _report_best(study, len(directions), start)


def _report_best(study: Any, directions: int, start: Path) -> None:
    """Print the best parameters, and write them to the multirun's directory."""
    import optuna  # noqa: PLC0415

    completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    if not completed:
        log("No trial was completed, so there is no best value.")
        return
    summary: dict[str, Any]
    if directions == 1:
        best = study.best_trial
        summary = {
            "name": "optuna",
            "best_params": dict(best.params),
            "best_value": best.value,
        }
        log(f"Best parameters: {dict(best.params)}")
        log(f"Best value: {best.value}")
    else:
        solutions = [
            {"params": dict(trial.params), "values": list(trial.values or [])}
            for trial in study.best_trials
        ]
        summary = {"name": "optuna", "solutions": solutions}
        log(f"Number of Pareto solutions: {len(solutions)}")
        for solution in solutions:
            log(f"    Values: {solution['values']}, parameters: {solution['params']}")
    yaml = YAML(typ="safe", pure=True)
    yaml.default_flow_style = False
    with (start / "optimization_results.yaml").open("w", encoding="utf-8") as stream:
        yaml.dump(summary, stream)


class _TooManyFailuresError(Exception):
    pass


def _objective(value: Any, count: int) -> list[float]:
    if count == 1:
        try:
            return [float(value)]
        except TypeError, ValueError:
            raise ValueError(
                f"The function has to return a number as the objective, not {value!r}."
            ) from None
    try:
        values = [float(item) for item in value]
    except TypeError, ValueError:
        raise ValueError(
            f"With {count} directions, the function has to return {count} "
            f"numbers, not {value!r}."
        ) from None
    if len(values) != count:
        raise ValueError(
            f"With {count} directions, the function has to return {count} "
            f"numbers, not {len(values)}."
        )
    return values


def run_multirun(run: Multirun) -> int:
    """Run the sweep. Returns the exit code of the program."""
    sweeper = run.settings.sweeper
    stopped: str | None = None
    try:
        if isinstance(sweeper, OptunaSweeper):
            _sweep_optuna(run, sweeper)
        else:
            assert isinstance(sweeper, BasicSweeper)
            _sweep_basic(run)
    except _TooManyFailuresError as exc:
        stopped = str(exc)

    failed = [result for result in run.results if result.error is not None]
    for result in failed:
        assert result.error is not None
        log(f"Job #{result.job.number} ({job_label(result.job)}) failed:")
        text = "".join(traceback.format_exception(result.error))
        print(textwrap_indent(text), file=sys.stderr, flush=True)
    if failed:
        log(
            f"{len(failed)} of {len(run.results)} jobs failed: "
            + ", ".join(f"#{result.job.number}" for result in failed)
        )
    if stopped is not None:
        log(stopped)
        return 1
    if isinstance(sweeper, OptunaSweeper):
        # Failures within `max_failure_rate` are tolerated.
        return 0
    return 1 if failed else 0


def textwrap_indent(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.rstrip().splitlines())
