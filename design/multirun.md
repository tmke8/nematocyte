# Multirun

How `--multirun` turns one command into several jobs, and how the jobs are
run.

## How Hydra does it

Read from the source of Hydra 1.4.0rc1, its submitit launcher and its Optuna
sweeper, and confirmed with small test runs.

### Two roles

- A **sweeper** turns the arguments into jobs. A job is a list of overrides.
- A **launcher** runs a batch of jobs and returns one result per job.

Both are plugins, selected with `hydra/sweeper=NAME` and
`hydra/launcher=NAME`. The sweeper owns the loop and calls the launcher once
per batch.

### The basic sweeper and launcher

- The basic sweeper takes the product of all discrete sweeps (`a,b`,
  `choice(...)`, `range(...)`). `interval(...)` is an error. It also reads
  sweeps from `hydra.sweeper.params`.
- Before launching, it composes the config of every job, so a wrong key is
  found before anything runs.
- The basic launcher runs the jobs one after another in the launching
  process.

### The submitit launcher

- A batch is submitted as one SLURM job array. The launching process waits
  until every job has finished.
- What is sent to a job is the launcher object, pickled, with the function
  and the list of overrides. The job composes its config again on the compute
  node, from the config files as they are at that moment.
- The launcher's settings are the arguments of `sbatch`, passed through
  submitit. Two defaults refer to Hydra's own config:
  `submitit_folder: ${hydra.sweep.dir}/.submitit/%j` and
  `name: ${hydra.job.name}`.

### The Optuna sweeper

- It creates a study, then repeats: ask Optuna for `n_jobs` trials, launch
  them as one batch, wait, report the return values. It stops after
  `n_trials`.
- The search space comes from `params` in the sweeper config and from the
  command line:

  | Sweep | Optuna distribution |
  | --- | --- |
  | `a,b` and `choice(a, b)` | Categorical |
  | `range(1, 4)` | Integer, or float if the step is a float |
  | `interval(1e-5, 1e-3)` | Float; integer if both ends are integers |
  | `tag(log, interval(...))` | The same on a log scale |

  An override that is not a sweep is a fixed value for every trial.
- The sampler is a nested config group, `hydra/sweeper/sampler`, with `tpe`
  as the default and `random`, `cmaes`, `nsgaii`, `nsgaiii`, `grid`, `gp` and
  `qmc` as the other options. Each is a dataclass that mirrors the arguments
  of the Optuna class.
- `direction` is one direction or a list of them. With a list, the function
  returns one value per objective.

### Observed behaviour

| Situation | Hydra |
| --- | --- |
| Job 1 of 3 raises, basic launcher | All three jobs run. The error of job 1 is reported at the end |
| A failed trial, Optuna sweeper | Raised after the batch, if the share of failed trials in it exceeds `max_failure_rate` (default 0). Read from the source, not run |
| `hydra/launcher=NAME` without `--multirun` | The job runs locally. The launcher is ignored without a message |
| `${now:...}` in several jobs | The same or different, depending on the pattern ([interpolation.md](interpolation.md)) |

### Things `hydra-example` does not use

Each job gets an output directory (`multirun/DATE/TIME/N`) with the composed
config and a log file, optionally becomes the working directory, and can read
its own number and id from Hydra's config. `hydra-example` uses none of this
([FEATURES.md](../FEATURES.md)).

## Decided

### In other documents

| Decision | Where |
| --- | --- |
| `--multirun` enables sweeps; a sweep value without it is an error | [cli.md](cli.md) |
| Sweeps are written `a,b`, `choice(...)`, `range(...)`, `interval(...)` and `tag(log, ...)` | [cli.md](cli.md) |
| Launcher and sweeper are set with `nematocyte.launcher` and `nematocyte.sweeper` | [cli.md](cli.md) |
| Their named configs are in `configs/nematocyte/launcher/` and `configs/nematocyte/sweeper/` | [composition.md](composition.md) |
| The options are built in: `basic` and `submitit_slurm`, `basic` and `optuna`. There are no plugins | [unions.md](unions.md) |
| submitit and Optuna are optional extras | [unions.md](unions.md) |
| The return value of the entry point is the objective | [api.md](api.md) |
| `${now:...}` has the same value in every job | [interpolation.md](interpolation.md) |
| Keys and values under `params` are checked when config files are validated, and again at the start of a multirun | [validation.md](validation.md) |

### The same two roles, without a plugin interface

A sweeper produces batches of jobs. A launcher runs a batch and returns, for
each job, the return value or the exception. Since the options are built in,
this interface is internal and can change.

### A job is a composed config, not a list of overrides

The launching process does all the config work for every job:

1. Merge the files and apply the overrides of the job.
2. Resolve the interpolations.
3. Build the config object, to check it. The object is then thrown away.

What is sent to the job is the resolved tree: plain dicts, lists and scalars.
The job builds the config object from it and calls the function.

Consequences:

- A mistake in the config of any job is found before anything runs or is
  submitted. With the basic sweeper all jobs are known at the start. With
  Optuna the values of a trial are only known per batch, but the keys under
  `params` are checked at the start.
- A job does not read the config files. A job that waits in the SLURM queue
  is not affected by edits made in the meantime. In Hydra it is.
- `${now:...}` is the same in every job without further work.
- Every `__post_init__` runs twice, once in the launching process and once
  in the job. That fits the convention that it only checks the config
  ([validation.md](validation.md)).

### No output directories

A job gets no output directory, its working directory is not changed, and no
config or log file is saved for it. The only files nematocyte causes to be
written are those of submitit.

The function is not told its job number. A W&B run can be told apart by its
config.

A directory per multirun, `multirun/DATE/TIME/` as in Hydra, could be added
later. The files of submitit would then move into it.

### The `basic` sweeper

One field, `params`, with the same meaning as on the command line:

```yaml
params:
  seed: 42,43
  model.num_hidden: range(1, 4)
```

The jobs are the product of all discrete sweeps, from `params` and from the
command line. A key given on the command line replaces the same key in
`params`. `interval(...)` is an error that names the `optuna` sweeper.

Hydra's `max_batch_size` is left out.

### The `optuna` sweeper

| Field | Default | Meaning |
| --- | --- | --- |
| `sampler` | `tpe` | A group; see below |
| `direction` | `minimize` | `minimize` or `maximize`, or a list of them |
| `study_name` | `null` | Name of the study |
| `storage` | `null` | URL of a database in which Optuna keeps the study |
| `n_trials` | `20` | Total number of trials |
| `n_jobs` | `2` | Number of trials per batch |
| `max_failure_rate` | `0.0` | Share of failed trials per batch that is tolerated |
| `params` | `null` | The search space |

The loop and the mapping of sweeps to distributions are Hydra's. Hydra's
`custom_search_space`, which names a Python function, is left out.

The sampler is a group field of the `optuna` option, so it follows the
ordinary rules: `nematocyte.sweeper.sampler=random` selects one,
`nematocyte.sweeper.sampler.seed=1` sets a value, and named configs would be
in `configs/nematocyte/sweeper/sampler/`
([composition.md](composition.md)). In `fcn_params.yaml` from the example,
`sampler: {warn_independent_sampling: true}` sets a value of the default
sampler.

The options of `sampler` are `tpe`, the default, and `random`. The samplers
are Optuna's classes, such as `optuna.samplers.TPESampler`. Each option is a
dataclass that copies the arguments of the class's constructor, so that the
sampler's settings are in the schema, and each copy has to be kept in step
with Optuna. The others that Hydra offers can be added as they are needed.

At the end the best parameters and the best value are printed. Hydra also
writes them to a file in the output directory; without output directories,
`storage` is the way to keep the results.

### `direction` can be a list

`direction` is `minimize` or `maximize`, or a list of them, as the example
writes it. With more than one entry the function returns a tuple with one
value per objective, as in Hydra.

### The `basic` launcher

No fields. The jobs run one after another in the launching process, so
global state carries over from one job to the next. This is Hydra's
behaviour, and the reason the example passes `reinit=True` to `wandb.init()`.

### The `submitit_slurm` launcher

A batch is one job array, and the launching process waits for it, as in
Hydra. The Optuna sweeper needs the results, and the basic sweeper reports
them.

Hydra's `submitit_local` launcher is left out.

### The SLURM launcher has all of Hydra's settings

The 28 fields keep Hydra's names, so the launcher files of `hydra-example`
work unchanged. The values are passed on to submitit, which takes them in
three places (submitit 1.5.4):

| Fields | Where submitit takes them |
| --- | --- |
| `submitit_folder`, `max_num_timeout`, `python` | Arguments of the constructor of `SlurmExecutor` |
| `name`, `timeout_min`, `mem_gb`, `nodes`, `cpus_per_task`, `gpus_per_node`, `tasks_per_node` | Names that submitit shares between its executors. For SLURM it translates them to `job_name`, `time`, `mem`, `nodes`, `cpus_per_task`, `gpus_per_node` and `ntasks_per_node` |
| The other 18, such as `partition`, `qos`, `gres`, `constraint`, `setup` and `additional_parameters` | Keyword arguments of `_make_sbatch_string` in `submitit/slurm/slurm.py`, the function that writes the `sbatch` script |

Hydra's dataclass is a hand-written copy of these, and nematocyte's will be
too. `_make_sbatch_string` is private to submitit, so the copy has to be
kept in step by hand.

submitit accepts a few arguments that Hydra's dataclass does not have:
`teardown`, `mail_type`, `mail_user`, `nodelist`, `dependency`, `exclusive`
and `wckey`. They can be given through `additional_parameters`.

### The defaults of `submitit_folder` and `name`

Hydra's defaults refer to its output directory and its job name. In
nematocyte:

- `submitit_folder` is `.submitit/DATE/TIME/%j` under the working directory,
  filled in by nematocyte.
- `name` is the name of the script without `.py`.

### A launcher without `--multirun` is an error

`python main.py nematocyte.launcher=slurm_kyiv` is an error that asks for
`--multirun`. The same goes for a sweeper. Hydra ignores both silently, and
the job that was meant for the cluster runs on the login node.

### A failed job does not stop the others

As in Hydra, the other jobs still run. At the end every failure is listed,
not only the first, and the exit code is not zero. For Optuna,
`max_failure_rate` keeps its meaning.

### Ctrl-C leaves submitted jobs running

Interrupting the launching process does not cancel jobs that are already in
the SLURM queue. This is how to stop waiting for a sweep of the basic
sweeper, once the jobs are submitted. There is no option to submit and exit.

An Optuna search does not survive it: the loop that asks for the next trials
is in the launching process.
