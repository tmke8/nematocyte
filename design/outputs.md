# Output directories

Where a run can put its files, such as model checkpoints.

## How Hydra does it

Observed with Hydra 1.4.0rc1.

### Layout

```text
outputs/2026-11-16/16-49-10/          a single run
    .hydra/config.yaml
    .hydra/hydra.yaml
    .hydra/overrides.yaml
    app.log

multirun/2026-11-16/16-49-10/         a multirun
    multirun.yaml
    .submitit/
    0/                                one directory per job
        .hydra/...
        app.log
    1/
```

The paths are settings: `hydra.run.dir` for a single run, and
`hydra.sweep.dir` with `hydra.sweep.subdir` for a multirun. The subdirectory
of a job is its number.

### What is in a directory

| File | Content |
| --- | --- |
| `.hydra/config.yaml` | The config of the job, with interpolations as written |
| `.hydra/overrides.yaml` | The overrides of the job |
| `.hydra/hydra.yaml` | Hydra's own config |
| `NAME.log` | What the program logs through `logging`. Empty if it logs nothing |
| `multirun.yaml` | In the directory of a multirun: the config before the sweep is applied |
| `optimization_results.yaml` | In the directory of a multirun: the best parameters found by Optuna |

### Observed behaviour

| Situation | Hydra |
| --- | --- |
| Finding the directory in Python | `HydraConfig.get().runtime.output_dir` |
| Finding it in the config | `${hydra:runtime.output_dir}/checkpoints` |
| The working directory | Not changed, unless `hydra.job.chdir=true` |
| A run that writes nothing | The directory is created anyway, with the files above |
| `--help` and `--cfg` | No directory is created |
| Two single runs started in the same second | They share one directory, and the second overwrites the files of the first |
| A single run and a multirun started in the same second | They get the same date and time. Only the two roots keep them apart |
| Starting again from the saved `config.yaml` (`--config-path DIR/.hydra --config-name config`) | It loads, but as plain dicts: the schema is not applied (`seed=abc` is accepted), the file does not say which option of a group was chosen, and `${now:...}` gets a new value |

Nothing in Hydra's source says why there are two roots. The last row is the
one effect that was found.

## Decided

### A run has an output directory, under one root

As in Hydra, a run has a directory for its files. Unlike in Hydra, there is
one root, `outputs/`, for single runs and multiruns.

### Every run is a numbered directory

```text
outputs/2026-11-16/
    16-49-10/                         a single run
        0/
            .nematocyte/
            checkpoints/              whatever the program writes
    17-02-33/                         a multirun
        .submitit/
        0/                            one directory per job
            .nematocyte/
            checkpoints/
        1/
```

`outputs/DATE/TIME` belongs to one start of the program, and holds one
numbered directory per job. A single run is a start with one job, `0`.

Every run is then at the same depth, and one glob finds the checkpoints of
all of them: `outputs/*/*/*/checkpoints/`.

### The directory is created when it is first asked for

A run that never asks for its directory leaves nothing behind. In Hydra
every run creates a directory, so `outputs/` fills up with directories that
hold a copy of the config and an empty log.

For a multirun, the directory of the program start is created at the start,
because submitit writes its files there. The directories of the jobs are
created when they are first asked for.

### One setting

`nematocyte.output_dir` is the directory of the program start. If it is not
set, nematocyte fills in `outputs/DATE/TIME`. It can be set like any other
value, with interpolations:

```bash
python main.py 'nematocyte.output_dir=outputs/${wandb.group}/${now:%H-%M-%S}'
```

The directory of a job is always its number. Hydra's `hydra.sweep.subdir`
has no counterpart.

### Finding the directory

```python
checkpoint_dir = nematocyte.output_dir() / "checkpoints"
```

`nematocyte.output_dir()` returns the directory of the current job as a
`Path`, and creates it if it does not exist yet. Outside a run, such as
after `nematocyte.compose(...)`, it is an error.

The working directory is not changed, which is Hydra's default too. Relative
paths in the config, such as `dm.root`, keep their meaning. Hydra's setting
that changes it, `hydra.job.chdir`, is not implemented.

### What nematocyte writes

When the directory of a job is created, two files are written into
`.nematocyte/`:

| File | Content |
| --- | --- |
| `config.json` | The resolved config of the job, as `nematocyte.run_config()` returns it, with `_option_` for the groups ([api.md](api.md#the-config-of-a-run-as-plain-dicts)) |
| `overrides.json` | The arguments of the job: the added configs (`+NAME`) in their order, then the overrides, including its values of the sweep |

The files are JSON, not YAML as in Hydra. The saved config is a record of
what ran, not a config file: it cannot be passed back as a primary config,
since the choices are in `_option_` keys and not in `defaults`
([unions.md](unions.md)). `nematocyte.from_run_config(Config, data)` builds
the object that the function of the job received from it, with every choice
([api.md](api.md#reading-the-config-of-a-run-back)). If a flag to run a job
again is ever added, it can read this file, which holds every value and
every choice.

A log file, `run.log`, is only written if the program calls
`nematocyte.configure_logging()` ([logging.md](logging.md)). On SLURM, the
standard output of a job is in the files of submitit, under
`.submitit/JOBID/`.

### A directory is never shared

If `outputs/DATE/TIME` already exists when a program start wants to create
it, a number is added: `16-49-10-2`. With one root this matters more than
in Hydra, since a single run and a multirun can now meet as well.

### What changes for multiruns

- The default of `submitit_folder` is `.submitit/%j` in the directory of the
  multirun, as in Hydra ([multirun.md](multirun.md)).
- The Optuna sweeper writes the best parameters to
  `optimization_results.yaml` in the directory of the multirun, as in Hydra.
- The number of a job is the name of its directory.
- On SLURM the directory has to be on a file system that the compute nodes
  share with the machine that submits, as in Hydra.

### Details

- `nematocyte.output_dir()` returns an absolute path.
- The number that keeps a directory from being shared is also added to a
  directory that was set with `nematocyte.output_dir`.
- `config.json` is written from the resolved tree, which is what `to_dict`
  gives, plus the values of `InitVar` fields; `run_config()` returns the
  same. A value that JSON does not have, such as a date, is written as its
  text, and so is a dict key that is not a string.

## Not in the first version

- **Referring to the directory in the config.** Hydra allows
  `${hydra:runtime.output_dir}/checkpoints` as a config value. In nematocyte
  the path is built in Python.
- **Running a job again from its saved config.** Hydra's saved config cannot
  be used this way either, for a program that expects dataclass instances;
  see the table above.
