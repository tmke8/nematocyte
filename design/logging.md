# Logging

What happens to messages a program writes with Python's `logging` module.

## How Hydra does it

Before each job, Hydra configures `logging` from the config group
`hydra/job_logging`. The config is a `logging.config.dictConfig` dictionary
in YAML. The default:

- sets the level of the root logger to `INFO`;
- adds a console handler that writes to standard output;
- adds a file handler that writes to `NAME.log` in the output directory of
  the job;
- uses the format `[time][logger name][level] - message` for both.

`hydra.verbose=true` switches every logger to `DEBUG`, and
`hydra.verbose=[train]` only the named ones. A second group,
`hydra/hydra_logging`, configures Hydra's own messages.

The same four calls, in plain Python and under Hydra 1.4.0rc1:

| Call | Plain Python | Under Hydra |
| --- | --- | --- |
| `log.debug(...)` | Dropped | Dropped, unless `hydra.verbose=true` |
| `log.info(...)` | Dropped | Shown, and written to the log file |
| `log.warning(...)` | Printed to standard error as the bare message | Shown, and written to the log file |
| `info` from the logger of a library | Dropped | Shown, and written to the log file |

Output of `print` is not captured by either.

## Decided

### The level is a setting of nematocyte, and the setup is opt-in

The level of log messages is not part of the experiment; it is orthogonal
to it, like the launcher or the output directory. So it is one of
nematocyte's own settings ([cli.md](cli.md)):

| Key | Values | Default |
| --- | --- | --- |
| `nematocyte.log` | `debug`, `info`, `warn` | `info` |

`python main.py nematocyte.log=debug` is then an ordinary override, the key
is shown by `--help` with its doc ([helptext.md](helptext.md)), and a
config file can set it under `nematocyte:`. Like the other settings, it is
the same for every job of a multirun and cannot be swept.

nematocyte does not set up `logging` by itself. A program that wants the
setup that Hydra makes calls a function for it, inside the decorated
function:

```python
@nematocyte.main(Config, config_dir="configs")
def main(config: Config) -> float:
    nematocyte.configure_logging()
    ...
```

`nematocyte.configure_logging()` does what this does:

```python
logging.basicConfig(
    level=nematocyte.log_level(),
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(nematocyte.output_dir() / "run.log"),
    ],
    force=True,
)
```

- The root logger gets the level, so the messages of libraries are shown at
  that level too.
- The messages go to standard error, and to `run.log` in the output
  directory of the job ([outputs.md](outputs.md)). The file handler creates
  the output directory. `configure_logging(file=False)` leaves the file out,
  for example on SLURM, where submitit keeps the output of each job.
- **`force=True` is needed for multiruns.** With the basic launcher the jobs
  run one after another in the same process ([multirun.md](multirun.md)).
  Without `force=True` the second call to `basicConfig` does nothing. Tested
  with two jobs: the messages of the second job went into the log file of
  the first, and its level was ignored.

A program that sets up `logging` in its own way reads the level with
`nematocyte.log_level()`. It returns a level of `logging`: `logging.DEBUG`,
`logging.INFO` or `logging.WARNING`, for `debug`, `info` and `warn`.

Both functions work only while the decorated function is running, as
`nematocyte.output_dir()` does; outside a run they are an error. The level
is not part of the config object that the function receives, which holds
only the program's own config. It is passed to each job next to the config,
also to a job on SLURM.

## Not in the first version

- **A logging setup configured through the config**, as with Hydra's
  `hydra/job_logging`: the format, the handlers and the name of the file.
  Perhaps later.
- **`DEBUG` for named loggers only**, as with `hydra.verbose=[train]`.
  `nematocyte.log=debug` also shows the debug messages of every library.
  A program that needs this can call `logging.getLogger(name).setLevel(...)`
  itself.
