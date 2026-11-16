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

### nematocyte does not set up logging

A program that uses `logging` configures it itself. Of the things Hydra sets
up, only the level is worth changing from the command line, and that needs
nothing from nematocyte: it can be a field of the program's own config.

```python
@dataclass(kw_only=True)
class Config:
    log_level: Literal["DEBUG", "INFO", "WARNING"] = field(
        default="INFO", doc="Level for log messages."
    )
```

```python
logging.basicConfig(
    level=config.log_level,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(nematocyte.output_dir() / "run.log"),
    ],
    force=True,
)
```

`python main.py log_level=DEBUG` is then an ordinary override, and the field
is shown by `--help` with its doc ([helptext.md](helptext.md)).

The format and the log file are chosen once for a program, so they are
written in Python.

Two things to know about the snippet:

- **`force=True` is needed for multiruns.** With the basic launcher the jobs
  run one after another in the same process ([multirun.md](multirun.md)).
  Without `force=True` the second call to `basicConfig` does nothing. Tested
  with two jobs: the messages of the second job went into the log file of
  the first, and its level was ignored.
- **The file handler creates the output directory.** A run that calls
  `nematocyte.output_dir()` always leaves a directory behind
  ([outputs.md](outputs.md)). On SLURM the handler can be left out, since
  submitit stores the standard output of the job.

## Not in the first version

- **A logging setup by nematocyte**, configured through the config as with
  Hydra's `hydra/job_logging`. Perhaps later.
- **A built-in `nematocyte.log_level`**, so that a program does not have to
  declare the field itself.
- **`DEBUG` for named loggers only**, as with `hydra.verbose=[train]`.
  Setting the root level to `DEBUG` also shows the debug messages of every
  library. A program that needs this can make its field a list of logger
  names.
