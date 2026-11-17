# Guesses

What the implementation decided on its own, because the design documents
did not cover it or were ambiguous. None of this has been reviewed. A guess
that is confirmed should move into the "Decided" section of the document it
belongs to; one that is turned down has to be changed in the code.

The guesses that are most likely to matter are marked **(check)**.

## Schema ([unions.md](unions.md), [deserializer.md](deserializer.md))

- **`defaults` is reserved, like `_self_`.** No config dataclass can have a
  field called `defaults`, because that key means something else at the top
  of every config file.
- **A union that mixes a dataclass with something else is an error**, such
  as `Sub | int`. A union with a dataclass in it is a group, and all of its
  members have to be named dataclasses.
- **A single `Annotated[X, option("x")]` is a group with one option.**
- **A class can be in a group only once**, and an option name only once.
- **Option names** start with a letter or an underscore, followed by
  letters, digits, `_` or `-`, so that they can be typed unquoted.
- **The default of a plain dataclass field has to be an instance of exactly
  the declared class.** An instance of a subclass is an error, since the
  node is built as the declared class. The default of a group field has to
  be an instance of exactly one of the option classes.
- **(check) Default values are validated by pydantic as well.** Every value
  goes through one path: the resolved tree in plain form, then pydantic,
  then the constructor. A value that nothing set is the default, converted
  to plain form (`Path` to `str` and so on) and validated like any other.
  This is what lets a job build its config from the resolved tree alone
  ([multirun.md](multirun.md)), and it makes a single run behave exactly
  like a job. The cost: a sloppy default such as `size: int = None` is an
  error, which pydantic itself would let through. OmegaConf rejects it too.
- **An `InitVar` is not kept on an instance**, so under a default instance
  its value falls back to the default in the class.
- **`init=False` fields are not config keys.** They cannot be set, and
  `to_dict` leaves them out, since a config file could not hold them.
- **A type that pydantic cannot handle** (such as `torch.device`) is
  accepted for a default value, by an `isinstance` check. It cannot be set
  from a config file, since registered converters are not in the first
  version.
- **`seed: true` is still accepted as `1`.** The one-line check that
  deserializer.md mentions was not added.

## Composition ([composition.md](composition.md))

- **`config_dir` is optional**, in `nematocyte.main` and in
  `nematocyte.compose`. Without it there are no config files. For
  `compose`, a relative path is relative to the working directory.
- **A config directory that does not exist is an error**, straight away.
- **Files end in `.yaml` or `.yml`.** `--config-name` accepts a name with
  or without the suffix.
- **`configs/model.yaml` is a primary config called `model`**, even though
  `model` is a field. Only a directory named after a field is special.
- **A key of a subconfig or group with nothing under it** (`model:` and
  then only comments, which YAML reads as null) is ignored.
- **A selection written among the values** (`model: cnn` outside
  `defaults`) is an error that points to `defaults`.
- **`defaults` as a list**, as in Hydra, is an error that shows the mapping
  form. An entry with `null` (Hydra's `- opt: null`) is an error that says
  it is not needed.
- **`nematocyte=NAME` selects a preset for all of nematocyte's settings**,
  from a file directly in `configs/nematocyte/`. This falls out of the
  general rule and was not prevented.

## Command line ([cli.md](cli.md))

- **The same key twice is an error** (`seed=1 seed=2`). The order of the
  arguments has no meaning, so neither could win.
- **A quoted key in a dict is rejected**, as in Hydra (`{'a b':1}`).
- **Function arguments can be named**, as in Hydra
  (`range(start=1, stop=4)`).
- **`interval(1, 12)` keeps its integers**, so that Optuna gets an integer
  distribution, as multirun.md says.
- **`--help` and `--cfg` with `--multirun` leave the swept values out**,
  since there is one config per job. Without `--multirun`, a sweep is the
  error that it always is.
- **`--cfg` always prints the docs as comments.** helptext.md says they
  "could be optional"; there is no switch. A group is shown with its
  options and the selected one. A value that is required and missing is
  shown as `???`, in `--help` too.
- **(check) `--help` also shows nematocyte's own settings and the flags**,
  after the program's keys. The sample in the example's README did not have
  them. They are a block of their own, so that their keys do not widen the
  left column. The left column is at most 40 characters wide, and the text
  is wrapped at the width of the terminal, between 60 and 100 columns (88
  when there is no terminal).
- **Exit codes:** 1 for an error in the configuration, 2 for a wrong flag
  (from `argparse`). A config error is printed without a traceback; an
  exception of the program itself is not caught.
- **(check) A launcher or sweeper without `--multirun` is an error whenever
  something is selected or set under `nematocyte.launcher` or
  `nematocyte.sweeper`**, by an override or by a config file. That includes
  `nematocyte.launcher=basic`.
- **The decorated function takes an optional list of arguments**
  (`main(["seed=1"])`) in place of `sys.argv`, for tests. It returns
  `None`.

## Interpolation ([interpolation.md](interpolation.md))

- **A reference can go into a dict value** (`${opt.optimizer_kwargs.eps}`),
  but not into a list.
- **Inside a longer string a value is formatted with `str()`**, so `true`
  becomes `True` and null becomes `None`.
- **There is no way to write a literal `${`.**
- **The argument of `now` is everything after the first colon**, as it is.

## Python API ([api.md](api.md), [validation.md](validation.md))

- **Names that the documents do not give:** `nematocyte.ConfigError` for
  every error in a configuration, `nematocyte.SchemaError` (a subclass) for
  a dataclass that is declared in an unsupported way,
  `nematocyte.ValidationResult` with `path`, `error` and `ok`,
  `nematocyte.EntryPoint` for the type of the decorated function, and
  `nematocyte.Option` for what `option()` returns.
- **`validate_configs` and `validate_config` take `config_name`** for the
  primary config that supplies the rest of the tree, and `construct`.
- **A named config for a nested field of a group** is checked with the
  first option that has the field, if the default option does not.
- **`to_dict`**: a value of an unknown type is left as it is; a set is
  sorted if it can be; a dataclass inside a list or dict becomes a dict
  without `_option_`; an instance of a subclass of an option gets the name
  of that option.
- **An invalid value names the class and type of the field** in its error,
  next to the doc: `Field: Opt.lr (float). Learning rate.`

## Output directories ([outputs.md](outputs.md))

- **`nematocyte.output_dir()` returns an absolute path.**
- **The number that keeps a directory from being shared** is also added to
  a directory that was set with `nematocyte.output_dir`.
- **`config.json` is written from the resolved tree**, which is what
  `to_dict` gives, plus the values of `InitVar` fields. A value that JSON
  does not have, such as a date, is written as its text, and so is a dict
  key that is not a string.

## Multirun ([multirun.md](multirun.md))

- **(check) The messages of nematocyte go to standard error**, with the
  prefix `[nematocyte]`. The example's README shows them among the output
  of the program.
- **Order of the jobs:** the product of the sweeps under `params`, then
  those on the command line, with the last one varying fastest.
- **A value under `params` that is not a string** (`seed: 1`) is a fixed
  value for every job.
- **nematocyte's own settings cannot be swept**
  (`nematocyte.launcher=a,b`).
- **Exit code of a multirun:** with the basic sweeper, 1 if any job failed;
  the traceback of each is printed at the end. With Optuna, 0 while the
  failures stay within `max_failure_rate`, and 1 when a batch exceeds it,
  after which the search stops.
- **Optuna:** a return value that is not a number makes the trial a failed
  one. A trial whose config is not valid stops the whole search, since it
  is a mistake in the config and not a failed job. If no trial completes,
  no `optimization_results.yaml` is written.
- **The exact check of the swept keys at the start** leaves out keys under
  a group that is itself swept (`model=choice(fcn,cnn)` with
  `model.num_hidden=...`); those get the check that validation makes.
- **A `range()` of floats** becomes a float distribution with a step.
  Hydra falls back to a categorical one when the floats do not round
  cleanly; this was not copied.
- **The sampler classes** copy the arguments of Optuna 5.1. Every field
  defaults to null, which means "Optuna's default", so that the copy also
  works with Optuna 4. The arguments that take a Python function (`gamma`,
  `weights`, `constraints_func`) are left out.
- **submitit:** `SlurmExecutor` is used directly, and the names that
  submitit shares between executors are translated by hand. `mem_gb` is a
  `float`. `submitit_folder` and `name` default to null and are filled in
  at launch.
- **What is pickled for a SLURM job** is the decorated function, the
  resolved tree and the directory of the multirun. The job needs the
  program's modules to be importable, as with Hydra.

## Project

- **The `test` dependency group includes submitit and Optuna**, since the
  tests of the launchers and sweepers need them. A new group `typecheck`
  has pyright and mypy.
- **The example** now depends on `nematocyte[optuna,submitit]`, ignores
  `outputs/`, and has a `uv.lock`. Its README no longer says that nothing
  runs, and its `--help` sample is the real output.
- **Python 3.15** generates dataclass docstrings lazily, through an object
  that is not a string. The check for generated docstrings in helptext.md
  handles both.
