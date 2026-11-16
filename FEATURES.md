# Features to support

The Hydra features that `hydra-example` relies on, which is the scope for
nematocyte. File references are relative to that project.

## Schema

- The config is a tree of stdlib dataclasses (`src/run.py`, `src/datasets.py`,
  `src/models.py`, `src/optimisation.py`, `src/logging.py`).
  - `@dataclass(kw_only=True)`, inheritance, and abstract base classes with
    abstract methods and properties.
  - Nested subconfigs with `field(default_factory=...)`.
  - Config classes carry behaviour (`build()`, `prepare()`, `run()`), so the
    program needs real instances, not a dict-like view.
- Field types in use:
  - `int`, `float`, `bool`, `str`, `Path`
  - `Literal[...]`, including through PEP 695 aliases (`type Activation = Literal[...]`)
  - `X | None` for all of the above
  - `list[int] | None`, `list[str] | None`
  - `dict[int, int] | None`, `dict[str, Any] | None`
- Values are type-checked against the dataclass fields.
- Further validation in `__post_init__` (the GPU index check in `Config`),
  which runs when the config object is created.

## Config groups

- A field typed as an abstract base (`dm: DataModule`, `model: ModelFactory`)
  is a group. The options are registered by name in `CONFIG_GROUPS`:
  `dm` is `celeba` or `cmnist`, `model` is `fcn` or `cnn`.
- A group field has no default in the dataclass; the default option is chosen
  in the primary config.
- The valid keys under a group depend on the selected option:
  `model=cnn model.hidden_dim=10` is an error.
- YAML files in `configs/<group>/` add further named options. Each one extends
  an existing option through `defaults` and overrides some values:
  - `configs/dm/celeba_male_smiling.yaml` extends the registered `celeba`.
  - `configs/dm/celeba_male_smiling_small.yaml` extends another YAML option,
    so inheritance chains.
- A plain subconfig can also have named presets: `configs/opt/weight_decay_adamw.yaml`
  is selected with `opt=weight_decay_adamw`, although `opt` is not a group of
  classes. This needs `- opt: null` in the primary config's `defaults`.

## Primary configs

- A default primary config, named in the entry point (`configs/default_config.yaml`).
  Its `defaults` list names the schema (`base_config`) and selects an option
  for every group.
- Alternative primary configs, selected on the command line
  (`configs/good_run.yaml`). These select group options and override values
  anywhere in the tree, including under groups.
- Primary configs in a subdirectory (`configs/experiments/cmnist_cnn.yaml`),
  selected as `experiments/cmnist_cnn`. In Hydra this needs
  `# @package _global_` and absolute entries in `defaults` (`/dm: cmnist`);
  nematocyte is meant to need neither (see
  [design/composition.md](design/composition.md)).

## Command line

- `key=value` for top-level values (`seed=123`).
- `a.b=value` for nested values (`model.num_hidden=2`).
- `group=option` to select a group option (`model=cnn`, `dm=celeba_male_smiling`).
- Overrides apply on top of whatever the config files set.
- `--config-name NAME` / `-cn NAME` to pick a primary config.
- `--help` lists the groups with their options, and prints the full config
  with defaults. It reflects the other arguments: `model=cnn --help` shows the
  CNN keys.
- A clear error for an unknown key, naming the key and the class it was
  looked up in.

## Interpolation

- The `now` resolver with a `strftime` pattern, on the command line and in
  YAML: `wandb.group=${now:%Y-%m-%d}-cnn-dropout`.
- The value is fixed within a run. In a multirun, whether the jobs share it
  depends on the pattern; nematocyte computes it once for all jobs (see
  [design/interpolation.md](design/interpolation.md)).

## Multirun and sweeps

- `--multirun` with comma-separated values (`seed=42,43`). Several swept
  parameters give the Cartesian product.
- Jobs run sequentially by default.
- Launchers, selected with `hydra/launcher=NAME`: SLURM through submitit
  (`configs/hydra/launcher/`). Settings in use: `partition`, `qos`, `name`,
  `cpus_per_task`, `nodes`, `tasks_per_node`, `timeout_min`, `mem_gb`, `gres`,
  `constraint`, and `setup` shell commands.
- Sweepers, selected with `hydra/sweeper=NAME`: Optuna
  (`configs/hydra/sweeper/fcn_params.yaml`).
  - Settings in use: `sampler`, `study_name`, `n_trials`, `n_jobs`, `direction`.
  - Search spaces per parameter: `interval(a, b)`, `range(a, b)`,
    `choice(...)`, and `tag(log, ...)`.
  - The entry point returns a `float`, which is the objective.

## Entry point and logging

- A decorated `main` function that receives the composed config (`main.py`).
  Today this takes three steps: register the schema and groups, receive a
  `DictConfig`, convert it with `OmegaConf.to_object`.
- A plain-dict version of the resolved config for logging to W&B
  (`prepare_for_logging`): enums as strings, interpolations resolved.

## Testing

- Every YAML file under `configs/` is validated against the dataclasses
  without running an experiment, one test case per file
  (`tests/test_configs.py`). The test has to be told which directories hold
  primary configs rather than group options.

## Output directories

- Not used by the example, but in scope: a directory per run, as the place
  for model checkpoints, with a saved copy of the config (see
  [design/outputs.md](design/outputs.md)).
