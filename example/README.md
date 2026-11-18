# An annotated example of using nematocyte

This code base is meant to show-case how to use nematocyte to manage configuration in a project.

> **Status:** nematocyte is implemented as described in the documents in [`../design/`](../design/), and everything on this page runs. The wording of the program output shown below may differ in details.

## Installing dependencies

If you have `uv` installed:
```bash
uv sync
```

## Running the code

To run the code with default configs, simply do:
```bash
python main.py
```

The output should be something like this:
```
[2026-11-17 11:50:24,875][src.run][INFO] - Starting a run with seed 42 on CPU.
[2026-11-17 11:50:24,932][src.run][INFO] - Model architecture:
Sequential(
  (0): Sequential(
    (0): Linear(in_features=28, out_features=28, bias=True)
    (1): LayerNorm((28,), eps=1e-05, elementwise_affine=True, bias=True)
    (2): GELU(approximate='none')
  )
  (1): Linear(in_features=28, out_features=10, bias=True)
)
```

The program reports what it does through Python's `logging` module; see [Logging](#logging) below.

No config file is involved here: the default values all come from the `Config` class in `src/run.py`.

We can, for example, set the seed with:
```bash
python main.py seed=123
```

To find out which arguments are available, run:
```bash
python main.py --help
```

It should look like this:

<details>
<summary> Help output (quite long) </summary>

```
Main configuration class for the code base.

dm=cmnist                            The dataset to train on.
                                       celeba   Data-module for the CelebA dataset.
                                     * cmnist   Data-module for the ColoredMNIST dataset.
                                     From files: celeba_male_blond,
                                     celeba_male_smiling, celeba_male_smiling_small
dm.root=/srv/galene0/shared/data     [path] Directory in which the datasets are
                                     stored.
dm.default_res=28                    [int] Resolution of the images.
dm.label_map=null                    [dict[int, int] | null] Mapping from the original
                                     labels to new labels.
dm.colors=null                       [list[int] | null] Indices of the colors to use.
dm.num_colors=10                     [int] Number of colors.
dm.val_prop=0.2                      [float] Proportion of the data that is used for
                                     validation.

model=fcn                            The model architecture.
                                     * fcn      Factory for fully-connected neural networks.
                                       cnn      Factory for a very simple CNN.
                                     From files: single_linear_layer
model.num_hidden=1                   [int] Number of hidden layers.
model.hidden_dim=null                [int | null] Width of the hidden layers. If not
                                     set, the input dimension is used.
model.activation=gelu                [relu|gelu|selu|silu] Activation function.

model.norm=LN                        Normalization layer to use.
                                       BN       Batch normalization.
                                     * LN       Layer normalization.
                                       none     No normalization layer.
                                     From files: slow_bn
model.norm.eps=                      [float] (not set)
model.norm.elementwise_affine=       [bool] (not set)

model.dropout_prob=0.0               [float] Dropout probability. No dropout is
                                     applied if this is 0.
model.final_bias=true                [bool] Whether the final linear layer has a bias.
model.input_norm=false               [bool] Whether to also apply the normalization
                                     layer to the input.

opt:                                 Config class for the optimisation.
                                     Presets: strong_weight_decay, weight_decay_adamw
opt.lr=5e-05                         [float] Learning rate.
opt.weight_decay=0.0                 [float] Weight decay.
opt.optimizer_cls=torch.optim.AdamW  [str] Import path of the optimizer class.
opt.optimizer_kwargs=null            [dict[str, any] | null] Further keyword arguments
                                     for the optimizer.

wandb:                               Dictionary of the parameters for `wandb.init()`.
wandb.name=                          [str | null] (not set)
wandb.dir=./local_logging            [str]
wandb.id=                            [str | null] (not set)
wandb.project=nematocyte-example     [str]
wandb.group=                         [str | null] (not set)
wandb.entity=example-org             [str]
wandb.tags=                          [list[str] | null] (not set)
wandb.job_type=                      [str | null] (not set)
wandb.mode=disabled                  [online|offline|disabled]
wandb.resume=                        [bool | allow|never|must|auto | null] (not set)

seed=42                              [int] Random seed.
gpu=-1                               [int] Index of the GPU to use. Set to -1 to use
                                     the CPU.

nematocyte:                     nematocyte's own settings.

nematocyte.launcher=basic       How the jobs of a multirun are run.
                                * basic            Runs the jobs one after another, in the process that was started.
                                  submitit_slurm   Submits the jobs to a SLURM cluster as one job array, through submitit.
                                From files: slurm_dipc, slurm_kyiv

nematocyte.sweeper=basic        How a multirun turns the swept values into jobs.
                                * basic    Runs one job for every combination of the swept values.
                                  optuna   Searches for the best values with Optuna.
                                From files: fcn_params
nematocyte.sweeper.params=null  [dict[str, any] | null] Sweeps by key, written as on the
                                command line: {seed: '42,43'}.

nematocyte.output_dir=null      [path | null] Directory of this start of the program,
                                which holds one directory per job. If not set, it is
                                outputs/DATE/TIME.
nematocyte.log=info             [debug|info|warn] Level of log messages, for
                                nematocyte.configure_logging() and
                                nematocyte.log_level().

Flags:
--help, -h               Print this help. It reflects the other arguments.
--cfg, -c {job,all}      Print the composed config instead of running.
--config-name, -cn NAME  Use the primary config called NAME.
+NAME                    Apply the primary config called NAME on top.
--multirun, -m           Run one job per combination of the swept values.
```

</details>

Every line shows a key with its current value, in the form in which you would set it on the command line, followed by the type of the value and a description. The descriptions come from the dataclasses in `src/` (see [Describing config values](#describing-config-values) below). The lines `From files:` and `Presets:` list the configs that are defined in YAML files. At the end come nematocyte's own settings, which are used for [multiruns](#multiruns) and for the output directory and logging, and the flags.

We can see that `seed` and `gpu` are "top-level" config values, but there are also a lot of values in subconfigs. For example, we can change the `num_hidden` value in the `model` config:
```bash
python main.py model.num_hidden=2
```
This should print a different model architecture.

But we can also see that there are different options available in the "model" group:
```
model=fcn                              The model architecture.
                                       * fcn      Factory for fully-connected neural networks.
                                         cnn      Factory for a very simple CNN.
                                       From files: single_linear_layer
```
This allows us to completely change the model architecture:
```bash
python main.py model=cnn
```

With the "cnn" architecture, the config key `model.hidden_dim` is not available anymore, so the following should raise an error:
```bash
python main.py model=cnn model.hidden_dim=10
```
Output:
```
Error in 'model.hidden_dim=10':
'hidden_dim' is not a field of SimpleCNNFactory (the option 'cnn' of 'model').
Available fields: kernel_size, pool_stride, activation
```

To see the available options for the "cnn" architecture, you can run:
```bash
python main.py model=cnn --help
```
Now the "model" part of the output should look like this:
```
model=cnn                              The model architecture.
                                         fcn      Factory for fully-connected neural networks.
                                       * cnn      Factory for a very simple CNN.
                                       From files: single_linear_layer
model.kernel_size=5                    [int] Kernel size of the convolutions.
model.pool_stride=2                    [int] Stride of the max-pooling layers.
model.activation=relu                  [relu|gelu|selu|silu] Activation function.
```
This tells us that we can set the `kernel_size`, `pool_stride`, and `activation` values for the "cnn" architecture. For example:
```bash
python main.py model=cnn model.kernel_size=3 model.pool_stride=1 model.activation=selu
```

### W&B config
To enable W&B logging, you can set the `wandb.mode` to "online" or "offline" (default is "disabled"):
```bash
python main.py wandb.mode=offline
```

### Values on the command line
Values on the command line are read like values in a YAML file: `123` is an integer, `1.5` is a float, `true` is a boolean and `null` is `None`. Anything else is a string.

If you want a string that looks like a number, you have to quote it, just as in YAML. `wandb.name` is a string, so this is an error:
```bash
python main.py wandb.name=123
```
Your shell removes quotes before the program sees them, so the quotes themselves have to be protected:
```bash
python main.py "wandb.name='123'"
```

Lists and dictionaries are written with brackets and braces:
```bash
python main.py 'wandb.tags=[baseline,cnn]' 'opt.optimizer_kwargs={eps:1e-8,amsgrad:true}'
```
A dictionary given like this replaces the previous value as a whole.

### Logging
The program uses Python's `logging` module instead of `print()`. Each module gets a logger named after it, as in `src/run.py`:
```python
log = logging.getLogger(__name__)

log.info("Starting a run with seed %d on %s.", self.seed, device)
```

`main.py` sets up `logging` at the start of the run, by calling `nematocyte.configure_logging()`. The messages then go to standard output, and to the file `run.log` in the output directory of the run, such as `outputs/2026-11-17/11-50-24/0/run.log`. If you don't want the file, call `nematocyte.configure_logging(file=False)`. If you don't call `configure_logging()` at all, nematocyte leaves `logging` alone.

How much is logged is not part of the experiment, so it is not a field of `Config`. It is one of nematocyte's own settings, `nematocyte.log`, which is `info` by default. To also see the debug messages, such as the full config of the run:
```bash
python main.py nematocyte.log=debug
```
This also shows the debug messages of the libraries that the program uses. With `nematocyte.log=warn`, only warnings and errors are shown.

If you set up `logging` in your own way, `nematocyte.log_level()` gives you the level, as `logging.DEBUG`, `logging.INFO` or `logging.WARNING`.

## Basics of config files
Let's say you are often working with a configuration of the CelebA dataset which looks like this:
```bash
python main.py dm=celeba dm.superclass=Smiling dm.subclass=Male
```

(Once again, you can set `dm=celeba` and use `--help` to see the available options for the CelebA dataset.)

But after a while, it gets tiring to always type this out. Instead, you can create a config file in the `configs/dm/` directory to store often-used datamodule configurations. For example, you can create a file `configs/dm/celeba_male_smiling.yaml` with the following content:
```yaml
defaults:
  _self_: celeba

superclass: Smiling
subclass: Male
```

The `_self_` entry under `defaults` says what this file is based on; here, the option `celeba` of the group. You can think of it as inheritance. A file in the directory of a config group always needs it, because it has to say which option of the group it is based on.

Subsequently, you can use the config file like this:
```bash
python main.py dm=celeba_male_smiling
```

You can still override any values on the command line, just as before:
```bash
python main.py dm=celeba_male_smiling dm.superclass=Blond_Hair dm.default_res=64
```

And you can create config files which inherit from other config files you created. For example, you can create a file `configs/dm/celeba_male_smiling_small.yaml` with the following content:
```yaml
defaults:
  _self_: celeba_male_smiling

default_res: 64
```

### Presets for subconfigs
The `opt` subconfig is not a config group: there is only one class, `OptimisationCfg`. You can still save often-used values for it, in `configs/opt/`. The file `configs/opt/weight_decay_adamw.yaml` looks like this:
```yaml
lr: 1.0e-3
weight_decay: 1.e-2
optimizer_cls: "torch.optim.AdamW"
```
There is no `defaults` entry, because there is no option to choose. You select the preset in the same way as an option of a group:
```bash
python main.py opt=weight_decay_adamw
```
This has the same effect as setting the three values on the command line.

A preset can also be based on another preset. The file `configs/opt/strong_weight_decay.yaml` takes the values of `weight_decay_adamw` and changes one of them:
```yaml
defaults:
  _self_: weight_decay_adamw

weight_decay: 0.1
```

## Multiruns
Often, we want to run the same experiment with only slightly different configurations. This can be done with the `--multirun` flag. For example, to run the code with seeds 42 and 43, we can do:
```bash
python main.py --multirun seed=42,43
```
The output should look like this (the lines that start with `[nematocyte]` come from nematocyte, the others are the log messages of the program):
```
[nematocyte] Launching 2 jobs locally
[nematocyte]        #0 : seed=42
[2026-11-17 11:50:32,455][src.run][INFO] - Starting a run with seed 42 on CPU.
[2026-11-17 11:50:32,508][src.run][INFO] - Model architecture:
Sequential(
  (0): Sequential(
    (0): Linear(in_features=28, out_features=28, bias=True)
    (1): LayerNorm((28,), eps=1e-05, elementwise_affine=True, bias=True)
    (2): GELU(approximate='none')
  )
  (1): Linear(in_features=28, out_features=10, bias=True)
)
[nematocyte]        #1 : seed=43
[2026-11-17 11:50:32,509][src.run][INFO] - Starting a run with seed 43 on CPU.
[2026-11-17 11:50:32,512][src.run][INFO] - Model architecture:
Sequential(
  (0): Sequential(
    (0): Linear(in_features=28, out_features=28, bias=True)
    (1): LayerNorm((28,), eps=1e-05, elementwise_affine=True, bias=True)
    (2): GELU(approximate='none')
  )
  (1): Linear(in_features=28, out_features=10, bias=True)
)
```

Each job writes its log messages to its own `run.log`, in `outputs/DATE/TIME/0/` and `outputs/DATE/TIME/1/`.

By default, nematocyte simply runs the code with the different config values sequentially. But if you have access to a SLURM cluster, you can also run the jobs in parallel. To do this, we need to set `nematocyte.launcher=...` to the appropriate value. nematocyte's own settings live under the key `nematocyte`, next to the settings of your program, and are set in the same way. In this code base, there is a config file at `configs/nematocyte/launcher/slurm_kyiv.yaml` that has the correct settings for the Kyiv machine. To use it, we can run:
```bash
python main.py --multirun seed=42,43 nematocyte.launcher=slurm_kyiv
```

You are not limited to iterating over just one parameter. You can also iterate over multiple parameters. For example, to run the code with seeds 42 and 43 and `model.num_hidden` set to 1, 2 and 3, you can do:
```bash
python main.py --multirun seed=42,43 model.hidden_dim=10 model.num_hidden=1,2,3
```
This will start 6 jobs: 2 seeds x 3 `model.num_hidden` values.

## Hyperparameter search
If you don't want to just blindly try out model parameters, you can also perform hyperparameter search with Optuna.

In order to do this, you need to return a value from the function decorated with `@nematocyte.main(...)` (located in `main.py`). The value should be a `float` and should represent the metric you want to optimize. For example, if you want to optimize the accuracy of the model, you can return the validation accuracy. If you want to minimize the loss, you can return the validation loss.

Create a config file in `configs/nematocyte/sweeper/`. The name doesn't matter, but we'll choose `fcn_params.yaml` and fill the content like this:
```yaml
defaults:
  # Select here which hyperparameter optimization library to use.
  _self_: optuna

sampler:
  warn_independent_sampling: true

study_name: fcn_params
n_trials: 25
n_jobs: 3
direction:
  # The direction depends on what you are returning from your main() function.
  - maximize

params:
  opt.lr: tag(log, interval(1.e-5, 1.e-3))
  model.num_hidden: range(1, 4)
  model.dropout_prob: choice(0.0, 0.1)
```

Let's break down what's going on here:

- First, the `_self_` entry under `defaults` says which sweeper this file configures. In this case, we want to use Optuna, so we set `optuna`.
- Then, some settings for Optuna follow.
- Finally, we specify `params`, which is the heart of this config file. Here, we specify the parameters we want to optimize. For each parameter, we have to specify the search space. For example, `opt.lr: tag(log, interval(1.e-5, 1.e-3))` means that we want to optimize the `opt.lr` parameter and we want to search in the log space between `1.e-5` and `1.e-3`. The `model.num_hidden: range(1, 4)` we want to search in the range from 1 to 3 (inclusive). The `model.dropout_prob: choice(0.0, 0.1)` means that we want to choose between 0.0 and 0.1.

Then start the search with:
```bash
python main.py --multirun nematocyte.sweeper=fcn_params
```

## Global experiment configs
We already saw that you can create config files to, for example, save a common data module configuration – such a config file is then put in `configs/dm/`. However, such config files can only configure one subcomponent of the whole configuration. This means you might end up with long commands like this one:
```bash
python main.py model=single_linear_layer model.activation=selu dm=celeba_male_smiling_small dm.download=false seed=1 opt.lr=0.001 opt.weight_decay=0.001
```
which you have to type over and over again.

Luckily, we can define a "primary config" in the top-level `configs/` directory, that acts as a *global* configuration. For the above command, we can create a file at `configs/good_run.yaml` with the following content:
```yaml
---
defaults:
  model: single_linear_layer
  dm: celeba_male_smiling_small

seed: 1

model:
  activation: selu

dm:
  download: false

opt:
  lr: 0.001
  weight_decay: 0.001
```

The `defaults` section selects options for config groups, in the same way as `model=...` and `dm=...` on the command line. Everything the file does not mention keeps its default from the `Config` class.

And then we can run the code with these config values by naming the file, without `.yaml`, after a `+`:
```bash
python main.py +good_run
```

You can still override values:
```bash
python main.py +good_run model.final_bias=false
```
Values on the command line always win over the file, whether you write them before or after `+good_run`. (Only the order of several `+` files among themselves matters; see below.)

Primary configs can also be kept in a subdirectory, as long as the directory does not have the name of a field of the `Config` class. For example, `configs/experiments/cmnist_cnn.yaml` is selected with:
```bash
python main.py +experiments/cmnist_cnn
```
The name after the `+` is the path of the file in `configs/`, with a slash and no `=`. (In Hydra, you would select such a file with `--config-name experiments/cmnist_cnn`.)

A primary config can also be based on another primary config, so that an experiment only has to say what is different about it. The file `configs/experiments/good_run_blond.yaml` looks like this:
```yaml
defaults:
  _self_: good_run
  dm: celeba_male_blond

seed: 2

dm:
  download: false
```
The `_self_` entry names the primary config to start from, written in the same way as after the `+`. Here, `good_run.yaml` is loaded first, and then this file is applied on top: it selects a different option for `dm` and changes the seed. What `good_run.yaml` sets for `model` and `opt` is kept.

Selecting an option for a config group always starts that group afresh: `dm` is now `celeba_male_blond` with its own values, and what `good_run.yaml` set under `dm` is gone. That is why `download: false` is set again here. The same happens on the command line: `python main.py +good_run model=cnn` gives you a CNN with its default values, without the `activation: selu` from `good_run.yaml`.

You can also give more than one primary config, and each one is applied on top of the ones before it. This is useful for settings that are not part of an experiment, but that you want for many experiments: while debugging, for example, you may not want to log to the wandb servers. The file `configs/presets/debug.yaml` looks like this:
```yaml
wandb:
  mode: offline
  dir: /tmp

nematocyte:
  log: debug
```
and is applied on top of any experiment with:
```bash
python main.py +experiments/cmnist_cnn +presets/debug
```
This works as if `debug.yaml` had `_self_: experiments/cmnist_cnn`: it only changes the values that it names. Unlike the other arguments, the order of the `+` arguments matters: they are applied from left to right, so where two of them set the same value, the later one wins. Values on the command line still win over all of them.

**Note**: It's very easy to fall into the trap of defining *everything* in these global configs, but that leads to lots of duplication. Try to put as much configuration as possible into the component-specific config files (i.e., those in `configs/dm/` and `configs/model` and so on), because those configs are easy to reuse across different experiments.

## Config validation
When you declare a field as, say, `gpu: int` in the dataclass, nematocyte will ensure that the value given for `gpu` is an `int`. However, often we need to validate configuration values in ways that cannot be expressed as a type. For example, in the code here we require that `gpu` is either `-1`, which indicates that no GPU is used, or the index of a GPU. To validate such constraints on the configuration value, we can use `__post_init__()` in the following way.

```python
@dataclass(kw_only=True)
class Config:
    """Main configuration class for the code base."""

    # ...all the other fields...
    gpu: int = field(
        default=-1, doc="Index of the GPU to use. Set to -1 to use the CPU."
    )

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        if self.gpu < -1:
            raise ValueError(
                f"Invalid GPU index {self.gpu}. "
                "Must be -1 (to use the CPU) or the index of a GPU."
            )
```

The `__post_init__` is run when nematocyte creates the configuration objects. You can try it out by passing an illegal value:

```bash
python main.py gpu=-2
```

The output should look something like this:

```
Error in the configuration:
Invalid GPU index -2. Must be -1 (to use the CPU) or the index of a GPU.
```

**Note**: Only check things in `__post_init__` that follow from the config values alone. Whether the GPU with the given index actually exists depends on the machine, so that check is at the beginning of `Config.run()` instead. The reason is that `__post_init__` also runs when the config files are checked in the tests (see below), on a machine that may look nothing like the one your experiments run on.

### Checking all config files
The YAML files are not checked by the type checker. If you rename a field in one of the dataclasses, a YAML file that still uses the old name only fails once someone tries to use it. The test in `tests/test_configs.py` catches this: it loads every config file in `configs/` in the same way as when the program runs, without running any experiments. Run it with:
```bash
pytest
```

## How to make your code easily configurable with nematocyte
What we found is the best method to structure your code to make it easy to configure is the "builder" or "factory" pattern.

This means you have a dataclass that contains all the configuration values for a particular component of your code (e.g. the model, the data, the optimiser, etc.). And then this class has a `build()` or `init()` method that takes additional arguments which are only available at runtime (e.g. the input size of the model, the number of classes in the dataset, etc.), and then instantiates the component.

For example, say your code has such a function for constructing a convolutional neural network:

```python
def build_simple_cnn(
    in_dim: int,
    out_dim: int,
    kernel_size: int = 5,
    pool_stride: int = 2,
    activation: Activation = "relu",
) -> nn.Sequential:
    return nn.Sequential(...)
```

Then we can pull the configurable parameters into a dataclass and re-write it like this

```python
@dataclass(kw_only=True, frozen=True)
class SimpleCNNFactory:
    """Factory for a very simple CNN."""

    kernel_size: int = field(default=5, doc="Kernel size of the convolutions.")
    pool_stride: int = field(default=2, doc="Stride of the max-pooling layers.")
    activation: Activation = field(default="relu", doc="Activation function.")

    def build(self, in_dim: int, *, out_dim: int) -> nn.Sequential:
        return nn.Sequential(...)  # use `self.kernel_size` etc.
```

### Declaring a config group
Which model factories can be chosen is declared in the `Config` class, in the type of the `model` field. Each option is given the name under which it is selected:

```python
@dataclass(kw_only=True)
class Config:
    model: (
        Annotated[FcnFactory, option("fcn")]
        | Annotated[SimpleCNNFactory, option("cnn")]
    ) = field(default_factory=FcnFactory)
```

The default says which option is used if nothing is selected.

### Describing config values
To tell the users of your program what a config value is for, give the field a `doc`:

```python
seed: int = field(default=42, doc="Random seed.")
```

This text is shown by `--help`, next to the key, its type and its current value. You don't have to repeat the type or the default in the text.

The description of an option of a config group is the docstring of its class, and the description of a group is the `doc` of its field in the `Config` class.

### A default primary config
This example has no default primary config: whatever no file sets comes from the defaults in the `Config` class, so `python main.py` runs without naming a file. But if your `Config` is a `TypedDict` instead of a dataclass, there is no place in Python for default values. Then the defaults go in a YAML file, say `configs/base.yaml`, which you name in the entry point:

```python
@nematocyte.main(Config, config_path="configs", config_name="base")
def main(config: Config) -> float: ...
```

`base.yaml` is then the primary config of every run, and a file that you name with `+` is applied on top of it: `python main.py +experiments/cmnist_cnn` starts from `base.yaml`. That is usually what you want. If you don't want to inherit from `base.yaml`, name the primary config with `--config-name` (or the shorthand `-cn`) instead, which replaces the default:
```bash
python main.py --config-name experiments/cmnist_cnn
```
Everything that has no default then has to come from that file, or from a file that it extends with `_self_`. `--config-name ''` means no primary config at all. Files named with `+` are still applied on top, as in `python main.py -cn experiments/cmnist_cnn +presets/debug`.

## Using the current date in config values
nematocyte comes with a built-in `now` resolver which formats the current date and time with a [`strftime`](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) pattern. This is useful, for example, for grouping W&B runs by the day they were started:
```bash
python main.py 'wandb.group=${now:%Y-%m-%d}-cnn-dropout'
```
This sets `wandb.group` to something like `2026-10-16-cnn-dropout`. Note the single quotes: without them, the shell would try to expand `${now:...}` itself.

You can use any `strftime` pattern, e.g. `${now:%Y-%m-%d_%H-%M}` to also include the time. The resolver also works in config files:
```yaml
wandb:
  group: ${now:%Y-%m-%d}-cnn-dropout
```

**Note**: the value is fixed when the program starts, so it doesn't change. This also holds for a multirun: all jobs get the same value, even if your sweep runs past midnight.

## Structure of the code

- `main.py`: The main entry point of the code. It sets up nematocyte and logging, and then calls the main `run()` function.
- `src/`
  - `run.py`: Contains the main `Config` class that is used to define valid config values, including the options of the config groups. It also contains the `run()` function that is called by `main.py`.
  - `datasets.py`: Contains the data module classes that are used to load the data.
  - `models.py`: Contains the model factory classes that are used to create the model.
  - `optimisation.py`: Contains the `OptimisationCfg` class that is used to build the optimiser that trains the model.
  - `logging.py`: Contains the `WandbCfg` class that is used to set up Weights & Biases logging.
- `configs/`
  - `good_run.yaml`: A primary config.
  - `experiments/`: Contains more primary configs.
  - `presets/`: Contains primary configs to apply on top of an experiment, such as `debug.yaml`.
  - `dm/`: Contains the config files for the different datasets.
  - `model/`: Contains the config files for the different model architectures.
    - `norm/`: Contains the config files for the normalization layer of the FCN.
  - `opt/`: Contains the presets for the optimiser.
  - `nematocyte/`
    - `launcher/`: Contains the SLURM launcher config files.
    - `sweeper/`: Contains the Optuna sweeper config files.
- `tests/test_configs.py`: Checks that all config files in `configs/` are valid, without running any experiments (run with `pytest`).
