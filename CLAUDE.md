# nematocyte

A configuration library for Python programs: a replacement for Hydra with
fewer features and more focus on usability. The config is a tree of stdlib
dataclasses, filled from YAML files and command-line overrides.

## Status

The library is implemented as designed, in `src/nematocyte/`, with tests in
`tests/`. Where the design documents were silent or ambiguous, the
implementation guessed; every guess is listed in `design/guesses.md` and has
not been reviewed yet. A guess that is confirmed should move into the
"Decided" section of the document it belongs to.

## Where the design is

Read these before writing code. They are the specification.

| File | What it covers |
| --- | --- |
| `FEATURES.md` | The scope: the Hydra features the example project relies on |
| `DIFFERENCES.md` | What works differently from Hydra, for users. Add an entry when a change makes nematocyte behave differently from Hydra in a way a user would notice |
| `design/deserializer.md` | Building dataclass instances from the config tree, with pydantic for field values |
| `design/unions.md` | Config groups as annotated unions, `nematocyte.option`, and how defaults work |
| `design/composition.md` | How config files are found and merged into one tree |
| `design/cli.md` | The override grammar, the flags, and the reserved `nematocyte` key |
| `design/interpolation.md` | `${...}` references and the `now` resolver |
| `design/api.md` | The public Python API |
| `design/validation.md` | `validate_configs`, for checking every YAML file in a test |
| `design/helptext.md` | `--help` and `--cfg` output, from `field(doc=...)` |
| `design/multirun.md` | `--multirun`, sweepers and launchers (submitit, Optuna) |
| `design/outputs.md` | Per-run output directories |
| `design/logging.md` | `nematocyte.log` and the opt-in `nematocyte.configure_logging()` |
| `design/guesses.md` | What the implementation decided on its own, to be reviewed |

`example/` is a complete project written against the designed API. It is the
target: when the library is done, `python main.py` and `pytest` have to work
there as its README describes. It is a separate project with its own
`pyproject.toml`, and is GPL-licensed while the library is MIT, so do not
copy code from it into `src/`.

## How to read the design documents

- **"Decided"** sections are binding.
- **"Not in the first version"** means do not implement it, and do not build
  hooks for it.
- **"Rejected ideas"** were considered and turned down. Do not bring them
  back.
- **"How Hydra does it"** is background. Hydra's behaviour is only the
  specification where a "Decided" section says "as in Hydra".
- The tables of inputs and results in the documents come from prototypes
  that are not in the repository. Turn them into tests.

If the documents do not cover something, or a decision turns out not to work
when implemented, stop and ask. Do not settle it silently in the code. When
a decision changes, update the design document in the same change, so the
documents stay true.

## Constraints

- Python 3.14 or later. `dataclasses.field(doc=...)` is required.
- Runtime dependencies: pydantic 2.12 or later, ruamel.yaml, which
  reads the config files as YAML 1.2 (`design/composition.md`), and
  typing_extensions 4.16 or later, for `sentinel`. Do not use PyYAML.
- submitit and Optuna are optional extras, imported only when their option
  is selected.
- No OmegaConf, Hydra or ANTLR at runtime. The override parser is written by
  hand.
- `hydra-core`, `hydra-qol` and `tyro` are in the `reference` dependency
  group, to compare behaviour during development. The library must not
  import them.

## Checks

Everything is run through uv. All of these have to pass before work is
reported as done:

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
```

The ruff and ty configuration in `pyproject.toml` is copied from
`hydra-qol` (https://github.com/tmke8/hydra-qol). Do not loosen it to make
an error go away; fix the code, or ask.

`ruff format` also formats the Python code blocks in the Markdown files.

## Typing

The public API has to be understood by every type checker, not only ty:

- Everything public is fully annotated, and `py.typed` is shipped.
- Use only standard typing features. No plugin for any checker, and nothing
  that only one checker understands.
- The API that carries the most type information has to work in all of
  them: the annotated unions with `nematocyte.option`, the
  `@nematocyte.main(Config, ...)` decorator, which should let a checker
  verify that the function takes `Config`, and `nematocyte.compose`, which
  returns an instance of the class it is given.
- Keep a file of API usage samples with `typing.assert_type`, and check it
  with ty, pyright and mypy.

## Tests

- Tests are in `tests/`, run with pytest. Write them along with the code,
  not at the end.
- Each "Decided" rule and each results table in the design documents should
  be covered by a test.
- Error messages are part of the product. Test that an error names the key,
  the file and the class involved; the exact wording is free.
- The example's own tests run from inside `example/`, not from the root.
- Launchers and sweepers need tests that run without a SLURM cluster.

## CI

At the end, add a GitHub Actions workflow modelled on the one in
`hydra-qol` (`.github/workflows/ci.yml`), with these jobs:

- lint: `ruff check` and `ruff format --diff`
- type check: `ty check`, plus pyright and mypy on the API usage samples
- test: `pytest`
- example: the tests in `example/` and a run of `python main.py`

## Suggested order

Each step can be tested on its own:

1. The walk over the dataclasses, groups and defaults (`deserializer.md`,
   `unions.md`)
2. Finding and merging config files (`composition.md`)
3. The override parser and applying overrides (`cli.md`)
4. Interpolation (`interpolation.md`)
5. The entry point, `compose` and `to_dict` (`api.md`)
6. `validate_configs` (`validation.md`)
7. `--help` and `--cfg` (`helptext.md`, `cli.md`)
8. Output directories (`outputs.md`)
9. `--multirun` with the basic sweeper and launcher, then submitit, then
   Optuna (`multirun.md`)

## Working in this repository

- Commit and push only when asked. Commits go directly to `main`.
- The original Hydra version of the example is in `../hydra-example`, for
  comparison.
