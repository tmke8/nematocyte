# nematocyte

A configuration library for Python programs: a replacement for Hydra with
fewer features and more focus on usability.

## Where the design is

The design documents are the specification. Read the relevant ones before
changing code.

| File | What it covers |
| --- | --- |
| `FEATURES.md` | The scope: the Hydra features the example project relies on |
| `DIFFERENCES.md` | What works differently from Hydra, for users. Add an entry when a change makes nematocyte behave differently from Hydra in a way a user would notice |
| `design/deserializer.md` | Building config instances from the config tree with one call to pydantic, and TypedDict configs |
| `design/unions.md` | Config groups as tagged unions with `nematocyte.pick`, and how defaults work |
| `design/composition.md` | How config files are found and merged into one tree |
| `design/cli.md` | The override grammar, the flags, and the reserved `nematocyte` key |
| `design/interpolation.md` | `${...}` references and the `now` resolver |
| `design/api.md` | The public Python API |
| `design/validation.md` | `validate_configs`, for checking every YAML file in a test |
| `design/helptext.md` | `--help` and `--cfg` output, from `field(doc=...)` |
| `design/multirun.md` | `--multirun`, sweepers and launchers (submitit, Optuna) |
| `design/outputs.md` | Per-run output directories |
| `design/logging.md` | `nematocyte.log` and the opt-in `nematocyte.configure_logging()` |

- **"Decided"** sections are binding.
- **"Not in the first version"** means do not implement it, and do not build
  hooks for it.
- **"Rejected ideas"** were considered and turned down. Do not bring them
  back.
- **"How Hydra does it"** is background. Hydra's behaviour is only the
  specification where a "Decided" section says "as in Hydra".

If the documents do not cover something, or a decision turns out not to work
when implemented, stop and ask. Do not settle it silently in the code. When
a decision changes, update the design document in the same change, so the
documents stay true.

## The example

`example/` is a complete project that uses nematocyte, with its own
`pyproject.toml`; `python main.py` and `pytest` have to work there as its
README describes. Its tests run from inside `example/`. It is GPL-licensed
while the library is MIT, so do not copy code from it into `src/`.

## Constraints

- Do not use PyYAML; ruamel.yaml reads the config files as YAML 1.2.
- No OmegaConf, Hydra or ANTLR at runtime. The override parser is written by
  hand.
- submitit and Optuna are optional extras, imported only when their option
  is selected.
- `hydra-core`, `hydra-qol` and `tyro` are in the `reference` dependency
  group, to compare behaviour during development. The library must not
  import them.

## Checks

Everything is run through uv, and everything that CI runs
(`.github/workflows/ci.yml`) has to pass before work is reported as done.

Do not loosen the ruff or ty configuration in `pyproject.toml` to make an
error go away; fix the code, or ask. `ruff format` also formats the Python
code blocks in the Markdown files.

## Typing

The public API has to be understood by every type checker, not only ty:

- Use only standard typing features. No plugin for any checker, and nothing
  that only one checker understands.
- The API that carries the most type information has to work in all of
  them: the tagged unions with `Tag` and `nematocyte.pick`, the
  `@nematocyte.main(Config, ...)` decorator, which lets a checker verify
  that the function takes `Config`, and `nematocyte.compose`, which returns
  an instance of the class it is given.
- `tests/typing_samples.py` holds API usage samples with
  `typing.assert_type`; extend it with the API, and check it with ty,
  pyright and mypy.

## Tests

- Write tests along with the code, not at the end.
- Each "Decided" rule and each results table in the design documents should
  be covered by a test.
- Error messages are part of the product. Test that an error names the key,
  the file and the class involved; the exact wording is free.
- Launchers and sweepers need tests that run without a SLURM cluster.

## Working in this repository

- Commit and push only when asked.
