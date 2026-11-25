# Interpolation

How `${...}` expressions in config values are resolved.

## Approach

Interpolation does not depend on the validation backend chosen in
[deserializer.md](deserializer.md). It is a pass over the raw tree, after all
files are merged and all overrides applied, and before any validation. Running
it last is what makes `${seed}` pick up a CLI override.

A 60-line prototype covered:

- `${a.b}` references. A value that is exactly one reference keeps the type of
  its target (`hidden_dim: ${dm.default_res}` is an `int`); a reference inside
  a longer string is formatted into it.
- `${name:arg}` resolvers from a registry, with `now` computed once at start-up
  (`${now:%Y-%m-%d}` is the only interpolation `hydra-example` uses).
- References to values that only exist as dataclass defaults. If the key is
  absent from the raw tree, the resolver follows the path through the schema,
  using the group choices (see [unions.md](unions.md)), and takes the field
  default. `seed${seed}-lr${opt.lr}` resolved with neither value set in YAML.
- Cycle detection and a clear error for unknown targets.

Because resolution happens before validation, an interpolated value is checked
against the type of the field it lands in, like any other value.

## Decided

### References between values are supported

`${a.b}` can refer to any value in the tree, as described above, although
`hydra-example` only uses the `now` resolver. Validation relies on this: a
named config is checked in the whole tree because it may refer to a value
outside its own node ([validation.md](validation.md)).

### `now` is computed once, also in a multirun

The launching process fixes the time, and every job of a multirun gets the
same value ([multirun.md](multirun.md)).

In Hydra this depends on the pattern. Tested with three jobs, with both the
basic and the submitit launcher:

| Pattern | Value in the jobs |
| --- | --- |
| `${now:%Y-%m-%d}`, `${now:%H-%M-%S}` | The same in every job |
| Any other, such as `${now:%M-%S.%f}` | Different in every job |

Hydra caches a resolved value per pattern and copies the cache to each job.
The two patterns in the first row are the ones Hydra uses for its own output
directory, so they are already in the cache when the jobs start. A sweep with
`wandb.group=${now:%Y-%m-%d_%H}` that runs for more than an hour ends up in
several groups.

### Details

- A reference can go into a dict value (`${opt.optimizer_kwargs.eps}`), but
  not into a list.
- Inside a longer string a value is formatted with `str()`, so `true`
  becomes `True` and null becomes `None`. A date, a datetime or a time from
  YAML is already its ISO string by then, as every value is kept in JSON
  form ([deserializer.md](deserializer.md#details-of-building)): `at ${dt}`
  becomes `at 2026-11-16T09:30:00`, the text that `dt` itself arrives as,
  and not the `str()` of the datetime, which has a space.
- There is no way to write a literal `${`.
- The argument of `now` is everything after the first colon, as it is.

## Not in the first version

### Interpolation strings as dataclass defaults

`name: str = "${seed}"` is not resolved. Interpolation is a pass over the raw
tree, and a dataclass default is not in it.

Supporting this would mean filling the raw tree with the defaults before
resolving, and it only type-checks for `str` fields. Until then, the value can
be set in a config file or derived in `__post_init__`.

### Custom resolvers and environment variables

`now` is the only resolver. In Hydra a program can register its own, and
OmegaConf's `oc.env` reads an environment variable. Both may be added
later.

### Relative references and nested interpolations

`${.sibling}` and `${a.${b}}` are not supported, and the error says so.
`hydra-example` uses neither.

Relative references matter less than in Hydra. A file's place in the config
directory fixes where its content goes in the tree
([composition.md](composition.md)), so a file in `configs/dm/` can always
write `${dm.default_res}`.
