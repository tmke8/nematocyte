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

## Open questions

- **Interpolation strings as dataclass defaults** (`name: str = "${seed}"`).
  This requires filling the raw tree with defaults before resolving, and only
  type-checks for `str` fields. A derived value in `__post_init__` may be the
  better answer.
- **Relative references and nested interpolations** (`${.sibling}`,
  `${a.${b}}`): not used in `hydra-example`; leave out until needed.
