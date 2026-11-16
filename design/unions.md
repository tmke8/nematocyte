# Unions and config groups

How the concrete class is chosen for a field that can hold one of several
config classes.

## No tagging

No tagging strategy is needed. In Hydra the choice of class never lives in the
data: it lives in the defaults list (`- model: cnn`), in a group file's
inheritance chain (`celeba_male_smiling_small` -> `celeba_male_smiling` ->
`celeba` -> `CelebADataModule`), or in a CLI override (`model=cnn`). All of
these are resolved during composition, so by the time values are validated
the composer holds a concrete class for every group node. Structuring never
has to guess which member of a union a dict belongs to.

Two things follow:

- The same base class can appear at several paths with different choices
  (`encoder: ModelFactory`, `decoder: ModelFactory`); the choice is keyed by
  path, not by type. Tested with both cattrs and pydantic.
- Because the data carries no tag, a dumped config is not self-describing.
  Whatever is written out for reproducibility has to record the choices next
  to the values, e.g. as a `defaults` list, so that it can be loaded again.

How the chosen class is then handed to the validator is covered in
[deserializer.md](deserializer.md): pydantic cannot take it directly, so
nematocyte walks the dataclass nodes itself.

## Open questions

- **How groups are declared**: a registry dict as in `hydra-example`, or
  derived from a union annotation as in tyro.
