"""Merging the values that the layers give for one key.

A mapping is merged into a mapping, deeply; anything else replaces what was
there. A mapping that has the key `_replace_` drops what the layers before
the one that wrote it gave at that place.
"""

from collections.abc import Iterator, Mapping
from dataclasses import MISSING, dataclass
from typing import Any, cast

from ._schema import REPLACE_KEY

__all__ = ["Part", "merge_parts", "replace_paths"]

type _Path = tuple[Any, ...]


@dataclass(frozen=True)
class Part:
    """What one file or override gave for a value."""

    value: Any
    """The value as written; ``MISSING`` for `~key`."""
    step: int
    """When it was set, from `next_step`."""
    since: int
    """When the layer that set it started: its `_replace_` drops what is older."""
    whole: bool = False
    """Whether it replaces what was there, even if both are mappings."""


def _as_mapping(value: Any) -> Mapping[Any, Any] | None:
    return cast("Mapping[Any, Any]", value) if isinstance(value, Mapping) else None


def replace_paths(value: Any, path: _Path = ()) -> Iterator[tuple[_Path, Any]]:
    """The places of the mappings in `value` that have `_replace_`, with its value."""
    mapping = _as_mapping(value)
    if mapping is None:
        return
    if REPLACE_KEY in mapping:
        yield path, mapping[REPLACE_KEY]
    for key, inner in mapping.items():
        if key != REPLACE_KEY:
            yield from replace_paths(inner, (*path, key))


def _strip(value: Any) -> Any:
    """`value` without the key `_replace_` in its mappings."""
    mapping = _as_mapping(value)
    if mapping is None:
        return value
    return {key: _strip(inner) for key, inner in mapping.items() if key != REPLACE_KEY}


def _drop(value: Any, path: _Path) -> Any:
    """`value` without what is at `path`; ``MISSING`` if that is all of it."""
    if not path:
        return MISSING
    mapping = _as_mapping(value)
    if mapping is None or path[0] not in mapping:
        return value
    result = dict(mapping)
    inner = _drop(mapping[path[0]], path[1:])
    if inner is MISSING:
        del result[path[0]]
    else:
        result[path[0]] = inner
    return result


def _merge(old: Any, new: Any) -> Any:
    """`new` merged into `old`: a mapping into a mapping, and deeply."""
    old_mapping, new_mapping = _as_mapping(old), _as_mapping(new)
    if old_mapping is None or new_mapping is None:
        return new
    result = dict(old_mapping)
    for key, inner in new_mapping.items():
        result[key] = _merge(result[key], inner) if key in result else inner
    return result


def merge_parts(base: Any, parts: list[Part]) -> Any:
    """The value that `parts` give on top of `base`, the default of the field.

    The base and each part that is older than the layer of a `_replace_`
    lose what they have at its place. ``MISSING`` if nothing is left.
    """
    replaces = [
        (part.since, path) for part in parts for path, _ in replace_paths(part.value)
    ]

    def older(value: Any, step: int) -> Any:
        for since, path in replaces:
            if step < since:
                value = _drop(value, path)
        return value

    result = older(base, -1)
    for part in parts:
        value = older(_strip(part.value), part.step)
        result = value if part.whole else _merge(result, value)
    return result
