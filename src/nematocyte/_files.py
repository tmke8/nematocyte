"""Reading config files: YAML 1.2 with ruamel.yaml, or JSON5 with json5."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from ._errors import ConfigError
from ._schema import DEFAULTS_KEY, DOC_KEY, SELF_KEY

__all__ = [
    "SUFFIXES",
    "ConfigFile",
    "check_unique",
    "defaults_block",
    "defaults_entry",
    "display_path",
    "find_file",
    "list_files",
    "load_file",
]

YAML_SUFFIXES = (".yaml", ".yml")
JSON5_SUFFIX = ".json5"
SUFFIXES = (*YAML_SUFFIXES, JSON5_SUFFIX)

# A key in `_defaults_` other than `_self_`: the name of a field.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class ConfigFile:
    """The content of one config file."""

    path: Path
    label: str
    """The path of the file as it is shown in messages."""
    base: str | None
    """What `_self_` names under `_defaults_`: the config that the file extends."""
    defaults: dict[tuple[str, ...], str | None]
    """The other entries of `_defaults_`, by their path below the node of the
    file. `None` switches off an optional subconfig or group."""
    values: dict[str, Any]
    doc: str = ""
    """What the file says about itself, in its `_doc_`; empty if nothing."""


def display_path(path: Path) -> str:
    """The path as it is shown in messages: relative to the working directory."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_file(path: Path) -> ConfigFile:
    """Read a config file and split off its `_defaults_` and its `_doc_`."""
    label = display_path(path)
    if path.suffix == JSON5_SUFFIX:
        content = _read_json5(path, label)
    else:
        content = _read_yaml(path, label)
    if not isinstance(content, dict):
        raise ConfigError(
            f"{label}: a config file has to be a mapping at the top level, "
            f"not a {type(content).__name__}."
        )
    # Only the keys at the top level have to be strings; further down, a key
    # that is not a string can be a key of a dict-typed field.
    for key in content:
        if not isinstance(key, str):
            raise ConfigError(f"{label}: the key {key!r} is not a string.")

    doc = content.pop(DOC_KEY, None)
    if doc is None:
        doc = ""
    if not isinstance(doc, str):
        raise ConfigError(
            f"{label}: '{DOC_KEY}' describes the file, and has to be a string, "
            f"not {doc!r}."
        )
    base, defaults = _read_defaults(content.pop(DEFAULTS_KEY, None), label)
    return ConfigFile(
        path=path,
        label=label,
        base=base,
        defaults=defaults,
        values=content,
        doc=doc,
    )


def _read_defaults(
    defaults: Any, label: str
) -> tuple[str | None, dict[tuple[str, ...], str | None]]:
    """The `_self_` of a file's `_defaults_`, and its other entries by path."""
    if defaults is None:
        return None, {}
    if isinstance(defaults, list):
        raise ConfigError(
            f"{label}: '{DEFAULTS_KEY}' is a list, as in Hydra. In nematocyte it "
            "is a mapping from a field to the name of the config to load there, "
            f"such as\n{format_defaults({'model': 'cnn'})}"
        )
    if not isinstance(defaults, dict):
        raise ConfigError(f"{label}: '{DEFAULTS_KEY}' has to be a mapping.")
    if _has_dotted_key(defaults):
        raise ConfigError(
            f"{label}: a key in '{DEFAULTS_KEY}' is a field, not a dotted path. "
            "A config for a nested field is selected in a nested mapping, in "
            "which '_self_' selects the config for the field itself:\n"
            f"{format_defaults(_undotted(defaults))}"
        )
    base: str | None = None
    if SELF_KEY in defaults:
        name = defaults[SELF_KEY]
        if not isinstance(name, str):
            raise ConfigError(
                f"{label}: the entry '{SELF_KEY}' in '{DEFAULTS_KEY}' names the "
                f"config that the file extends, but is {name!r}."
            )
        base = name
    entries: dict[tuple[str, ...], str | None] = {}
    _read_entries(defaults, (), entries, label)
    return base, entries


def _read_entries(
    mapping: dict[Any, Any],
    prefix: tuple[str, ...],
    entries: dict[tuple[str, ...], str | None],
    label: str,
) -> None:
    """Add the entries of `mapping`, at `prefix` in `_defaults_`, to `entries`.

    `_self_` in a nested mapping is the entry for the field that holds it.
    """
    where = f"under {'.'.join(prefix)!r} " if prefix else ""
    for key, name in mapping.items():
        if key == SELF_KEY:
            if not prefix:
                continue  # The config that the file extends.
            path = prefix
        elif isinstance(key, str) and _NAME.fullmatch(key):
            path = (*prefix, key)
        else:
            raise ConfigError(
                f"{label}: the key {key!r} {where}in '{DEFAULTS_KEY}' is not "
                f"the name of a field. A key in '{DEFAULTS_KEY}' is a field, or "
                f"'{SELF_KEY}'."
            )
        entry = ".".join(path)
        if isinstance(name, dict) and key != SELF_KEY:
            if not name:
                raise ConfigError(
                    f"{label}: the entry {entry!r} in '{DEFAULTS_KEY}' is an "
                    f"empty mapping, which selects nothing. Name a config for "
                    f"the field, or for its fields, or remove the entry."
                )
            _read_entries(name, path, entries, label)
        elif isinstance(name, str) or name is None:
            # Null is checked against the field when the file is applied.
            entries[path] = name
        else:
            raise ConfigError(
                f"{label}: the entry {entry!r} in '{DEFAULTS_KEY}' has to name "
                f"a config, but is {name!r}."
            )


def _has_dotted_key(mapping: dict[Any, Any]) -> bool:
    return any(
        (isinstance(key, str) and "." in key)
        or (isinstance(value, dict) and _has_dotted_key(value))
        for key, value in mapping.items()
    )


def _undotted(mapping: dict[Any, Any]) -> dict[Any, Any]:
    """`mapping` with each dotted key written as a nested mapping, for a hint."""
    result: dict[Any, Any] = {}
    for key, value in mapping.items():
        path = key.split(".") if isinstance(key, str) else [key]
        if isinstance(value, dict):
            value = _undotted(value)
        for segment in reversed(path[1:]):
            value = {segment: value}
        _merge_entry(result, path[0], value)
    return result


def _merge_entry(mapping: dict[Any, Any], key: Any, value: Any) -> None:
    """Add an entry to `mapping`; a name next to a mapping goes to its `_self_`."""
    if key not in mapping:
        mapping[key] = value
        return
    old = mapping[key]
    if not isinstance(old, dict):
        if not isinstance(value, dict):
            mapping[key] = value
            return
        old = mapping[key] = {SELF_KEY: old}
    if not isinstance(value, dict):
        value = {SELF_KEY: value}
    for inner_key, inner in value.items():
        _merge_entry(old, inner_key, inner)


def format_defaults(entries: dict[Any, Any], indent: str = "    ") -> str:
    """`_defaults_` with `entries`, as a block of YAML for a message."""
    return "\n".join(_yaml_lines({DEFAULTS_KEY: entries}, indent))


def _yaml_lines(mapping: dict[Any, Any], indent: str) -> list[str]:
    lines: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{indent}{key}:")
            lines.extend(_yaml_lines(value, indent + "  "))
        else:
            lines.append(f"{indent}{key}: {'null' if value is None else value}")
    return lines


def defaults_entry(path: tuple[str, ...], name: str | None) -> str:
    """The entry under `_defaults_` for `name` at `path`, on one line.

    `ema: null` for `ema`, and `outer: {ema: null}` for `outer.ema`.
    """
    text = "null" if name is None else name
    for key in path[1:][::-1]:
        text = f"{{{key}: {text}}}"
    return f"{path[0]}: {text}"


def defaults_block(path: tuple[str, ...], name: str | None) -> str:
    """`_defaults_` with the entry for `name` at `path`, as a block of YAML."""
    value: Any = name
    for key in reversed(path[1:]):
        value = {key: value}
    return format_defaults({path[0]: value})


def _read_yaml(path: Path, label: str) -> Any:
    """The content of a YAML file; an empty file is an empty mapping."""
    yaml = YAML(typ="safe", pure=True)
    yaml.allow_duplicate_keys = False
    try:
        with path.open(encoding="utf-8") as stream:
            content = yaml.load(stream)
    except YAMLError as exc:
        raise ConfigError(f"{label} is not valid YAML:\n{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{label} cannot be read: {exc}") from exc
    return {} if content is None else content


def _read_json5(path: Path, label: str) -> Any:
    """The content of a JSON5 file, read with the optional `json5`."""
    try:
        import json5  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            f"{label} is a JSON5 file, and reading it needs the extra "
            "'json5': pip install 'nematocyte[json5]'"
        ) from exc
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{label} cannot be read: {exc}") from exc
    try:
        return json5.loads(text, allow_duplicate_keys=False)
    except ValueError as exc:
        # The parser calls the text "<string>"; the label already names it.
        message = str(exc).replace("<string>:", "line ")
        raise ConfigError(f"{label} is not valid JSON5:\n{message}") from exc


def _clash(paths: list[Path]) -> ConfigError:
    names = " and ".join(display_path(path) for path in paths)
    return ConfigError(
        f"{names} are both there, so the name {paths[0].stem!r} is ambiguous. "
        "Remove or rename one of them."
    )


def find_file(directory: Path, name: str) -> Path | None:
    """The config file called `name` in `directory`, if there is one.

    A name with a file for each suffix is an error.
    """
    found = [
        candidate
        for suffix in SUFFIXES
        if (candidate := directory / f"{name}{suffix}").is_file()
    ]
    if len(found) > 1:
        raise _clash(found)
    return found[0] if found else None


def check_unique(paths: list[Path]) -> None:
    """Raise if two of the config files `paths` differ only in their suffix."""
    seen: dict[Path, Path] = {}
    for path in sorted(paths):
        stem = path.with_suffix("")
        if stem in seen:
            raise _clash([seen[stem], path])
        seen[stem] = path


def list_files(directory: Path) -> dict[str, Path]:
    """The config files in `directory`, by name."""
    if not directory.is_dir():
        return {}
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in SUFFIXES
    ]
    check_unique(paths)
    return {path.stem: path for path in paths}
