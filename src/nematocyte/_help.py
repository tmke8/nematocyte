"""The text of `--help` and of `--cfg`."""

from dataclasses import MISSING
import io
import shutil
import textwrap
from typing import Any

from ruamel.yaml import YAML

from ._compose import Composer
from ._overrides import format_value
from ._schema import RESERVED_KEY, FieldInfo, Kind, class_doc
from ._tree import Node
from ._types import type_text

__all__ = ["format_cfg", "format_help"]

_MAX_LEFT = 40
_REQUIRED = "???"

_FLAGS = [
    ("--help, -h", "Print this help. It reflects the other arguments."),
    ("--cfg, -c {job,all}", "Print the composed config instead of running."),
    ("--config-name, -cn NAME", "Use the primary config called NAME."),
    ("+NAME", "Apply the primary config called NAME on top."),
    ("+KEY=VALUE", "Set KEY, a key inside a dict, and keep the dict's other keys."),
    ("~KEY", "Leave out KEY, a NotRequired key of a TypedDict."),
    ("--multirun, -m", "Run one job per combination of the swept values."),
]


def _terminal_width() -> int:
    columns = shutil.get_terminal_size((88, 24)).columns
    return max(60, min(columns, 100))


def _value_text(node: Node, info: FieldInfo) -> str:
    value = node.value(info)
    if value is MISSING:
        return _REQUIRED if info.required else ""
    return format_value(value)


def _missing_note(node: Node, info: FieldInfo) -> str:
    """The note for a value that nothing set: required, or not set."""
    if node.value(info) is not MISSING:
        return ""
    return " (required)" if info.required else " (not set)"


def _listing(heading: str, docs: dict[str, str]) -> list[str]:
    """Config files, one per line under `heading`, each with its doc."""
    if not docs:
        return []
    width = max(6, *(len(name) for name in docs)) + 3
    return [
        heading,
        *(f"  {name:<{width}}{doc}".rstrip() for name, doc in docs.items()),
    ]


class _Help:
    def __init__(self, composer: Composer, width: int) -> None:
        self.composer = composer
        self.width = width
        # (left, wrapped text, lines that are kept as they are) or None for a gap
        self.rows: list[tuple[str, str, list[str], list[str]] | None] = []

    def gap(self) -> None:
        if self.rows and self.rows[-1] is not None:
            self.rows.append(None)

    def row(
        self,
        left: str,
        text: str,
        fixed: list[str] | None = None,
        after: list[str] | None = None,
    ) -> None:
        self.rows.append((left, text, fixed or [], after or []))

    def node(self, node: Node, prefix: str, *, only: str | None = None) -> None:
        """Add the rows for the fields of `node`.

        At the root, nematocyte's own settings are left out, or with `only`
        they are the one field that is shown.
        """
        for info in node.fields.values():
            if not node.path and (info.name == RESERVED_KEY) != (only is not None):
                continue
            key = prefix + info.name
            if info.kind is Kind.VALUE:
                doc = info.doc
                note = _missing_note(node, info)
                text = f"[{type_text(info.annotation)}]{note} {doc}".rstrip()
                self.row(f"{key}={_value_text(node, info)}", text)
                continue
            child = node.children.get(info.name)
            files = self.composer.named_config_docs((*node.path, info.name), info)
            self.gap()
            if info.kind is Kind.GROUP:
                selected = child.option if child is not None else None
                if node.is_off(info):
                    selected = "null"
                width = max(6, *(len(name) for name in info.options)) + 3
                options = []
                for name, cls in info.options.items():
                    mark = "*" if name == selected else " "
                    options.append(f"{mark} {name:<{width}}{class_doc(cls)}".rstrip())
                options += _listing("From files:", files)
                self.row(f"{key}={selected or _REQUIRED}", info.doc, options)
            else:
                assert info.config_cls is not None
                doc = info.doc or class_doc(info.config_cls)
                left = f"{key}:"
                if info.optional:
                    # An optional subconfig shows its type, and `null` if off.
                    doc = f"[{type_text(info.annotation)}] {doc}".rstrip()
                    left = left if child is not None else f"{key}=null"
                self.row(left, doc, _listing("Presets:", files))
            if child is not None:
                self.node(child, key + ".")
            self.gap()

    def render(self) -> list[str]:
        lefts = [len(row[0]) for row in self.rows if row is not None]
        column = min(max(lefts, default=0) + 2, _MAX_LEFT)
        room = max(20, self.width - column)
        indent = " " * column
        lines: list[str] = []
        while self.rows and self.rows[-1] is None:
            self.rows.pop()
        for row in self.rows:
            if row is None:
                lines.append("")
                continue
            left, text, fixed, after = row
            right = textwrap.wrap(text, room) if text else []
            right += fixed
            for paragraph in after:
                right += textwrap.wrap(paragraph, room)
            if not right:
                lines.append(left)
            elif len(left) + 2 > column:
                # As in argparse: a long key gets a line of its own.
                lines.append(left)
                lines.extend(indent + line for line in right)
            else:
                lines.append(f"{left:<{column}}{right[0]}")
                lines.extend(indent + line for line in right[1:])
        return lines


def format_help(composer: Composer, root: Node, width: int | None = None) -> str:
    """The text of `--help` for a composed tree."""
    width = _terminal_width() if width is None else width
    lines: list[str] = []
    doc = class_doc(composer.config_cls)
    if doc:
        lines += [*textwrap.wrap(doc, width), ""]
    helper = _Help(composer, width)
    helper.node(root, "")
    lines += helper.render()
    if primary := composer.primary_config_docs():
        lines += [
            "",
            *_listing("Primary configs, for --config-name NAME or +NAME:", primary),
        ]
    # nematocyte's own settings are a block of their own, so that their keys
    # do not widen the column of the program's keys.
    own = _Help(composer, width)
    own.node(root, "", only=RESERVED_KEY)
    lines += ["", *own.render()]
    lines += ["", "Flags:"]
    flag_width = max(len(flag) for flag, _ in _FLAGS) + 2
    lines += [f"{flag:<{flag_width}}{text}" for flag, text in _FLAGS]
    return "\n".join(lines) + "\n"


def _yaml() -> YAML:
    """A dumper for the values of `--cfg`; making one is slow, so it is reused."""
    yaml = YAML(typ="safe", pure=True)
    yaml.default_flow_style = False
    yaml.width = 4096
    return yaml


def _dump_yaml(yaml: YAML, key: str, value: Any) -> list[str]:
    stream = io.StringIO()
    yaml.dump({key: value}, stream)
    return stream.getvalue().splitlines()


def _comment(text: str, indent: str, width: int) -> list[str]:
    room = max(30, width - len(indent) - 2)
    return [f"{indent}# {line}" for line in textwrap.wrap(text, room)]


def _cfg_node_doc(node: Node, info: FieldInfo) -> str:
    """The comment above a subconfig or a group in `--cfg`; empty if none."""
    if info.kind is Kind.GROUP:
        child = node.children.get(info.name)
        selected = child.option if child is not None else "none"
        if node.is_off(info):
            selected = "null"
        note = f"Options: {', '.join(info.options)}. Selected: {selected}."
        return f"{info.doc} {note}".strip()
    assert info.config_cls is not None
    doc = info.doc or class_doc(info.config_cls)
    if info.optional:
        # An optional subconfig shows its type, as a value does.
        doc = f"{doc} ({type_text(info.annotation)})".strip()
    return doc


def _cfg_node(
    node: Node,
    indent: str,
    width: int,
    lines: list[str],
    yaml: YAML,
    *,
    skip: str | None,
) -> None:
    empty = True
    for info in node.fields.values():
        if info.name == skip:
            continue
        empty = False
        if info.kind is Kind.VALUE:
            doc = info.doc
            lines += _comment(
                f"{doc} ({type_text(info.annotation)})".strip(), indent, width
            )
            value = node.value(info)
            if value is MISSING and not info.required:
                lines.append(f"{indent}# {info.name}: not set")
            elif value is MISSING:
                lines.append(f"{indent}{info.name}: {_REQUIRED}  # a value is required")
            else:
                lines += [indent + line for line in _dump_yaml(yaml, info.name, value)]
            continue
        child = node.children.get(info.name)
        lines += _comment(_cfg_node_doc(node, info), indent, width)
        if node.is_off(info):
            lines.append(f"{indent}{info.name}: null")
            continue
        if child is None:
            lines.append(f"{indent}{info.name}: {_REQUIRED}  # no option is selected")
            continue
        before = len(lines)
        lines.append(f"{indent}{info.name}:")
        _cfg_node(child, indent + "  ", width, lines, yaml, skip=None)
        # Only comments below the key, e.g. for a TypedDict with no key set.
        if all(line.lstrip().startswith("#") for line in lines[before + 1 :]):
            lines[before] += " {}"
    if empty and not indent:
        lines.append("{}")


def format_cfg(
    composer: Composer, root: Node, *, own_settings: bool, width: int | None = None
) -> str:
    """The text of `--cfg`: the composed config as YAML, with the docs as comments.

    Interpolations are shown as they were written.
    """
    width = _terminal_width() if width is None else width
    lines: list[str] = []
    doc = class_doc(composer.config_cls)
    if doc:
        lines += [*_comment(doc, "", width), ""]
    skip = None if own_settings else RESERVED_KEY
    _cfg_node(root, "", width, lines, _yaml(), skip=skip)
    return "\n".join(lines) + "\n"
