"""The parser for command-line overrides, written by hand.

The grammar is Hydra's, less what nematocyte leaves out (`design/cli.md`)::

    argument:  '+' NAME | '~' key | override     '+' NAME adds a primary config,
                                               '~' key leaves out a TypedDict key
    override:  key '=' value?
    key:       ID ('.' ID)*
    value:     element (',' element)*          more than one is a choice sweep
    element:   primitive | list | dict | function
    list:      '[' (element (',' element)*)? ']'
    dict:      '{' (dictKey ':' element (',' dictKey ':' element)*)? '}'
    function:  ID '(' ((ID '=')? element (',' (ID '=')? element)*)? ')'
    primitive: a quoted string, or unquoted text

Unquoted text is typed by its form: an int, a float, a bool, null, or else
a string.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
import math
import re
from typing import Any

from ._errors import ConfigError

__all__ = [
    "AddedConfig",
    "ChoiceSweep",
    "IntervalSweep",
    "Override",
    "RangeSweep",
    "Sweep",
    "format_value",
    "is_sweep",
    "last_wins",
    "parse_argument",
    "parse_override",
    "parse_value",
    "split_arguments",
]


@dataclass(frozen=True)
class ChoiceSweep:
    """`a,b` or `choice(a, b)`: one job per value."""

    values: tuple[Any, ...]
    tags: frozenset[str] = frozenset()

    def with_tags(self, tags: frozenset[str]) -> ChoiceSweep:
        return ChoiceSweep(self.values, self.tags | tags)


@dataclass(frozen=True)
class RangeSweep:
    """`range(start, stop, step)`: as Python's `range`, also for floats."""

    start: int | float
    stop: int | float
    step: int | float = 1
    tags: frozenset[str] = frozenset()

    def with_tags(self, tags: frozenset[str]) -> RangeSweep:
        return RangeSweep(self.start, self.stop, self.step, self.tags | tags)

    def values(self) -> list[int] | list[float]:
        if self.step == 0:
            raise ConfigError("The step of range() must not be zero.")
        if all(isinstance(x, int) for x in (self.start, self.stop, self.step)):
            return list(range(int(self.start), int(self.stop), int(self.step)))
        # Decimal arithmetic, so that range(0, 1, 0.1) has no rounding noise.
        start, stop, step = (
            Decimal(str(x)) for x in (self.start, self.stop, self.step)
        )
        count = max(0, math.ceil((stop - start) / step))
        return [float(start + i * step) for i in range(count)]


@dataclass(frozen=True)
class IntervalSweep:
    """`interval(start, end)`: a continuous range, for a sweeper that samples."""

    start: int | float
    end: int | float
    tags: frozenset[str] = frozenset()

    def with_tags(self, tags: frozenset[str]) -> IntervalSweep:
        return IntervalSweep(self.start, self.end, self.tags | tags)


type Sweep = ChoiceSweep | RangeSweep | IntervalSweep


@dataclass(frozen=True)
class Override:
    """A parsed `key=value` argument, or a `~key` argument."""

    key: tuple[str, ...]
    value: Any
    """A plain value, or a sweep; `None` for `~key`."""
    text: str = field(compare=False)
    """The argument as it was written."""
    unset: bool = False
    """Whether this is `~key`, which leaves an optional TypedDict key out."""

    @property
    def key_text(self) -> str:
        return ".".join(self.key)

    @property
    def is_sweep(self) -> bool:
        return is_sweep(self.value)

    @property
    def is_mapping(self) -> bool:
        """Whether the value is a dict, or a sweep with a dict among its values.

        For a subconfig or a group, a dict stands for its dotted keys, and is
        not replaced by another argument for the same key.
        """
        if isinstance(self.value, dict):
            return True
        return isinstance(self.value, ChoiceSweep) and any(
            isinstance(value, dict) for value in self.value.values
        )


@dataclass(frozen=True)
class AddedConfig:
    """A parsed `+NAME` argument: a primary config applied on top of the tree."""

    name: str
    """The name of the primary config, as `--config-name` takes it."""
    text: str = field(compare=False)
    """The argument as it was written."""


_KEY_SEGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INT_UNSIGNED = r"(?:0|[1-9](?:_?[0-9])*)"
_DIGITS = r"[0-9](?:_?[0-9])*"
_POINT_FLOAT = rf"(?:{_INT_UNSIGNED}\.(?:{_DIGITS})?|\.{_DIGITS})"
_EXPONENT_FLOAT = rf"(?:{_INT_UNSIGNED}|{_POINT_FLOAT})[eE][+-]?{_DIGITS}"
_INT = re.compile(rf"[+-]?{_INT_UNSIGNED}")
_FLOAT = re.compile(
    rf"[+-]?(?:{_EXPONENT_FLOAT}|{_POINT_FLOAT}|[Ii][Nn][Ff]|[Nn][Aa][Nn])"
)
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
# The characters that Hydra's grammar allows in unquoted text.
_UNQUOTED = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/\\+.$%*@?| \t"
)
_ESCAPABLE = frozenset("\\()[]{}:=, \t")
_WHITESPACE = " \t"


class _ParseError(Exception):
    def __init__(self, message: str, pos: int) -> None:
        super().__init__(message)
        self.message = message
        self.pos = pos


class _Parser:
    def __init__(self, text: str, start: int = 0) -> None:
        self.text = text
        self.pos = start

    # -- helpers --------------------------------------------------------

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _skip_ws(self) -> None:
        while self._peek() in _WHITESPACE and self._peek():
            self.pos += 1

    def _at_end(self) -> bool:
        return self.pos >= len(self.text)

    def _error(self, message: str, pos: int | None = None) -> _ParseError:
        return _ParseError(message, self.pos if pos is None else pos)

    # -- grammar --------------------------------------------------------

    def value(self) -> Any:
        """A whole value: one element, or several separated by commas."""
        self._skip_ws()
        if self._at_end():
            return ""
        elements = [self.element(stop=",")]
        while self._peek() == ",":
            self.pos += 1
            elements.append(self.element(stop=","))
        if not self._at_end():
            raise self._unexpected()
        if len(elements) == 1:
            return elements[0]
        for element in elements:
            if isinstance(element, (ChoiceSweep, RangeSweep, IntervalSweep)):
                raise self._error(
                    "A sweep function cannot be an item of a comma-separated sweep.",
                    0,
                )
        return ChoiceSweep(tuple(elements))

    def _unexpected(self) -> _ParseError:
        char = self._peek()
        if char in "'\"":
            return self._error(
                "A quote can only be at the start of a value. Quote the whole "
                "value, or escape the quote with a backslash."
            )
        if char == "=":
            return self._error("A second '=' has to be quoted or escaped as '\\='.")
        if char == "~":
            return self._error(
                "'~' has to be quoted; the shell does not expand it inside quotes."
            )
        if not char.isascii():
            return self._error(
                f"The non-ASCII character {char!r} is only allowed inside quotes."
            )
        return self._error(f"Unexpected character {char!r}; quote the value to use it.")

    def element(self, *, stop: str) -> Any:
        """One element. It ends before a character of `stop`, or at the end."""
        self._skip_ws()
        char = self._peek()
        if char == "[":
            result = self._list()
        elif char == "{":
            result = self._dict()
        elif char in ("'", '"'):
            result = self._quoted()
        else:
            match = _ID.match(self.text, self.pos)
            if match is not None:
                after = match.end()
                while after < len(self.text) and self.text[after] in _WHITESPACE:
                    after += 1
                if after < len(self.text) and self.text[after] == "(":
                    self.pos = after + 1
                    result = self._function(match.group(), match.start())
                    self._skip_ws()
                    self._expect_end_of_element(stop)
                    return result
            return self._unquoted(stop=stop, in_key=False)
        self._skip_ws()
        self._expect_end_of_element(stop)
        return result

    def _expect_end_of_element(self, stop: str) -> None:
        if not self._at_end() and self._peek() not in stop:
            raise self._unexpected()

    def _list(self) -> list[Any]:
        self.pos += 1  # [
        items: list[Any] = []
        self._skip_ws()
        if self._peek() == "]":
            self.pos += 1
            return items
        while True:
            items.append(self._plain_element(stop=",]"))
            char = self._peek()
            self.pos += 1
            if char == "]":
                return items
            if char != ",":
                raise self._error("A list is not closed; ']' is missing.")

    def _dict(self) -> dict[Any, Any]:
        self.pos += 1  # {
        result: dict[Any, Any] = {}
        self._skip_ws()
        if self._peek() == "}":
            self.pos += 1
            return result
        while True:
            self._skip_ws()
            key_start = self.pos
            if self._peek() in ("'", '"'):
                raise self._error("A key of a dict cannot be quoted.")
            key = self._unquoted(stop=":", in_key=True)
            if self._peek() != ":":
                raise self._error(
                    "A key of a dict has to be followed by ':'.", key_start
                )
            self.pos += 1
            if key in result:
                raise self._error(f"The key {key!r} is given twice.", key_start)
            result[key] = self._plain_element(stop=",}")
            char = self._peek()
            self.pos += 1
            if char == "}":
                return result
            if char != ",":
                raise self._error("A dict is not closed; '}' is missing.")

    def _plain_element(self, *, stop: str) -> Any:
        start = self.pos
        result = self.element(stop=stop)
        if isinstance(result, (ChoiceSweep, RangeSweep, IntervalSweep)):
            raise self._error("A sweep cannot be inside a list or a dict.", start)
        return result

    def _quoted(self) -> str:
        quote = self._peek()
        start = self.pos
        self.pos += 1
        chars: list[str] = []
        while True:
            char = self._peek()
            if not char:
                raise self._error(f"The quote {quote} is not closed.", start)
            self.pos += 1
            if char == "\\" and self._peek() in (quote, "\\"):
                chars.append(self._peek())
                self.pos += 1
            elif char == quote:
                return "".join(chars)
            else:
                chars.append(char)

    def _unquoted(self, *, stop: str, in_key: bool) -> Any:
        """Unquoted text, typed by its form."""
        start = self.pos
        chars: list[str] = []
        literal = False  # an escape or an interpolation makes it a string
        while not self._at_end():
            char = self._peek()
            if char in stop:
                break
            if char == "\\" and self.pos + 1 < len(self.text):
                following = self.text[self.pos + 1]
                if following in _ESCAPABLE:
                    chars.append(following)
                    literal = True
                    self.pos += 2
                    continue
            if self.text.startswith("${", self.pos) and not in_key:
                end = self.text.find("}", self.pos)
                if end == -1:
                    raise self._error("The interpolation '${' is not closed.")
                chars.append(self.text[self.pos : end + 1])
                literal = True
                self.pos = end + 1
                continue
            if char == ":" and not in_key:
                chars.append(char)
                self.pos += 1
                continue
            if char not in _UNQUOTED:
                raise self._unexpected()
            chars.append(char)
            self.pos += 1
        text = "".join(chars)
        if not literal:
            text = text.strip(_WHITESPACE)
            if not text:
                raise self._error("A value is missing.", start)
            return _typed(text)
        # Only whitespace that was not escaped is trimmed. An escaped space is
        # rare enough that the difference is ignored at the right-hand end.
        return text.lstrip(_WHITESPACE) if text.strip(_WHITESPACE) else text

    def _function(self, name: str, start: int) -> Any:
        function = _FUNCTIONS.get(name)
        if function is None:
            raise self._error(
                f"Unknown function {name!r}. The functions are: "
                f"{', '.join(_FUNCTIONS)}. To pass the text as a string, quote it.",
                start,
            )
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        self._skip_ws()
        if self._peek() == ")":
            self.pos += 1
        else:
            while True:
                self._skip_ws()
                arg_name = None
                match = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)[ \t]*=").match(
                    self.text, self.pos
                )
                if match is not None:
                    arg_name = match.group(1)
                    self.pos = match.end()
                value = self.element(stop=",)")
                if arg_name is None:
                    if kwargs:
                        raise self._error(
                            "A positional argument cannot follow a named one."
                        )
                    args.append(value)
                elif arg_name in kwargs:
                    raise self._error(f"The argument {arg_name!r} is given twice.")
                else:
                    kwargs[arg_name] = value
                char = self._peek()
                self.pos += 1
                if char == ")":
                    break
                if char != ",":
                    raise self._error(f"The call of {name}() is not closed.", start)
        try:
            return function(*args, **kwargs)
        except TypeError as exc:
            raise self._error(f"Wrong arguments for {name}(): {exc}", start) from exc
        except ConfigError as exc:
            raise self._error(str(exc), start) from exc


def _typed(text: str) -> Any:
    if _INT.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    return text


def is_sweep(value: Any) -> bool:
    """Whether `value` is a sweep, and not a plain value."""
    return isinstance(value, (ChoiceSweep, RangeSweep, IntervalSweep))


def _number(value: Any, function: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{function}() takes numbers, not {value!r}.")
    return value


def _choice(*values: Any) -> ChoiceSweep:
    if not values:
        raise ConfigError("choice() needs at least one value.")
    if any(is_sweep(value) for value in values):
        raise ConfigError("A sweep cannot be an item of choice().")
    return ChoiceSweep(tuple(values))


def _range(start: Any, stop: Any = None, step: Any = 1) -> RangeSweep:
    if stop is None:
        start, stop = 0, start
    sweep = RangeSweep(
        _number(start, "range"), _number(stop, "range"), _number(step, "range")
    )
    if sweep.step == 0:
        raise ConfigError("The step of range() must not be zero.")
    return sweep


def _interval(start: Any, end: Any) -> IntervalSweep:
    return IntervalSweep(_number(start, "interval"), _number(end, "interval"))


def _tag(*args: Any) -> Sweep:
    sweep = args[-1] if args else None
    if not isinstance(sweep, (ChoiceSweep, RangeSweep, IntervalSweep)):
        raise ConfigError(
            "The last argument of tag() has to be a sweep, as in "
            "tag(log, interval(1e-5, 1e-3))."
        )
    tags = args[:-1]
    if not all(isinstance(tag, str) for tag in tags):
        raise ConfigError("The tags of tag() have to be names, such as 'log'.")
    return sweep.with_tags(frozenset(tags))


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "choice": _choice,
    "range": _range,
    "interval": _interval,
    "tag": _tag,
}


def _fail(text: str, exc: _ParseError) -> ConfigError:
    pointer = " " * exc.pos + "^"
    return ConfigError(
        f"Cannot parse {text!r}: {exc.message}\n    {text}\n    {pointer}"
    )


def parse_value(text: str) -> Any:
    """Parse the value part of an override: a plain value or a sweep."""
    try:
        return _Parser(text).value()
    except _ParseError as exc:
        raise _fail(text, exc) from None


def parse_override(text: str) -> Override:
    """Parse one `key=value` argument."""
    key_text, equals, _ = text.partition("=")
    if not equals:
        if text.startswith("~"):
            return _parse_unset(text)
        raise ConfigError(f"Cannot parse {text!r}: an override has the form key=value.")
    key_text = key_text.strip(_WHITESPACE)
    problem = None
    if key_text.startswith("++"):
        problem = (
            "the '++' prefix is not supported, and not needed: a key is added "
            "or changed with a plain key=value"
        )
    elif key_text.startswith("+"):
        bare = key_text[1:]
        value_text = text.partition("=")[2].strip(_WHITESPACE)
        name = f"{bare.replace('.', '/')}/{value_text}"
        problem = (
            f"'+' takes the name of a primary config, without '=': +{name} "
            f"applies the primary config {name!r} on top of the one from "
            f"--config-name. To set or select a key, no '+' is needed: "
            f"{bare}={value_text}"
        )
    elif key_text.startswith("~"):
        bare = key_text[1:].strip(_WHITESPACE)
        value_text = text.partition("=")[2].strip(_WHITESPACE)
        problem = (
            f"'~' takes only a key, without '=': ~{bare} leaves the key out. "
            f"To set it, no '~' is needed: {bare}={value_text}. Deleting a "
            "key only if it has a given value is not supported"
        )
    else:
        problem = _key_problem(key_text)
    if problem is not None:
        raise ConfigError(f"Cannot parse {text!r}: {problem}.")
    try:
        value = _Parser(text, start=len(text.partition("=")[0]) + 1).value()
    except _ParseError as exc:
        raise _fail(text, exc) from None
    return Override(tuple(key_text.split(".")), value, text)


def _key_problem(key_text: str) -> str | None:
    """What is wrong with the key of an override, or `None`."""
    if "@" in key_text:
        return "'key@package' is not supported"
    if "/" in key_text:
        dotted = key_text.replace("/", ".")
        return f"a key is written with dots at every depth: {dotted!r}"
    segments = key_text.split(".")
    if not all(_KEY_SEGMENT.fullmatch(segment) for segment in segments):
        return (
            "the key has to be a name, or several names joined by dots"
            if key_text
            else "the key is missing"
        )
    return None


def _parse_unset(text: str) -> Override:
    """Parse one `~key` argument, which leaves an optional TypedDict key out.

    `~key=value` goes to `parse_override`, which explains the error.
    """
    key_text = text.removeprefix("~").strip(_WHITESPACE)
    if not key_text or key_text.startswith(("~", "+")):
        problem = "'~' is followed by a key, as in ~wandb.name"
    else:
        problem = _key_problem(key_text)
    if problem is not None:
        raise ConfigError(f"Cannot parse {text!r}: {problem}.")
    return Override(tuple(key_text.split(".")), None, text, unset=True)


def parse_argument(text: str) -> Override | AddedConfig:
    """Parse one command-line argument: `+NAME` or `key=value`."""
    if text.startswith("+") and "=" not in text:
        name = text[1:].strip(_WHITESPACE)
        if name.startswith("+"):
            raise ConfigError(
                f"Cannot parse {text!r}: '+' is followed by the name of a "
                "primary config; '++' is not supported."
            )
        if not name:
            raise ConfigError(
                f"Cannot parse {text!r}: '+' is followed by the name of a "
                "primary config, as in +presets/debug."
            )
        return AddedConfig(name, text)
    return parse_override(text)


def split_arguments(
    texts: Sequence[str],
) -> tuple[list[AddedConfig], list[Override]]:
    """Parse the arguments into the added configs and the overrides.

    The added configs keep their order, which is the order in which they are
    applied. A config or a key that is given more than once keeps only its
    last occurrence.
    """
    added: list[AddedConfig] = []
    overrides: list[Override] = []
    for text in texts:
        argument = parse_argument(text)
        if isinstance(argument, AddedConfig):
            added = [a for a in added if a != argument]
            added.append(argument)
        else:
            overrides.append(argument)
    return added, last_wins(overrides)


def last_wins(overrides: Sequence[Override]) -> list[Override]:
    """Drop every override whose key is given again later.

    The overrides that are left keep their order, and a repeated key takes
    the place of its last occurrence. A dict is never dropped, and does not
    drop another override: for a subconfig or a group it stands for its
    dotted keys, which are applied in order with the values.
    """
    last = {
        override.key: index
        for index, override in enumerate(overrides)
        if not override.is_mapping
    }
    return [
        o for index, o in enumerate(overrides) if o.is_mapping or last[o.key] == index
    ]


def iter_sweep(sweep: Sweep) -> Iterator[Any]:
    """The values of a discrete sweep."""
    if isinstance(sweep, ChoiceSweep):
        yield from sweep.values
    elif isinstance(sweep, RangeSweep):
        yield from sweep.values()
    else:
        raise ConfigError("An interval() has no discrete values.")


# -- writing values back in the grammar ---------------------------------

_SAFE_STRING = re.compile(r"[A-Za-z0-9_\-/\\+.$%*@?|: ]+")


def _format_string(value: str) -> str:
    if (
        _SAFE_STRING.fullmatch(value)
        and value == value.strip()
        and isinstance(_typed(value), str)
    ):
        return value
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def format_value(value: Any) -> str:
    """Write a value the way it is typed on the command line."""
    if value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float)):
        text = repr(value)
    elif isinstance(value, dict):
        items = (f"{format_value(k)}:{format_value(v)}" for k, v in value.items())
        text = "{" + ",".join(items) + "}"
    elif isinstance(value, (list, tuple)):
        text = "[" + ",".join(format_value(item) for item in value) + "]"
    else:
        text = _format_string(str(value))
    return text
