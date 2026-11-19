"""The exceptions and warnings that nematocyte raises."""

__all__ = ["ConfigError", "DroppedValueWarning", "SchemaError"]


class ConfigError(Exception):
    """Something is wrong with a config file, an override or a config value."""


class SchemaError(ConfigError):
    """A config dataclass is declared in a way that nematocyte does not support."""


class DroppedValueWarning(UserWarning):
    """A selection dropped a value that a lower layer had set under its node."""
