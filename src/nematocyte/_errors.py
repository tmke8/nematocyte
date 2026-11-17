"""The exceptions that nematocyte raises."""

__all__ = ["ConfigError", "SchemaError"]


class ConfigError(Exception):
    """Something is wrong with a config file, an override or a config value."""


class SchemaError(ConfigError):
    """A config dataclass is declared in a way that nematocyte does not support."""
