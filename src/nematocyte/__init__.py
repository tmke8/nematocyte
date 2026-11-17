"""nematocyte: configuration for Python programs, as a tree of dataclasses."""

from ._api import EntryPoint as EntryPoint
from ._api import compose as compose
from ._api import main as main
from ._errors import ConfigError as ConfigError
from ._errors import SchemaError as SchemaError
from ._plain import to_dict as to_dict
from ._run import output_dir as output_dir
from ._schema import Option as Option
from ._schema import option as option
from ._validation import ValidationResult as ValidationResult
from ._validation import validate_config as validate_config
from ._validation import validate_configs as validate_configs

__all__ = [
    "ConfigError",
    "EntryPoint",
    "Option",
    "SchemaError",
    "ValidationResult",
    "compose",
    "main",
    "option",
    "output_dir",
    "to_dict",
    "validate_config",
    "validate_configs",
]
