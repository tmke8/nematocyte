"""nematocyte: configuration for Python programs, as a tree of dataclasses.

A node of the tree can also be a `TypedDict`.
"""

from ._api import EntryPoint as EntryPoint
from ._api import compose as compose
from ._api import from_run_config as from_run_config
from ._api import main as main
from ._errors import ConfigError as ConfigError
from ._errors import DroppedValueWarning as DroppedValueWarning
from ._errors import SchemaError as SchemaError
from ._plain import to_dict as to_dict
from ._run import configure_logging as configure_logging
from ._run import job_num as job_num
from ._run import log_level as log_level
from ._run import output_dir as output_dir
from ._run import run_config as run_config
from ._schema import Option as Option
from ._schema import option as option
from ._types import JSONValue as JSONValue
from ._validation import ValidationResult as ValidationResult
from ._validation import Validator as Validator
from ._validation import validate_configs as validate_configs

__all__ = [
    "ConfigError",
    "DroppedValueWarning",
    "EntryPoint",
    "JSONValue",
    "Option",
    "SchemaError",
    "ValidationResult",
    "Validator",
    "compose",
    "configure_logging",
    "from_run_config",
    "job_num",
    "log_level",
    "main",
    "option",
    "output_dir",
    "run_config",
    "to_dict",
    "validate_configs",
]
