"""Check that all YAML config files in `configs/` are compatible with the config classes.

The YAML files are not checked by the type checker, so if, for example, a field in one
of the dataclasses is renamed, a YAML file that still uses the old name would only fail
once someone tries to use it. This test catches that without running any experiments:
every config file is loaded in the same way as when the program runs, and the `Config`
object is built from the result.

Usage:
    pytest
"""

from pathlib import Path

import nematocyte
import pytest

from src.run import Config

CONFIG_DIR = Path(__file__).parent.parent / "configs"


# One test case per config file, so that every broken file is reported separately.
# What kind of config a file is follows from where it is: `configs/dm/` holds configs
# for the `dm` field of `Config`, and a directory that matches no field, such as
# `configs/experiments/`, holds primary configs.
@pytest.mark.parametrize(
    "path",
    sorted(path.relative_to(CONFIG_DIR) for path in CONFIG_DIR.rglob("*.y*ml")),
    ids=Path.as_posix,
)
def test_config(path: Path) -> None:
    result = nematocyte.validate_config(path, Config, config_dir=CONFIG_DIR)
    if result.error is not None:
        raise result.error
