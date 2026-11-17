from pathlib import Path

import pytest

from .helpers import CONFIG_FILES, write_files


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A config directory with the files of `CONFIG_FILES`."""
    return write_files(tmp_path / "configs", CONFIG_FILES)


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in an empty directory, so that `outputs/` lands there."""
    monkeypatch.chdir(tmp_path)
