from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlipaudit.benchmark import Benchmark


@pytest.fixture(scope="session")
def get_data_input_dir() -> Path:
    """Fixture to provide a data input directory with test data.

    Returns:
        The path to the test data directory.
    """
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_force_field() -> MagicMock:
    """Provides a mock ForceField object.

    This is a "dummy" object that can be passed to the Benchmark constructor,
    satisfying the type requirement without loading a real ML model.
    It will accept any method calls without erroring, which is perfect since
    the functions that would use it are also going to be mocked.

    Returns:
        A mock force field object.
    """
    return MagicMock()


@pytest.fixture
def mocked_benchmark_init(mocker):
    """A reusable fixture that mocks the __init__ side-effects of the base Benchmark.
    Currently, this just prevents the data download.
    """
    mocker.patch.object(Benchmark, "_download_data")
