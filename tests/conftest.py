from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest
from ase import Atoms
from mlip.typing import Prediction

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
    """A reusable fixture that mocks the __init__ side effects of the base Benchmark.
    Currently, this just prevents the data download.
    """
    mocker.patch.object(Benchmark, "_download_data")


@pytest.fixture
def mocked_batched_inference() -> Callable:
    """A reusable fixture that provides a mocked version of a batched inference
    function from mlip.

    Returns:
        A function that can be set as a mock for batched inference.
    """

    def _batched_inference(atoms_list: list[Atoms], force_field, **kwargs):
        """Mock running batched inference on a list of conformers.

        Returns:
            A list of random energy predictions.
        """
        return [Prediction(energy=0.0) for _ in range(len(atoms_list))]

    return _batched_inference
