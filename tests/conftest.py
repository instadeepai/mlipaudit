# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
from ase import Atoms
from mlip.simulation import SimulationState
from mlip.simulation.jax_md import JaxMDSimulationEngine
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


@pytest.fixture
def mock_jaxmd_simulation_engine() -> Callable[[SimulationState], MagicMock]:
    """Provides a mock JaxMDSimulationEngine object with a default Simulation
    State. A custom simulation state can be provided when creating the engine.
    The engine will always return the same simulation state.

    Returns:
        A callable taking as optional argument a simulation state and returning
        an engine that always returns the same simulation state.
    """

    def _factory(simulation_state: SimulationState | None = None):
        mock_engine = create_autospec(JaxMDSimulationEngine, instance=True)
        if simulation_state:
            state = simulation_state
        else:
            state = SimulationState(
                atomic_numbers=np.array([0, 1]),
                positions=np.random.rand(10, 2, 3),
                forces=np.random.rand(10, 2, 3),
                temperature=np.random.rand(10),
            )
        mock_engine.configure_mock(state=state)
        return mock_engine

    return _factory
