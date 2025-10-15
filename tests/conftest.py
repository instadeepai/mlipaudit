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
import pydantic
import pytest
from ase import Atoms
from ase.symbols import symbols2numbers
from mlip.models import ForceField
from mlip.simulation import SimulationState
from mlip.simulation.ase import ASESimulationEngine
from mlip.simulation.jax_md import JaxMDSimulationEngine
from mlip.typing import Prediction
from pydantic import ConfigDict

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.benchmarks.nudged_elastic_band.engine import NEBSimulationEngine


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
    magic_mock = MagicMock(spec=ForceField)
    allowed_element_types = {
        "Xe",
        "N",
        "I",
        "Ar",
        "H",
        "Se",
        "O",
        "S",
        "As",
        "Ne",
        "Br",
        "He",
        "Kr",
        "P",
        "C",
        "Cl",
        "F",
        "B",
    }
    magic_mock.allowed_atomic_numbers = symbols2numbers(allowed_element_types)
    return magic_mock


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


@pytest.fixture
def mock_ase_simulation_engine() -> Callable[[SimulationState], MagicMock]:
    """Provides a mock ASESimulationEngine object with a default Simulation
    State. A custom simulation state can be provided when creating the engine.
    The engine will always return the same simulation state.

    Returns:
        A callable taking as optional argument a simulation state and returning
        an engine that always returns the same simulation state.
    """

    def _factory(simulation_state: SimulationState | None = None):
        mock_engine = create_autospec(ASESimulationEngine, instance=True)
        if simulation_state:
            state = simulation_state
        else:
            state = SimulationState(
                atomic_numbers=np.array([0, 1]),
                positions=np.random.rand(10, 2, 3),
                forces=np.random.rand(10, 2, 3),
                temperature=np.random.rand(10),
            )
        atoms = Atoms(state.atomic_numbers, state.positions[0])
        mock_engine.atoms = atoms
        mock_engine.configure_mock(state=state)
        return mock_engine

    return _factory


@pytest.fixture
def mock_neb_simulation_engine() -> Callable[[SimulationState], MagicMock]:
    """Provides a mock NEBSimulationEngine object with a default Simulation
    State. A custom simulation state can be provided when creating the engine.
    The engine will always return the same simulation state.

    Returns:
        A callable taking as optional argument a simulation state and returning
        an engine that always returns the same simulation state.
    """

    def _factory(simulation_state: SimulationState | None = None):
        mock_engine = create_autospec(NEBSimulationEngine, instance=True)
        if simulation_state:
            state = simulation_state
        else:
            state = SimulationState(
                atomic_numbers=np.array([0, 1]),
                positions=np.random.rand(10, 2, 3),
                forces=np.random.rand(10, 2, 3),
                temperature=np.random.rand(10),
            )
        atoms = Atoms(state.atomic_numbers, state.positions[0])
        mock_engine.atoms = atoms
        mock_engine.configure_mock(state=state)
        return mock_engine

    return _factory


class DummyBenchmarkResultLarge(BenchmarkResult):
    """A dummy benchmark result with 5 entries."""

    a: int
    b: str
    c: list[float]
    d: list[tuple[float, float]]


class DummyBenchmarkResultSmallSubclass(BenchmarkResult):
    """A dummy benchmark result subclass."""

    value: float


class DummyBenchmarkResultSmall(BenchmarkResult):
    """A dummy benchmark result with one entry."""

    values: list[DummyBenchmarkResultSmallSubclass]


class DummySubclassModelOutput(pydantic.BaseModel):
    """A dummy model output subclass used in the other model output."""

    name: str
    state: SimulationState

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DummyModelOutput(ModelOutput):
    """A dummy model output class."""

    structure_names: list[str]
    simulation_states: list[SimulationState]
    subclasses: list[DummySubclassModelOutput]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DummyBenchmark1(Benchmark):
    """Dummy benchmark 1."""

    name = "benchmark_1"
    result_class = DummyBenchmarkResultLarge
    model_output_class = DummyModelOutput

    required_elements = {"H", "O"}

    def run_model(self) -> None:
        """No need to implement this for this test."""
        pass

    def analyze(self) -> DummyBenchmarkResultLarge:  # type:ignore
        """No need to implement this for this test."""
        pass


class DummyBenchmark2(Benchmark):
    """Dummy benchmark 2."""

    name = "benchmark_2"
    result_class = DummyBenchmarkResultSmall
    model_output_class = DummyModelOutput

    required_elements = {"H", "O"}

    def run_model(self) -> None:
        """No need to implement this for this test."""
        pass

    def analyze(self) -> list[DummyBenchmarkResultSmall]:  # type:ignore
        """No need to implement this for this test."""
        pass


@pytest.fixture
def dummy_benchmark_results_model_1():
    """Dummy benchmark results."""
    return {
        "benchmark_1": DummyBenchmarkResultLarge(
            a=7, b="test", c=[3.4, 5.6, 7.8], d=[(1.0, 1.1), (1.2, 1.3)]
        ),
        "benchmark_2": DummyBenchmarkResultSmall(
            values=[
                DummyBenchmarkResultSmallSubclass(value=0.1),
                DummyBenchmarkResultSmallSubclass(value=0.2),
            ]
        ),
    }


@pytest.fixture
def dummy_benchmark_results_model_2():
    """Dummy benchmark results."""
    return {
        "benchmark_1": DummyBenchmarkResultLarge(
            a=17, b="test", c=[13.4, 15.6, 17.8], d=[(11.0, 11.1), (11.2, 11.3)]
        ),
        "benchmark_2": DummyBenchmarkResultSmall(
            values=[
                DummyBenchmarkResultSmallSubclass(value=10.1),
                DummyBenchmarkResultSmallSubclass(value=10.2),
            ]
        ),
    }


@pytest.fixture
def dummy_model_output_class():
    """Dummy model output class."""
    return DummyModelOutput


@pytest.fixture
def dummy_subclass_model_output_class():
    """Dummy model output subclass."""
    return DummySubclassModelOutput


@pytest.fixture
def dummy_small_result_class():
    """Dummy model output class."""
    return DummyBenchmarkResultSmall


@pytest.fixture
def dummy_benchmark_1_class():
    """Dummy model class 1."""
    return DummyBenchmark1


@pytest.fixture
def dummy_benchmark_2_class():
    """Dummy model class 2."""
    return DummyBenchmark2


@pytest.fixture
def all_dummy_benchmark_classes():
    """Dummy model classes as list."""
    return [DummyBenchmark1, DummyBenchmark2]


@pytest.fixture
def dummy_benchmark1_instance() -> DummyBenchmark1:
    """Dummy model class 1 instance."""
    return DummyBenchmark1(mock_force_field)
