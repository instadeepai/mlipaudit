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

import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from mlip.simulation import SimulationState

# Import the base class as well to help with mocking
from mlipaudit.benchmarks import (
    BondLengthDistributionBenchmark,
    BondLengthDistributionModelOutput,
    BondLengthDistributionResult,
)
from mlipaudit.benchmarks.bond_length_distribution.bond_length_distribution import (
    MoleculeModelOutput,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def bond_length_distribution_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> BondLengthDistributionBenchmark:
    """Assembles a fully configured and isolated
    BondLengthDistributionBenchmark instance.

    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized BondLengthDistributionBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return BondLengthDistributionBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize(
    "bond_length_distribution_benchmark", [True, False], indirect=True
)
def test_full_run_with_mocked_engine(
    bond_length_distribution_benchmark, mock_jaxmd_simulation_engine
):
    """Integration test testing a full run of the benchmark."""
    benchmark = bond_length_distribution_benchmark
    mock_engine = mock_jaxmd_simulation_engine()
    with patch(
        "mlipaudit.utils.simulation.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        benchmark.run_model()

        # Assert that the engine was initialized and run for each molecule
        num_molecules = len(benchmark._bond_length_distribution_data)
        assert mock_engine_class.call_count == num_molecules
        assert mock_engine.run.call_count == num_molecules

        assert isinstance(benchmark.model_output, BondLengthDistributionModelOutput)

        benchmark.model_output = BondLengthDistributionModelOutput(
            molecules=[
                MoleculeModelOutput(
                    molecule_name="carbon-carbon (single)",
                    simulation_state=SimulationState(
                        positions=np.ones((10, 8, 3)), temperature=np.ones(10)
                    ),
                ),
                MoleculeModelOutput(
                    molecule_name="carbon-oxygen (double)",
                    simulation_state=SimulationState(
                        positions=np.ones((10, 13, 3)), temperature=np.ones(10)
                    ),
                ),
            ],
        )

        result = benchmark.analyze()
        assert len(result.molecules) == num_molecules
        assert type(result) is BondLengthDistributionResult


def test_analyze_raises_error_if_run_first(bond_length_distribution_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        bond_length_distribution_benchmark.analyze()


def test_analyze(bond_length_distribution_benchmark):
    """Check the analysis method."""
    benchmark = bond_length_distribution_benchmark
    num_frames = 10

    cc_single_coordinates = np.array([
        [-0.0187, 1.5256, 0.0104],  # Bond of interest
        [0.0021, -0.0039, 0.002],  # Bond of interest
        [0.9949, 1.9397, 0.0029],
        [-0.5421, 1.9236, -0.8651],
        [-0.5252, 1.9142, 0.9],
        [0.5255, -0.4019, 0.8775],
        [-1.0115, -0.418, 0.0095],
        [0.5086, -0.3925, -0.8876],
    ])
    cc_single_stationary_trajectory = np.tile(
        cc_single_coordinates,
        reps=(num_frames, 1, 1),  # Create 10 duplicate frames
    )

    # Modifying other atoms shouldn't affect result
    cc_single_stationary_trajectory[-1][5][0] += 0.01

    co_double_coordinates = np.array([
        [0.073, 1.3884, 0.0644],
        [0.2105, -0.0374, 0.0461],
        [0.501, -0.5722, -1.1598],  # Bond of interest
        [0.6222, -2.0738, -1.0603],  # Bond of interest
        [0.6392, 0.0822, -2.1616],
        [1.0003, 1.8706, -0.2553],
        [-0.7334, 1.7078, -0.6008],
        [-0.1575, 1.6515, 1.0969],
        [-0.3133, -2.502, -0.6894],
        [0.8565, -2.4843, -2.0416],
        [1.4065, -2.3406, -0.3463],
    ])
    co_double_trajectory = np.tile(co_double_coordinates, reps=(num_frames, 1, 1))

    # Scale the distances between atoms of interest
    for frame in range(1, num_frames):
        co_double_trajectory[frame, 2, :] = co_double_trajectory[frame - 1, 2, :] + (
            0.001 * co_double_coordinates[2, :]
        )
        co_double_trajectory[frame, 3, :] = co_double_trajectory[frame - 1, 3, :] + (
            0.001 * co_double_coordinates[3, :]
        )

    unstable_trajectory = co_double_trajectory.copy()
    unstable_trajectory[-1, -1, -1] += 100.0

    benchmark.model_output = BondLengthDistributionModelOutput(
        molecules=[
            MoleculeModelOutput(
                molecule_name="carbon-carbon (single)",
                simulation_state=SimulationState(
                    positions=cc_single_stationary_trajectory, temperature=np.ones(10)
                ),
            ),
            MoleculeModelOutput(
                molecule_name="carbon-oxygen (double)",
                simulation_state=SimulationState(
                    positions=co_double_trajectory, temperature=np.ones(10)
                ),
            ),
            MoleculeModelOutput(
                molecule_name="failed molecule",
                simulation_state=SimulationState(
                    positions=unstable_trajectory, temperature=np.ones(10)
                ),
            ),
        ],
    )

    result = benchmark.analyze()

    assert result.molecules[0].molecule_name == "carbon-carbon (single)"
    assert result.molecules[1].molecule_name == "carbon-oxygen (double)"

    assert np.all(
        np.array(result.molecules[0].deviation_trajectory) == pytest.approx(0.0)
    )
    assert result.molecules[0].avg_deviation == pytest.approx(0.0)

    deviation_trajectory = result.molecules[1].deviation_trajectory
    deviation_increases = [
        deviation_trajectory[frame] - deviation_trajectory[frame - 1]
        for frame in range(1, num_frames)
    ]

    assert np.all(
        np.array(deviation_increases) == pytest.approx(deviation_increases[0])
    )

    assert result.molecules[1].avg_deviation > 1e-4
