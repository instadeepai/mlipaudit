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
from mlipaudit.benchmarks.ring_planarity.ring_planarity import (
    MoleculeModelOutput,
    RingPlanarityBenchmark,
    RingPlanarityModelOutput,
    RingPlanarityResult,
    deviation_from_plane,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


def test_deviation_from_plane():
    """Test deviation from planar coordinates."""
    # Points lying on XY plane should have 0 deviation
    planar_coords = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [-0.5, 0.866, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.5, -0.866, 0.0],
        [0.5, -0.866, 0.0],
    ])
    rmsd = deviation_from_plane(planar_coords)

    assert rmsd == pytest.approx(0.0)

    # Test a line
    two_points = np.array([[1.0, 1.0, 1.0], [-4.0, -3.0, -2.0]])
    rmsd = deviation_from_plane(two_points)
    assert rmsd == pytest.approx(0.0)

    line_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])

    rmsd = deviation_from_plane(line_coords)

    assert rmsd == pytest.approx(0.0)

    # One point not on z=0
    non_planar_coords = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 2.0],
    ])

    rmsd = deviation_from_plane(non_planar_coords)

    assert rmsd > 0.0


@pytest.fixture
def ring_planarity_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> RingPlanarityBenchmark:
    """Assembles a fully configured and isolated RingPlanarityBenchmark instance.

    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized RingPlanarityBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return RingPlanarityBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize("ring_planarity_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_engine(
    ring_planarity_benchmark, mock_jaxmd_simulation_engine
):
    """Integration test testing a full run of the benchmark."""
    benchmark = ring_planarity_benchmark
    mock_engine = mock_jaxmd_simulation_engine()
    with patch(
        "mlipaudit.utils.simulation.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        benchmark.run_model()

        # Assert that the engine was initialized and run for each molecule
        num_molecules = len(benchmark._qm9_structures)
        assert mock_engine_class.call_count == num_molecules
        assert mock_engine.run.call_count == num_molecules

        assert isinstance(benchmark.model_output, RingPlanarityModelOutput)

        benchmark.model_output = RingPlanarityModelOutput(
            molecules=[
                MoleculeModelOutput(
                    molecule_name="benzene",
                    simulation_state=SimulationState(
                        positions=np.ones((10, 12, 3)), temperature=np.ones(10)
                    ),
                ),
                MoleculeModelOutput(
                    molecule_name="furan",
                    simulation_state=SimulationState(
                        positions=np.ones((10, 9, 3)), temperature=np.ones(10)
                    ),
                ),
            ],
        )

        result = benchmark.analyze()
        assert len(result.molecules) == num_molecules
        assert type(result) is RingPlanarityResult


def test_analyze_raises_error_if_run_first(ring_planarity_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        ring_planarity_benchmark.analyze()


def test_analyze(ring_planarity_benchmark):
    """Check the analysis method."""
    benchmark = ring_planarity_benchmark

    # Frame 1: A regular pentagon on the XY plane (z=0 for all atoms)
    planar_pentagon_frame = np.array([
        [0.0, 1.0, 0.0],
        [0.951, 0.309, 0.0],
        [0.588, -0.809, 0.0],
        [-0.588, -0.809, 0.0],
        [-0.951, 0.309, 0.0],
    ])

    # Frame 2: Take the planar frame and make it non-planar
    non_planar_pentagon_frame = planar_pentagon_frame.copy()
    non_planar_pentagon_frame[0, 2] = 1.5

    furan_positions = np.array([planar_pentagon_frame, non_planar_pentagon_frame])

    benchmark.model_output = RingPlanarityModelOutput(
        molecules=[
            MoleculeModelOutput(
                molecule_name="benzene",
                simulation_state=SimulationState(
                    positions=np.zeros((10, 10, 3)), temperature=np.ones(10)
                ),
            ),
            MoleculeModelOutput(
                molecule_name="furan",
                simulation_state=SimulationState(
                    positions=furan_positions, temperature=np.ones(10)
                ),
            ),
        ],
    )
    result = benchmark.analyze()

    assert result.molecules[0].avg_deviation == pytest.approx(0.0)

    assert result.molecules[1].deviation_trajectory[0] == pytest.approx(0.0)
    assert len(result.molecules[1].deviation_trajectory) == 2
    assert result.molecules[1].deviation_trajectory[1] > 1e-9
    assert result.molecules[1].avg_deviation > 1e-9
