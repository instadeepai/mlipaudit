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
from ase.io import read as ase_read
from mlip.simulation import SimulationState

from mlipaudit.benchmarks import (
    StabilityBenchmark,
    StabilityModelOutput,
)
from mlipaudit.benchmarks.stability.stability import (
    detect_hydrogen_drift,
    find_first_drifting_frames,
)
from mlipaudit.run_mode import RunMode
from mlipaudit.utils import create_mdtraj_trajectory_from_simulation_state
from mlipaudit.utils.stability import is_frame_stable

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


def _generate_fake_model_output() -> StabilityModelOutput:
    def _gen_sim_output(num_atoms: int) -> SimulationState:
        return SimulationState(
            positions=np.ones((10, num_atoms, 3)), temperature=np.ones(10)
        )

    return StabilityModelOutput(
        structure_names=["Small_molecule_HCNO", "Small_molecule_Sulfur"],
        simulation_states=[_gen_sim_output(46), _gen_sim_output(36)],
    )


@pytest.fixture
def stability_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> StabilityBenchmark:
    """Assembles a fully configured and isolated
    `StabilityBenchmark` instance. This fixture is parameterized
    to handle the `run_mode` flag.

    Returns:
        An initialized `StabilityBenchmark` instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return StabilityBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


def test_is_stable():
    """Test basic stability check."""
    num_atoms = 10
    X = np.random.rand(num_atoms, 3)
    assert is_frame_stable(X, cutoff=4)

    X[0][1] += 20
    assert not is_frame_stable(X, cutoff=4)


def test_find_first_drifting_frames():
    """Test finding first drifting frames."""
    matrix = np.array([
        [False, True, False, True],
        [True, False, True, True],
        [True, False, False, True],
        [True, True, True, True],
        [True, True, False, True],
    ])
    assert np.all(find_first_drifting_frames(matrix) == np.array([1, 3, 5, 0]))

    matrix = np.array([[False, False, False, False], [False, False, False, False]])
    assert np.all(find_first_drifting_frames(matrix) == np.array([2, 2, 2, 2]))


def test_detect_hydrogen_drift():
    """Test detecting hydrogen drift."""
    chignolin_filename = INPUT_DATA_DIR / StabilityBenchmark.name / "138_1uao_chignolin"
    atoms = ase_read(chignolin_filename.with_suffix(".xyz"))
    symbols = atoms.get_chemical_symbols()
    assert symbols[-1] == "H"

    traj_positions = np.tile(atoms.get_positions(), (4, 1, 1))

    simulation_state = SimulationState(
        atomic_numbers=atoms.get_atomic_numbers(), positions=traj_positions
    )
    traj = create_mdtraj_trajectory_from_simulation_state(
        simulation_state=simulation_state,
        topology_path=chignolin_filename.with_suffix(".pdb"),
    )
    first_drifting_frame, first_drifting_hydrogen_index = detect_hydrogen_drift(traj)
    assert first_drifting_frame == -1
    assert first_drifting_hydrogen_index == -1

    # Modify position of H in only penultimate frame
    traj_positions[2][-1][0] += 100.0
    simulation_state = SimulationState(
        atomic_numbers=atoms.get_atomic_numbers(), positions=traj_positions
    )

    traj = create_mdtraj_trajectory_from_simulation_state(
        simulation_state=simulation_state,
        topology_path=chignolin_filename.with_suffix(".pdb"),
    )

    first_drifting_frame, first_drifting_hydrogen_index = detect_hydrogen_drift(traj)
    assert first_drifting_frame == -1
    assert first_drifting_hydrogen_index == -1

    # Make H drift
    traj_positions[3][-1][0] += 100.0
    simulation_state = SimulationState(
        atomic_numbers=atoms.get_atomic_numbers(), positions=traj_positions
    )

    traj = create_mdtraj_trajectory_from_simulation_state(
        simulation_state=simulation_state,
        topology_path=chignolin_filename.with_suffix(".pdb"),
    )

    first_drifting_frame, first_drifting_hydrogen_index = detect_hydrogen_drift(traj)

    assert first_drifting_frame == 2
    assert first_drifting_hydrogen_index == atoms.get_number_of_atoms() - 1


@pytest.mark.parametrize("stability_benchmark", [True], indirect=True)
def test_full_run_with_mocked_engine(stability_benchmark, mock_jaxmd_simulation_engine):
    """Integration test testing a full run of the benchmark."""
    benchmark = stability_benchmark
    mock_engine = mock_jaxmd_simulation_engine()
    with patch(
        "mlipaudit.utils.simulation.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        benchmark.run_model()

        assert mock_engine_class.call_count == 2
        assert isinstance(benchmark.model_output, StabilityModelOutput)

        benchmark.model_output = _generate_fake_model_output()
        result = benchmark.analyze()

        assert result.structure_results[0].num_frames == 10
        assert result.structure_results[0].exploded_frame == -1
        assert result.structure_results[0].drift_frame == -1
        assert result.structure_results[0].score == 1.0

        # Test H drift

        # Modify H in last frame
        benchmark.model_output.simulation_states[0].positions[-1][-1][0] += 200.0
        # Modify H in penultimate frame
        benchmark.model_output.simulation_states[0].positions[-2][-1][0] += 100.0

        result = benchmark.analyze()

        assert result.structure_results[0].exploded_frame == -1
        assert result.structure_results[0].drift_frame == 8
        assert result.structure_results[0].score == pytest.approx(0.5 + (8 / 10) / 2)


def test_analyze_raises_error_if_run_first(stability_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        stability_benchmark.analyze()
