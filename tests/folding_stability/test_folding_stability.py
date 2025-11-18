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

import ase
import numpy as np
import pytest
from ase.io import read as ase_read
from mlip.simulation import SimulationState

from mlipaudit.benchmarks.folding_stability.folding_stability import (
    FoldingStabilityBenchmark,
    FoldingStabilityModelOutput,
    FoldingStabilityMoleculeResult,
    FoldingStabilityResult,
)
from mlipaudit.benchmarks.folding_stability.helpers import (
    compute_radius_of_gyration_for_ase_atoms,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def folding_stability_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> FoldingStabilityBenchmark:
    """Assembles a fully configured and isolated FoldingStabilityBenchmark instance.

    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized FoldingStabilityBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return FoldingStabilityBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize("folding_stability_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_simulation_with_static_and_random_trajectory(
    folding_stability_benchmark,
    mock_jaxmd_simulation_engine,
):
    """Integration test using the modular fixture for fast dev run."""
    benchmark = folding_stability_benchmark

    atoms = ase_read(
        INPUT_DATA_DIR
        / "folding_stability"
        / "starting_structures"
        / "chignolin_1uao_xray.xyz"
    )

    num_steps = 10

    # Case 1: we set the trajectory to be just 10 identical structures
    if benchmark.run_mode == RunMode.DEV:
        traj = np.array([atoms.positions] * num_steps)
        forces = np.zeros(shape=traj.shape)
    # Case 2: we use random structures
    else:
        np.random.seed(42)
        traj = np.random.rand(num_steps, len(atoms), 3)
        forces = np.random.rand(num_steps, len(atoms), 3)

    mock_engine = mock_jaxmd_simulation_engine(
        SimulationState(
            atomic_numbers=atoms.numbers,
            positions=traj,
            forces=forces,
            temperature=np.zeros(num_steps),
        )
    )

    with patch(
        "mlipaudit.utils.simulation.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        if benchmark.run_mode == RunMode.DEV:  # Case 1
            benchmark.run_model()
        else:  # Case 2
            with pytest.raises(FileNotFoundError):
                benchmark.run_model()

        assert mock_engine_class.call_count == 1
        assert mock_engine.run.call_count == 1

    assert type(benchmark.model_output) is FoldingStabilityModelOutput
    assert benchmark.model_output.structure_names == ["chignolin_1uao_xray"]

    with patch(
        "mlipaudit.benchmarks.folding_stability.folding_stability."
        "FoldingStabilityBenchmark._assert_structure_names_in_model_output"
    ):
        result = benchmark.analyze()

    assert type(result) is FoldingStabilityResult
    assert type(result.molecules[0]) is FoldingStabilityMoleculeResult
    assert len(result.molecules) == 1

    # Case 1: values will all predictable as trajectory is static
    if benchmark.run_mode == RunMode.DEV:
        for i in range(num_steps):
            assert result.molecules[0].rmsd_trajectory[i] < 1e-6
            assert result.molecules[0].tm_score_trajectory[i] == pytest.approx(1.0)
            assert len(set(result.molecules[0].radius_of_gyration_deviation)) == 1
            assert len(set(result.molecules[0].match_secondary_structure)) == 1
            assert result.molecules[0].radius_of_gyration_fluctuation == 0.0
        assert result.avg_rmsd < 1e-6
        assert result.avg_tm_score == pytest.approx(1.0)
        assert result.avg_match == result.molecules[0].match_secondary_structure[0]
        assert result.max_abs_deviation_radius_of_gyration == 0.0
    # Case 2: values will fluctuate between frames
    else:
        for i in range(num_steps):
            assert result.molecules[0].rmsd_trajectory[i] > 1e-6
            assert 0.0 < result.molecules[0].tm_score_trajectory[i] < 1.0
            assert len(set(result.molecules[0].radius_of_gyration_deviation)) > 1
            assert len(set(result.molecules[0].match_secondary_structure)) > 1
            assert result.molecules[0].radius_of_gyration_fluctuation > 0.0
        assert result.avg_rmsd > 1e-6
        assert 0.0 < result.avg_tm_score < 1.0
        assert result.avg_rmsd == result.molecules[0].avg_rmsd
        assert result.avg_tm_score == result.molecules[0].avg_tm_score
        assert 0.0 < result.avg_match < 1.0
        assert 0.0 < result.max_abs_deviation_radius_of_gyration < 1.0


def test_analyze_raises_error_if_run_first(folding_stability_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        folding_stability_benchmark.analyze()


def test_compute_radius_of_gyration_for_ase_atoms():
    """Tests the radius of gyration function by inputting a system
    with a simple structure and center of mass of (0, 0, 0).
    """
    atom_numbers = [6, 8, 8]
    pos = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)]
    atoms = ase.Atoms(numbers=atom_numbers, positions=pos)
    rad_of_gyr = compute_radius_of_gyration_for_ase_atoms(atoms)

    # 128 is the sum of mass times squared distances, and 44 is the sum of masses.
    # 128, because: 2^2 * 16 + 2^2 * 16
    assert rad_of_gyr == pytest.approx(np.sqrt(128 / 44), abs=1e-3)
