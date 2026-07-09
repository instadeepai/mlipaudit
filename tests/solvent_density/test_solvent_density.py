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
    SolventDensityBenchmark,
    SolventDensityModelOutput,
    SolventDensityResult,
    SolventRadialDistributionBenchmark,
    SolventRadialDistributionModelOutput,
)
from mlipaudit.benchmarks_cli import _transfer_model_output
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"
# The solvent density benchmark reuses the solvent RDF benchmark's input data.
SOLVENT_DATA_DIR = INPUT_DATA_DIR / "solvent_radial_distribution"


@pytest.fixture
def solvent_density_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> SolventDensityBenchmark:
    """Assembles a fully configured and isolated SolventDensityBenchmark instance.

    Returns:
        An initialized SolventDensityBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return SolventDensityBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


def _mock_ccl4_simulation_state() -> SimulationState:
    """Build a stationary CCl4 trajectory from the equilibrated structure."""
    atoms = ase_read(SOLVENT_DATA_DIR / "CCl4_eq.pdb")
    num_frames = 2
    positions = np.tile(np.array(atoms.positions), reps=(num_frames, 1, 1))
    cells = np.tile(np.array(atoms.get_cell()), reps=(num_frames, 1, 1))
    return SimulationState(
        positions=positions, temperature=np.ones(num_frames), cell=cells
    )


def test_data_name_points_to_shared_data(solvent_density_benchmark):
    """The density benchmark should read from the RDF benchmark's data directory."""
    assert solvent_density_benchmark.data_name == "solvent_radial_distribution"
    assert solvent_density_benchmark.data_dir == SOLVENT_DATA_DIR


@pytest.mark.parametrize("solvent_density_benchmark", [True], indirect=True)
def test_full_run_with_mocked_engine(
    solvent_density_benchmark, mock_jaxmd_simulation_engine
):
    """Integration test testing a full run of the benchmark."""
    benchmark = solvent_density_benchmark
    mock_engine = mock_jaxmd_simulation_engine()
    with patch(
        "mlipaudit.utils.simulation.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        benchmark.run_model()

        assert mock_engine_class.call_count == 1
        assert isinstance(benchmark.model_output, SolventDensityModelOutput)

        benchmark.model_output = SolventDensityModelOutput(
            structure_names=["CCl4"],
            simulation_states=[_mock_ccl4_simulation_state()],
        )

        result = benchmark.analyze()
        assert type(result) is SolventDensityResult

        assert len(result.structures) == 1
        assert result.structures[0].structure_name == "CCl4"

        # Target density = 1.594, initial density = 1.368
        assert 1.3 < result.structures[0].average_density < 1.6
        assert result.structures[0].density_deviation < 0.3
        assert 0.0 <= result.score <= 1.0


def test_reuses_solvent_rdf_simulation_output():
    """The density benchmark must be able to reuse the RDF benchmark's model output.

    This mirrors the CLI reuse path in `benchmarks_cli.run_benchmarks`, where a
    cached `ModelOutput` is transferred between benchmarks that share a
    `reusable_output_id`.
    """
    assert (
        SolventDensityBenchmark.reusable_output_id
        == SolventRadialDistributionBenchmark.reusable_output_id
    )

    rdf_output = SolventRadialDistributionModelOutput(
        structure_names=["CCl4"],
        simulation_states=[_mock_ccl4_simulation_state()],
    )
    transferred = _transfer_model_output(rdf_output, SolventDensityModelOutput)

    assert isinstance(transferred, SolventDensityModelOutput)
    assert transferred.structure_names == ["CCl4"]


def test_analyze_raises_error_if_run_first(solvent_density_benchmark):
    """Verifies the RuntimeError is raised when analyze is called first."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        solvent_density_benchmark.analyze()
