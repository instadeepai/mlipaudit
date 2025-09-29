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
    SolventRadialDistributionBenchmark,
    SolventRadialDistributionModelOutput,
    SolventRadialDistributionResult,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def solvent_radial_distribution_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> SolventRadialDistributionBenchmark:
    """Assembles a fully configured and isolated
    SolventRadialDistributionBenchmark instance.
    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized SolventRadialDistributionBenchmark  instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return SolventRadialDistributionBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize("solvent_radial_distribution_benchmark", [True], indirect=True)
def test_full_run_with_mocked_engine(
    solvent_radial_distribution_benchmark, mock_jaxmd_simulation_engine
):
    """Integration test testing a full run of the benchmark."""
    benchmark = solvent_radial_distribution_benchmark
    mock_engine = mock_jaxmd_simulation_engine()
    with patch(
        "mlipaudit.utils.simulation.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        benchmark.run_model()

        assert mock_engine_class.call_count == 1
        assert isinstance(benchmark.model_output, SolventRadialDistributionModelOutput)

        atoms = ase_read(INPUT_DATA_DIR / benchmark.name / "CCl4_eq.pdb")

        num_frames = 2
        positions = np.tile(
            np.array(atoms.positions),
            reps=(num_frames, 1, 1),
        )

        benchmark.model_output = SolventRadialDistributionModelOutput(
            structure_names=["CCl4"],
            simulation_states=[
                SimulationState(positions=positions, temperature=np.ones(10))
            ],
        )

        result = benchmark.analyze()
        assert type(result) is SolventRadialDistributionResult

        assert len(result.structures) == 1
        assert result.structures[0].structure_name == "CCl4"

        # Some leeway around the maximum of 5.9
        assert 5.4 < result.structures[0].first_solvent_peak < 6.4

        assert result.structures[0].peak_deviation < 0.5


def test_analyze_raises_error_if_run_first(solvent_radial_distribution_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        solvent_radial_distribution_benchmark.analyze()
