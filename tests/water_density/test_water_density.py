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

from mlipaudit.benchmarks import (
    WaterDensityBenchmark,
    WaterDensityModelOutput,
    WaterDensityResult,
    WaterRadialDistributionBenchmark,
    WaterRadialDistributionModelOutput,
)
from mlipaudit.benchmarks_cli import _transfer_model_output
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"
# The water density benchmark reuses the water RDF benchmark's input data.
WATER_DATA_DIR = INPUT_DATA_DIR / "water_radial_distribution"


@pytest.fixture
def water_density_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> WaterDensityBenchmark:
    """Assembles a fully configured and isolated WaterDensityBenchmark instance.

    Returns:
        An initialized WaterDensityBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return WaterDensityBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


def _mock_simulation_state() -> SimulationState:
    """Build a stationary water box trajectory with a fixed 24.772 Å box."""
    num_frames = 2
    positions = np.tile(
        np.load(WATER_DATA_DIR / "positions.npy"),
        reps=(num_frames, 1, 1),
    )
    cells = np.tile(24.772 * np.eye(3), reps=(num_frames, 1, 1))
    return SimulationState(
        positions=positions, temperature=np.ones(num_frames), cell=cells
    )


def test_data_name_points_to_shared_data(water_density_benchmark):
    """The density benchmark should read from the RDF benchmark's data directory."""
    assert water_density_benchmark.data_name == "water_radial_distribution"
    assert water_density_benchmark.data_dir == WATER_DATA_DIR


@pytest.mark.parametrize("water_density_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_engine(
    water_density_benchmark, mock_jaxmd_simulation_engine
):
    """Integration test testing a full run of the benchmark."""
    benchmark = water_density_benchmark
    mock_engine = mock_jaxmd_simulation_engine()
    with patch(
        "mlipaudit.utils.simulation.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        benchmark.run_model()

        assert mock_engine_class.call_count == 1
        assert isinstance(benchmark.model_output, WaterDensityModelOutput)

        benchmark.model_output = WaterDensityModelOutput(
            simulation_state=_mock_simulation_state()
        )

        result = benchmark.analyze()
        assert type(result) is WaterDensityResult

        # Target density = 0.997773, initial density = 0.9859266
        assert 0.9 < result.average_density < 1.1
        assert result.density_deviation < 0.1
        assert 0.0 <= result.score <= 1.0


def test_reuses_water_rdf_simulation_output():
    """The density benchmark must be able to reuse the RDF benchmark's model output.

    This mirrors the CLI reuse path in `benchmarks_cli.run_benchmarks`, where a
    cached `ModelOutput` is transferred between benchmarks that share a
    `reusable_output_id`.
    """
    assert (
        WaterDensityBenchmark.reusable_output_id
        == WaterRadialDistributionBenchmark.reusable_output_id
    )

    rdf_output = WaterRadialDistributionModelOutput(
        simulation_state=_mock_simulation_state(), failed=False
    )
    transferred = _transfer_model_output(rdf_output, WaterDensityModelOutput)

    assert isinstance(transferred, WaterDensityModelOutput)
    assert transferred.simulation_state is rdf_output.simulation_state
    assert transferred.failed is False


def test_analyze_raises_error_if_run_first(water_density_benchmark):
    """Verifies the RuntimeError is raised when analyze is called first."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        water_density_benchmark.analyze()
