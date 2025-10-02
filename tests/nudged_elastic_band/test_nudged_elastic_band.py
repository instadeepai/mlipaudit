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
from unittest.mock import patch

import numpy as np
import pytest

from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import (
    NudgedElasticBandBenchmark,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def nudged_elastic_band_benchmark(request, mocked_benchmark_init, mock_force_field):
    """Fixture for the nudged elastic band benchmark."""
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return NudgedElasticBandBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize("nudged_elastic_band_benchmark", [True], indirect=True)
def test_nudged_elastic_band_benchmark_can_be_run(
    nudged_elastic_band_benchmark,
    mock_ase_simulation_engine,
    mock_neb_simulation_engine,
):
    """Integration test for a full run of the nudged elastic band benchmark."""
    mock_ase_1 = mock_ase_simulation_engine()
    mock_ase_2 = mock_ase_simulation_engine()
    mock_ase_3 = mock_ase_simulation_engine()
    mock_ase_4 = mock_ase_simulation_engine()

    mock_neb_1 = mock_neb_simulation_engine()
    mock_neb_2 = mock_neb_simulation_engine()
    mock_neb_3 = mock_neb_simulation_engine()
    mock_neb_4 = mock_neb_simulation_engine()

    with patch(
        "mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band.ASESimulationEngine",
        side_effect=[mock_ase_1, mock_ase_2, mock_ase_3, mock_ase_4],
    ) as mock_ase_engine_class:
        with patch(
            "mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band.NEBSimulationEngine",
            side_effect=[mock_neb_1, mock_neb_2, mock_neb_3, mock_neb_4],
        ) as mock_neb_engine_class:
            nudged_elastic_band_benchmark.run_model()
            assert mock_ase_engine_class.call_count == 4
            assert mock_neb_engine_class.call_count == 4

            assert (
                len(nudged_elastic_band_benchmark.model_output.simulation_states) == 2
            )

            for state in nudged_elastic_band_benchmark.model_output.simulation_states:
                state.forces = np.zeros(state.forces.shape)

            result = nudged_elastic_band_benchmark.analyze()
            assert result.convergence_rate == pytest.approx(1.0)
            assert result.score == pytest.approx(1.0)
            assert len(result.reaction_results) == 2

            for state in nudged_elastic_band_benchmark.model_output.simulation_states:
                state.forces[0] += 0.1

            result = nudged_elastic_band_benchmark.analyze()
            assert result.convergence_rate == pytest.approx(0.0)
            assert result.score == pytest.approx(0.0)
