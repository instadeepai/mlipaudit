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

import numpy as np
import pytest
from mlip.simulation import SimulationState

from mlipaudit.benchmarks import (
    ScalingBenchmark,
    ScalingModelOutput,
    ScalingResult,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def scaling_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> ScalingBenchmark:
    """Assembles a fully configured and isolated Scaling instance.
    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized Scaling  instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return ScalingBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize("scaling_benchmark", [True], indirect=True)
def test_full_run_with_mocked_engine(scaling_benchmark, mock_jaxmd_simulation_engine):
    """Integration test testing a full run of the benchmark."""
    benchmark = scaling_benchmark

    # We skip running the model due to complexity in running the
    # Timer with a mocked simulation engine.
    num_frames = 10
    positions_1jrs = np.tile(
        np.ones((71, 3)),
        reps=(num_frames, 1, 1),
    )
    positions_1ay3 = np.tile(
        np.ones((121, 3)),
        reps=(num_frames, 1, 1),
    )

    benchmark.model_output = ScalingModelOutput(
        structure_names=["71_1jrs_leupeptin", "121_1ay3"],
        simulation_states=[
            SimulationState(positions=positions_1jrs),
            SimulationState(positions=positions_1ay3),
        ],
        average_episode_times=[0.05, 0.1],
    )

    result = benchmark.analyze()
    assert type(result) is ScalingResult

    assert len(result.structures) == 2
    assert result.structures[0].structure_name == "71_1jrs_leupeptin"
    assert result.structures[0].num_atoms == 71
    assert result.structures[0].num_steps == 10
    assert result.structures[0].num_episodes == 10
    assert result.structures[0].average_episode_time == 0.05
    assert result.structures[0].average_step_time == 0.05

    assert result.structures[1].average_episode_time == 0.1


def test_analyze_raises_error_if_run_first(scaling_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        scaling_benchmark.analyze()
