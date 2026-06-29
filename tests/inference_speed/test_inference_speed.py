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
    InferenceSpeedBenchmark,
    InferenceSpeedModelOutput,
    InferenceSpeedResult,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def inference_speed_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> InferenceSpeedBenchmark:
    """Assembles a fully configured and isolated InferenceSpeed instance.
    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized InferenceSpeed instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return InferenceSpeedBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


def test_reuses_scaling_dataset(inference_speed_benchmark):
    """The benchmark reads its inputs from the shared ``scaling`` dataset."""
    assert inference_speed_benchmark._dataset_name == "scaling"


@pytest.mark.parametrize("inference_speed_benchmark", [True], indirect=True)
def test_full_run_with_mocked_engine(inference_speed_benchmark):
    """Integration test testing the analysis of a full run of the benchmark."""
    benchmark = inference_speed_benchmark

    num_frames = 10
    positions_2jof = np.tile(np.ones((284, 3)), reps=(num_frames, 1, 1))
    positions_1r0r = np.tile(np.ones((748, 3)), reps=(num_frames, 1, 1))

    benchmark.model_output = InferenceSpeedModelOutput(
        structure_names=["284_2jof_A", "748_1r0r_I"],
        simulation_states=[
            SimulationState(positions=positions_2jof),
            SimulationState(positions=positions_1r0r),
        ],
        average_episode_times=[0.05, 0.1],
        episode_times=[[0.04, 0.06], [0.09, 0.11]],
        forward_times=[[0.002, 0.003], [0.004, 0.006]],
    )

    result = benchmark.analyze()
    assert type(result) is InferenceSpeedResult

    assert len(result.structures) == 2
    s0 = result.structures[0]
    assert s0.structure_name == "284_2jof_A"
    assert s0.num_atoms == 284
    # MD throughput metric.
    assert s0.average_step_time == 0.05
    assert s0.timestep_fs == 1
    assert s0.episode_times == [0.04, 0.06]
    # Model throughput metric (mean of the timed forward passes).
    assert s0.average_forward_time == 0.0025
    assert s0.forward_times == [0.002, 0.003]
    assert not s0.failed

    # The benchmark produces a speed score in [0, 1] from the forward-pass times.
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


@pytest.mark.parametrize("inference_speed_benchmark", [True], indirect=True)
def test_structure_fails_only_when_both_measurements_fail(inference_speed_benchmark):
    """A structure is `failed` only if both the forward pass and MD produced nothing."""
    benchmark = inference_speed_benchmark
    benchmark.model_output = InferenceSpeedModelOutput(
        structure_names=["284_2jof_A", "748_1r0r_I"],
        simulation_states=[None, None],
        # First: MD failed but forward succeeded -> not failed.
        # Second: both failed -> failed.
        average_episode_times=[None, None],
        episode_times=[[], []],
        forward_times=[[0.002, 0.003], []],
    )

    result = benchmark.analyze()
    assert result.structures[0].average_step_time is None
    assert result.structures[0].average_forward_time == 0.0025
    assert not result.structures[0].failed
    assert result.structures[1].failed


def test_analyze_raises_error_if_run_first(inference_speed_benchmark):
    """Verifies the RuntimeError when analyze is called before run_model."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        inference_speed_benchmark.analyze()
