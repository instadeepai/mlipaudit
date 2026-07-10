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

import pytest

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
    assert inference_speed_benchmark.data_name == "scaling"
    assert inference_speed_benchmark.data_dir.name == "scaling"


@pytest.mark.parametrize("inference_speed_benchmark", [True], indirect=True)
def test_full_run_with_mocked_engine(inference_speed_benchmark):
    """Integration test testing the analysis of a full run of the benchmark."""
    benchmark = inference_speed_benchmark

    # md_step_times are already per-step step times (s/step), per backend.
    benchmark.model_output = InferenceSpeedModelOutput(
        structure_names=["284_2jof_A", "748_1r0r_I"],
        forward_times=[[0.002, 0.003], [0.004, 0.006]],
        md_step_times=[
            {"jax_md": [0.04, 0.06], "ase": [0.4, 0.6]},
            {"jax_md": [0.09, 0.11], "ase": [0.9, 1.1]},
        ],
    )

    result = benchmark.analyze()
    assert type(result) is InferenceSpeedResult

    assert len(result.structures) == 2
    s0 = result.structures[0]
    assert s0.structure_name == "284_2jof_A"
    assert s0.num_atoms == 284
    assert s0.timestep_fs == 1
    # MD throughput, per backend (mean of the per-chunk step times).
    assert s0.md["jax_md"].average_step_time == 0.05
    assert s0.md["ase"].average_step_time == 0.5
    assert s0.md["jax_md"].step_time_samples == [0.04, 0.06]
    # Model throughput metric (mean of the timed forward passes).
    assert s0.average_forward_time == 0.0025
    assert s0.forward_times == [0.002, 0.003]
    assert not s0.failed

    # The benchmark produces a speed score in [0, 1] from the forward-pass times.
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


@pytest.mark.parametrize("inference_speed_benchmark", [True], indirect=True)
def test_structure_fails_only_when_all_measurements_fail(inference_speed_benchmark):
    """A structure is `failed` only if the forward pass and every MD backend fail."""
    benchmark = inference_speed_benchmark
    benchmark.model_output = InferenceSpeedModelOutput(
        structure_names=["284_2jof_A", "748_1r0r_I"],
        # First: forward succeeded, no MD -> not failed.
        # Second: nothing at all -> failed.
        forward_times=[[0.002, 0.003], []],
        md_step_times=[{}, {}],
    )

    result = benchmark.analyze()
    assert result.structures[0].average_forward_time == 0.0025
    assert result.structures[0].md == {}
    assert not result.structures[0].failed
    assert result.structures[1].failed


def test_step_times_from_samples():
    """Step times use real step deltas and drop only the compilation chunk."""
    from_samples = InferenceSpeedBenchmark._step_times_from_samples

    # ASE-like: first sample at step 0, so the opening 0->100 interval (which contains
    # compilation) is dropped; remaining chunks give the true per-step time.
    ase = from_samples([(0, 0.0), (100, 5.0), (200, 5.2), (300, 5.4)])
    assert ase == pytest.approx([0.002, 0.002])

    # JAX-MD-like: first logger call is already past the compilation episode (step>0),
    # so every chunk is kept.
    jax_md = from_samples([(250, 10.0), (500, 10.5), (750, 11.0)])
    assert jax_md == pytest.approx([0.002, 0.002])

    # Too little data -> empty.
    assert from_samples([(0, 0.0)]) == []


def test_analyze_raises_error_if_run_first(inference_speed_benchmark):
    """Verifies the RuntimeError when analyze is called before run_model."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        inference_speed_benchmark.analyze()
