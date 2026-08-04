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
from mlip.simulation.ase import ASESimulationEngine
from mlip.simulation.jax_md import JaxMDSimulationEngine

from mlipaudit.benchmarks import (
    InferenceSpeedBenchmark,
    InferenceSpeedModelOutput,
    InferenceSpeedResult,
)
from mlipaudit.benchmarks.inference_speed.inference_speed import (
    SCORE_REFERENCE_OVERHEAD_S,
    SCORE_REFERENCE_PER_ATOM_S,
    SIMULATION_CONFIG,
    SIMULATION_CONFIG_DEV,
    InferenceSpeedStructureResult,
    reference_forward_time,
    trim_forward_times,
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


def test_uses_own_dataset(inference_speed_benchmark):
    """The benchmark reads its inputs from its own ``inference_speed`` dataset."""
    assert inference_speed_benchmark.data_name is None
    assert inference_speed_benchmark.data_dir.name == "inference_speed"


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
    # At least one structure succeeded -> the model is not marked failed overall.
    assert not result.failed


@pytest.mark.parametrize("inference_speed_benchmark", [True], indirect=True)
def test_result_failed_when_all_structures_fail(inference_speed_benchmark):
    """If every structure fails, the whole result is marked failed with score 0 (so the
    model surfaces as failed in the UI rather than as an empty, zero-scored chart).
    """
    benchmark = inference_speed_benchmark
    benchmark.model_output = InferenceSpeedModelOutput(
        structure_names=["284_2jof_A", "748_1r0r_I"],
        forward_times=[[], []],
        md_step_times=[{}, {}],
    )

    result = benchmark.analyze()
    assert all(s.failed for s in result.structures)
    assert result.failed
    assert result.score == 0.0


@pytest.mark.parametrize("config", [SIMULATION_CONFIG, SIMULATION_CONFIG_DEV])
def test_md_config_pins_log_interval_for_ase_sampling(config):
    """``log_interval`` must be pinned and strictly below ``num_steps``.

    The ASE backend logs at steps ``0, log_interval, 2*log_interval, ..., num_steps``;
    ``_step_times_from_samples`` then drops the opening (compilation) chunk. If
    ``log_interval`` were left unset it would default to ``num_steps`` here (mlip's
    ``MAX_LOG_FREQ``), leaving only the step-0 and final samples and hence no timing
    chunks at all. Keeping it strictly below ``num_steps`` (and an even divisor)
    guarantees the ASE MD metric is actually populated. It is also kept equal to
    ``snapshot_interval`` so the logger/state-update cadences match, which keeps the
    per-step timing robust regardless of the order mlip attaches those callbacks.
    """
    log_interval = config["log_interval"]
    num_steps = config["num_steps"]
    assert log_interval is not None
    assert 0 < log_interval < num_steps
    assert num_steps % log_interval == 0
    assert log_interval == config["snapshot_interval"]


@pytest.mark.parametrize("config", [SIMULATION_CONFIG, SIMULATION_CONFIG_DEV])
def test_md_config_builds_both_engine_configs(config):
    """The shared MD config dict must be accepted by both engine configs.

    ``_md_kwargs`` is passed unchanged to ``get_simulation_engine``, which builds a
    ``JaxMDSimulationEngine.Config`` for mlip models and an
    ``ASESimulationEngine.Config`` (with ``num_episodes`` removed) for the ASE path.
    If either engine rejected a key — e.g. the pinned ``log_interval``, which is an ASE
    config field — the ``ValidationError`` would be swallowed by ``_time_md`` and that
    backend's metric would silently vanish. Constructing both here guards against that.
    """
    JaxMDSimulationEngine.Config(**config)
    ASESimulationEngine.Config(**{
        key: value for key, value in config.items() if key != "num_episodes"
    })


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


def test_trim_forward_times_drops_slowest_and_sorts():
    """Trimming keeps the fastest 80% of passes, whatever order they arrive in."""
    # 10 passes -> keep 8; the two slowest (0.9, 1.0) go.
    times = [0.5, 1.0, 0.2, 0.4, 0.9, 0.1, 0.3, 0.6, 0.8, 0.7]
    assert trim_forward_times(times) == pytest.approx([
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
    ])

    # Never trims away everything, and handles the empty case.
    assert trim_forward_times([0.5]) == [0.5]
    assert trim_forward_times([]) == []


def test_run_model_keeps_forward_times_in_measurement_order(
    inference_speed_benchmark, monkeypatch
):
    """`run_model` stores the raw series so drift over a run stays inspectable.

    The trimming happens in `analyze`; if it crept back into the measurement path the
    stored order would be lost and a thermal-throttling ramp would be undetectable.
    """
    benchmark = inference_speed_benchmark
    descending = [0.3, 0.2, 0.1]

    monkeypatch.setattr(benchmark, "_warm_up_device", lambda: None)
    monkeypatch.setattr(benchmark, "_time_md", lambda atoms, backend: [])
    monkeypatch.setattr(
        benchmark, "_measure_model_throughput", lambda atoms: list(descending)
    )

    benchmark.run_model()

    assert benchmark.model_output.forward_times[0] == descending
    # ...and `analyze` is what applies the trim.
    assert benchmark.analyze().structures[0].forward_times == pytest.approx([0.1, 0.2])


def test_score_is_size_independent_for_a_model_on_the_reference_curve():
    """A model sitting exactly on the reference curve scores 0.5 at every system size.

    This is the property the affine reference curve buys us over normalising by atom
    count: the forward pass has a large size-independent overhead, so a per-atom
    midpoint would score the same model very differently at 71 and at 6713 atoms.
    """
    B = InferenceSpeedBenchmark

    structures = [
        InferenceSpeedStructureResult(
            structure_name=f"{n}_test",
            num_atoms=n,
            num_steps=10,
            num_episodes=1,
            average_forward_time=reference_forward_time(n),
        )
        for n in (71, 634, 6713)
    ]

    assert B._compute_score(structures) == pytest.approx(0.5)
    # Every structure individually, not just on average.
    for structure in structures:
        assert B._compute_score([structure]) == pytest.approx(0.5)


def test_faster_models_score_higher():
    """The score is monotonically decreasing in forward-pass time."""
    B = InferenceSpeedBenchmark

    def score_at(slowdown: float) -> float:
        return B._compute_score([
            InferenceSpeedStructureResult(
                structure_name="1000_test",
                num_atoms=1000,
                num_steps=10,
                num_episodes=1,
                average_forward_time=reference_forward_time(1000) * slowdown,
            )
        ])

    assert score_at(0.25) > score_at(0.5) > score_at(1.0) > score_at(2.0)
    assert 0.0 < score_at(4.0) < 0.5 < score_at(0.25) < 1.0


def test_reference_forward_time_is_affine_in_system_size():
    """The reference curve is ``overhead + per_atom * N``, both parts strictly used."""
    assert reference_forward_time(0) == pytest.approx(SCORE_REFERENCE_OVERHEAD_S)
    assert reference_forward_time(1000) == pytest.approx(
        SCORE_REFERENCE_OVERHEAD_S + 1000 * SCORE_REFERENCE_PER_ATOM_S
    )
    assert SCORE_REFERENCE_OVERHEAD_S > 0.0
    assert SCORE_REFERENCE_PER_ATOM_S > 0.0


def test_analyze_raises_error_if_run_first(inference_speed_benchmark):
    """Verifies the RuntimeError when analyze is called before run_model."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        inference_speed_benchmark.analyze()
