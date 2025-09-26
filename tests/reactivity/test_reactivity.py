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

# Import the base class as well to help with mocking
from mlipaudit.benchmarks import (
    ReactivityBenchmark,
    ReactivityModelOutput,
)
from mlipaudit.benchmarks.reactivity.reactivity import ReactionModelOutput
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def reactivity_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> ReactivityBenchmark:
    """Assembles a fully configured and isolated ReactivityBenchmark instance.

    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized ReactivityBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return ReactivityBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize("reactivity_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_inference(
    reactivity_benchmark, mocked_batched_inference, mocker
):
    """Integration test using the modular fixture for fast dev run."""
    _mocked_batched_inference = mocker.patch(
        "mlipaudit.utils.inference.run_batched_inference",
        side_effect=mocked_batched_inference,
    )

    reactivity_benchmark.run_model()

    assert type(reactivity_benchmark.model_output) is ReactivityModelOutput

    expected_call_count = 1
    assert _mocked_batched_inference.call_count == expected_call_count


def test_analyze_raises_error_if_run_first(reactivity_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        reactivity_benchmark.analyze()


@pytest.mark.parametrize(
    "reactivity_benchmark, expected_molecules",
    [(True, 2), (False, 3)],
    indirect=["reactivity_benchmark"],
)
def test_data_loading(reactivity_benchmark, expected_molecules):
    """Unit test for the data loading property, parameterized for fast dev run."""
    data = reactivity_benchmark._grambow_data
    assert len(data) == expected_molecules
    assert "005639" in data.keys() and "001299" in data.keys()
    if reactivity_benchmark.run_mode != RunMode.DEV:
        assert "007952" in data.keys()


@pytest.mark.parametrize("reactivity_benchmark", [True], indirect=True)
def test_analyze(reactivity_benchmark):
    """Check the analysis method."""
    benchmark = reactivity_benchmark

    benchmark.model_output = ReactivityModelOutput(
        reaction_ids=["005639", "001299"],
        energy_predictions=[
            ReactionModelOutput(
                reactants_energy=1.0, products_energy=2.0, transition_state_energy=3.0
            ),
            ReactionModelOutput(
                reactants_energy=2.0, products_energy=4.0, transition_state_energy=1.0
            ),
        ],
    )
    result = benchmark.analyze()

    assert len(result.reaction_results) == 2
    assert result.reaction_results["005639"].activation_energy_pred == 2.0
    assert result.reaction_results["005639"].activation_energy_ref == pytest.approx(
        -168909.84985782535 - (-168967.17726805343)
    )
    assert result.reaction_results["005639"].enthalpy_of_reaction_pred == 1.0
    assert result.reaction_results["005639"].enthalpy_of_reaction_ref == pytest.approx(
        -168936.6688414344 - (-168967.17726805343)
    )

    assert result.reaction_results["001299"].activation_energy_pred == -1.0
    assert result.reaction_results["001299"].activation_energy_ref == pytest.approx(
        -203105.67949476154 - (-203179.72142996168)
    )
    assert result.reaction_results["001299"].enthalpy_of_reaction_pred == 2.0
    assert result.reaction_results["001299"].enthalpy_of_reaction_ref == pytest.approx(
        -203149.2080420019 - (-203179.72142996168)
    )

    activation_energy_abs_diffs = np.array([
        abs(2.0 - (-168909.84985782535 - (-168967.17726805343))),
        abs(-1.0 - (-203105.67949476154 - (-203179.72142996168))),
    ])
    enthalpy_abs_diffs = np.array([
        abs(1.0 - (-168936.6688414344 - (-168967.17726805343))),
        abs(2.0 - (-203149.2080420019 - (-203179.72142996168))),
    ])

    assert result.mae_activation_energy == float(np.mean(activation_energy_abs_diffs))
    assert result.rmse_activation_energy == float(
        np.sqrt(np.mean(np.square(activation_energy_abs_diffs)))
    )

    assert result.mae_enthalpy_of_reaction == float(np.mean(enthalpy_abs_diffs))
    assert result.rmse_enthalpy_of_reaction == float(
        np.sqrt(np.mean(np.square(enthalpy_abs_diffs)))
    )
