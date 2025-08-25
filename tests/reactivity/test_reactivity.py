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

# Import the base class as well to help with mocking
from mlipaudit.reactivity import (
    ReactivityBenchmark,
    ReactivityModelOutput,
)

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def reactivity_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> ReactivityBenchmark:
    """Assembles a fully configured and isolated ReactivityBenchmark instance.

    This fixture is parameterized to handle the `fast_dev_run` flag.

    Returns:
        An initialized ReactivityBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)

    return ReactivityBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        fast_dev_run=is_fast_run,
    )


@pytest.mark.parametrize("reactivity_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_inference(
    reactivity_benchmark, mocked_batched_inference, mocker
):
    """Integration test using the modular fixture for fast_dev_run."""
    _mocked_batched_inference = mocker.patch(
        "mlipaudit.reactivity.reactivity.run_batched_inference",
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
    """Unit test for the _wiggle150_data property, parameterized for fast_dev_run."""
    data = reactivity_benchmark._granbow_data
    assert len(data) == expected_molecules
    assert "005639" in data.keys() and "001299" in data.keys()
    if not reactivity_benchmark.fast_dev_run:
        assert "007952" in data.keys()
