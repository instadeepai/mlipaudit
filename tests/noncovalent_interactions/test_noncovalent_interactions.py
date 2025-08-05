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

import pytest

from mlipaudit.noncovalent_interactions import (
    NoncovalentInteractionsBenchmark,
)
from mlipaudit.noncovalent_interactions.noncovalent_interactions import (
    NoncovalentInteractionsModelOutput,
    NoncovalentInteractionsSystemModelOutput,
)

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def noncovalent_interactions_benchmark(
    request, mocked_benchmark_init, mock_force_field
) -> NoncovalentInteractionsBenchmark:
    """Assembles a fully configured and isolated NoncovalentInteractionsBenchmark
    instance.

    This fixture is parameterized to handle the `fast_dev_run` flag.

    Returns:
        An initialized NoncovalentInteractionsBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)

    return NoncovalentInteractionsBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        fast_dev_run=is_fast_run,
    )


@pytest.mark.parametrize(
    "noncovalent_interactions_benchmark", [True, False], indirect=True
)
def test_full_run_with_mocked_inference(
    noncovalent_interactions_benchmark, mocked_batched_inference, mocker
):
    """Integration test using the modular fixture for fast_dev_run."""
    benchmark = noncovalent_interactions_benchmark
    benchmark.fast_dev_run = True

    _mocked_batched_inference = mocker.patch(
        "mlipaudit.noncovalent_interactions.noncovalent_interactions.run_batched_inference",
        side_effect=mocked_batched_inference,
    )

    benchmark.run_model()

    assert type(benchmark.model_output) is NoncovalentInteractionsModelOutput
    assert len(benchmark.model_output.systems) == len(benchmark._nci_atlas_data)
    assert (
        type(benchmark.model_output.systems[0])
        is NoncovalentInteractionsSystemModelOutput
    )
    assert len(benchmark.model_output.systems[0].energy_profile) == len(
        benchmark._nci_atlas_data["1.01.01"].coords
    )

    benchmark.analyze()
