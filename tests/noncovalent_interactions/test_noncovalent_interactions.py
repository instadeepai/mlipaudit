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
from ase import units

from mlipaudit.benchmarks.noncovalent_interactions.noncovalent_interactions import (
    NoncovalentInteractionsBenchmark,
    NoncovalentInteractionsModelOutput,
    NoncovalentInteractionsResult,
    NoncovalentInteractionsSystemModelOutput,
    NoncovalentInteractionsSystemResult,
    compute_total_interaction_energy,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def noncovalent_interactions_benchmark(
    request, mocked_benchmark_init, mock_force_field
) -> NoncovalentInteractionsBenchmark:
    """Assembles a fully configured and isolated NoncovalentInteractionsBenchmark
    instance.

    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized NoncovalentInteractionsBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return NoncovalentInteractionsBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize(
    "noncovalent_interactions_benchmark", [True, False], indirect=True
)
def test_full_run_with_mocked_inference(
    noncovalent_interactions_benchmark, mocked_batched_inference, mocker
):
    """Integration test using the modular fixture for fast dev run."""
    benchmark = noncovalent_interactions_benchmark
    benchmark.force_field.allowed_atomic_numbers = list(range(1, 92))

    _mocked_batched_inference = mocker.patch(
        "mlipaudit.utils.inference.run_batched_inference",
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
        benchmark._nci_atlas_data["1.03.03"].coords
    )

    result = benchmark.analyze()

    assert type(result) is NoncovalentInteractionsResult
    assert len(result.systems) == len(benchmark._nci_atlas_data)
    for system_results in result.systems:
        assert system_results.dataset == "Dispersion"

    test_system = result.systems[0]
    assert type(test_system) is NoncovalentInteractionsSystemResult
    assert test_system.system_id == "1.03.03"
    assert len(test_system.reference_energy_profile) == len(
        benchmark._nci_atlas_data["1.03.03"].distance_profile
    )
    assert len(test_system.energy_profile) == len(
        benchmark._nci_atlas_data["1.03.03"].distance_profile
    )
    assert len(test_system.distance_profile) == len(
        benchmark._nci_atlas_data["1.03.03"].distance_profile
    )


def test_compute_total_interaction_energy():
    """Tests the compute_total_interaction_energy function."""
    distance_profile = [1.0, 2.0, 3.0]
    distance_profile_unsorted = [2.0, 3.0, 1.0]
    interaction_energy_profile_attractive = [1.5, -1.0, 0.0]
    interaction_energy_profile_repulsive = [1.4, 1.0, 0.0]
    interaction_energy_profile_unsorted = [-1.0, 0.0, 1.5]

    assert compute_total_interaction_energy(
        distance_profile, interaction_energy_profile_attractive, repulsive=False
    ) == pytest.approx(-1.0)

    assert compute_total_interaction_energy(
        distance_profile, interaction_energy_profile_repulsive, repulsive=True
    ) == pytest.approx(1.4)

    assert compute_total_interaction_energy(
        distance_profile_unsorted, interaction_energy_profile_unsorted, repulsive=False
    ) == pytest.approx(-1.0)


def test_analyze_raises_error_if_run_first(noncovalent_interactions_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        noncovalent_interactions_benchmark.analyze()


@pytest.mark.parametrize(
    "noncovalent_interactions_benchmark, expected_molecules",
    [(True, 2), (False, 2)],
    indirect=["noncovalent_interactions_benchmark"],
)
def test_data_loading(noncovalent_interactions_benchmark, expected_molecules):
    """Unit test for the _nci_atlas_data property, parameterized for fast dev run."""
    data = noncovalent_interactions_benchmark._nci_atlas_data
    assert len(data) == expected_molecules
    assert data["1.03.03"].system_id == "1.03.03"
    if noncovalent_interactions_benchmark.run_mode != RunMode.DEV:
        assert data["1.01.01"].system_id == "1.01.01"


def test_perfect_agreement(noncovalent_interactions_benchmark):
    """Tests that the core mathematical properties of the analyze method hold true."""
    benchmark = noncovalent_interactions_benchmark

    energy_profile_1_kcal_mol = [
        -0.009,
        -0.166,
        -0.17,
        -0.159,
        -0.127,
        0.613,
        -0.024,
        0.194,
        -0.103,
        -0.043,
    ]

    energy_profile_2_kcal_mol = [
        -0.004,
        -0.078,
        -0.087,
        -0.09,
        0.026,
        -0.049,
        -0.048,
        0.176,
        -0.081,
        -0.02,
    ]

    energy_profile_1_ev = [
        x * (units.kcal / units.mol) for x in energy_profile_1_kcal_mol
    ]

    energy_profile_2_ev = [
        x * (units.kcal / units.mol) for x in energy_profile_2_kcal_mol
    ]

    benchmark.model_output = NoncovalentInteractionsModelOutput(
        systems=[
            NoncovalentInteractionsSystemModelOutput(
                system_id="1.03.03",
                energy_profile=energy_profile_1_ev,
            ),
            NoncovalentInteractionsSystemModelOutput(
                system_id="1.01.01",
                energy_profile=energy_profile_2_ev,
            ),
        ],
        n_skipped_unallowed_elements=0,
    )

    result = benchmark.analyze()
    for system_results in result.systems:
        assert system_results.dataset == "Dispersion"
        assert system_results.group == "HCNO"

    assert result.systems[0].reference_interaction_energy == pytest.approx(-0.161)
    assert result.systems[1].reference_interaction_energy == pytest.approx(-0.086)

    assert result.rmse_interaction_energy_all == pytest.approx(0.0)

    assert result.rmse_interaction_energy_datasets["Dispersion"] == pytest.approx(0.0)
    assert result.mae_interaction_energy_datasets["Dispersion"] == pytest.approx(0.0)
    assert result.rmse_interaction_energy_subsets["Dispersion: HCNO"] == pytest.approx(
        0.0
    )
    assert result.mae_interaction_energy_subsets["Dispersion: HCNO"] == pytest.approx(
        0.0
    )


def test_bad_agreement(noncovalent_interactions_benchmark):
    """Tests that the core mathematical properties of the analyze method hold true."""
    benchmark = noncovalent_interactions_benchmark

    benchmark.model_output = NoncovalentInteractionsModelOutput(
        systems=[
            NoncovalentInteractionsSystemModelOutput(
                system_id="1.03.03",
                energy_profile=[0.0] * 10,
            ),
            NoncovalentInteractionsSystemModelOutput(
                system_id="1.01.01",
                energy_profile=[0.0] * 10,
            ),
        ],
        n_skipped_unallowed_elements=0,
    )

    result = benchmark.analyze()
    for system_results in result.systems:
        assert system_results.dataset == "Dispersion"
        assert system_results.group == "HCNO"

    assert result.systems[0].reference_interaction_energy == pytest.approx(-0.161)
    assert result.systems[1].reference_interaction_energy == pytest.approx(-0.086)

    assert result.systems[0].deviation == pytest.approx(0.161)
    assert result.systems[1].deviation == pytest.approx(0.086)

    assert result.rmse_interaction_energy_all == pytest.approx(0.12906781163403988)

    assert result.mae_interaction_energy_datasets["Dispersion"] == pytest.approx(0.1235)
    assert result.rmse_interaction_energy_datasets["Dispersion"] == pytest.approx(
        0.12906781163403988
    )
