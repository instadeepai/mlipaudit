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
from ase import units

# Import the base class as well to help with mocking
from mlipaudit.conformer_selection import (
    ConformerSelectionBenchmark,
)
from mlipaudit.conformer_selection.conformer_selection import (
    ConformerSelectionModelOutput,
    ConformerSelectionMoleculeModelOutput,
    ConformerSelectionMoleculeResult,
    ConformerSelectionResult,
)

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def conformer_selection_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> ConformerSelectionBenchmark:
    """Assembles a fully configured and isolated ConformerSelectionBenchmark instance.

    This fixture is parameterized to handle the `fast_dev_run` flag.

    Returns:
        An initialized ConformerSelectionBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)

    return ConformerSelectionBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        fast_dev_run=is_fast_run,
    )


@pytest.mark.parametrize("conformer_selection_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_inference(
    conformer_selection_benchmark, mocked_batched_inference, mocker
):
    """Integration test using the modular fixture for fast_dev_run."""
    benchmark = conformer_selection_benchmark

    original_data = benchmark._wiggle150_data

    _mocked_batched_inference = mocker.patch(
        "mlipaudit.conformer_selection.conformer_selection.run_batched_inference",
        side_effect=mocked_batched_inference,
    )

    benchmark.run_model()

    assert type(benchmark.model_output) is ConformerSelectionModelOutput
    assert (
        type(benchmark.model_output.molecules[0])
        is ConformerSelectionMoleculeModelOutput
    )
    assert len(benchmark.model_output.molecules[0].predicted_energy_profile) == len(
        benchmark._wiggle150_data[0].conformer_coordinates
    )
    result = benchmark.analyze()

    assert type(result) is ConformerSelectionResult
    assert len(result.molecules) == len(benchmark._wiggle150_data)
    assert type(result.molecules[0]) is ConformerSelectionMoleculeResult
    assert len(result.molecules[0].predicted_energy_profile) == len(
        benchmark._wiggle150_data[0].conformer_coordinates
    )
    assert len(result.molecules[0].reference_energy_profile) == len(
        benchmark._wiggle150_data[0].dft_energy_profile
    )
    maes = [mol.mae for mol in result.molecules]
    rmses = [mol.rmse for mol in result.molecules]
    assert result.avg_mae == sum(maes) / len(maes)
    assert result.avg_rmse == sum(rmses) / len(rmses)

    expected_call_count = 1 if benchmark.fast_dev_run else 2
    assert _mocked_batched_inference.call_count == expected_call_count
    assert len(result.molecules) == len(original_data)


def test_analyze_raises_error_if_run_first(conformer_selection_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        conformer_selection_benchmark.analyze()


@pytest.mark.parametrize(
    "conformer_selection_benchmark, expected_molecules",
    [(True, 1), (False, 2)],
    indirect=["conformer_selection_benchmark"],
)
def test_data_loading(conformer_selection_benchmark, expected_molecules):
    """Unit test for the _wiggle150_data property, parameterized for fast_dev_run."""
    data = conformer_selection_benchmark._wiggle150_data
    assert len(data) == expected_molecules
    assert data[0].molecule_name == "ado"
    if not conformer_selection_benchmark.fast_dev_run:
        assert data[1].molecule_name == "bpn"


@pytest.mark.parametrize("constant_offset", [0.0, 7.1234])
def test_good_agreement(conformer_selection_benchmark, constant_offset):
    """Tests that the core mathematical properties of the analyze method hold true."""
    benchmark = conformer_selection_benchmark

    benchmark.model_output = ConformerSelectionModelOutput(
        molecules=[
            ConformerSelectionMoleculeModelOutput(
                molecule_name="ado",
                predicted_energy_profile=[
                    (-603984.7444920692 + constant_offset) * (units.kcal / units.mol),
                    (-603878.2798760363 + constant_offset) * (units.kcal / units.mol),
                    (-603844.3333408346 + constant_offset) * (units.kcal / units.mol),
                ],
            )
        ]
    )

    result = benchmark.analyze()
    mol_result = result.molecules[0]

    assert mol_result.mae < 1e-9
    assert mol_result.rmse < 1e-9
    assert mol_result.spearman_correlation == 1
    assert mol_result.spearman_p_value == 0
    assert mol_result.predicted_energy_profile == pytest.approx(
        mol_result.reference_energy_profile
    )

    min_ref_idx = np.argmin(mol_result.reference_energy_profile)
    assert np.min(mol_result.reference_energy_profile) == pytest.approx(0.0)
    assert mol_result.reference_energy_profile[min_ref_idx] == pytest.approx(0.0)

    assert mol_result.predicted_energy_profile[min_ref_idx] == pytest.approx(0.0)


@pytest.mark.parametrize("offsets", [[1.0, 8.0], [42.0, 0], [-200.0, -400.0]])
def test_bad_agreement(conformer_selection_benchmark, offsets):
    """Verify analysis outputs in case of bad agreement."""
    benchmark = conformer_selection_benchmark

    benchmark.model_output = ConformerSelectionModelOutput(
        molecules=[
            ConformerSelectionMoleculeModelOutput(
                molecule_name="ado",
                predicted_energy_profile=[
                    -603984.7444920692 * (units.kcal / units.mol),
                    (-603878.2798760363 + offsets[0]) * (units.kcal / units.mol),
                    (-603844.3333408346 + offsets[1]) * (units.kcal / units.mol),
                ],
            )
        ]
    )

    result = benchmark.analyze()
    mol_result = result.molecules[0]

    if offsets[0] == 1.0:
        assert mol_result.mae == pytest.approx(3.0)
        assert mol_result.rmse == pytest.approx(np.sqrt((1 + 64) / 3))
        assert mol_result.spearman_correlation == 1
        assert mol_result.spearman_p_value == 0

    # This changes the order, so that the correlation is not perfect anymore
    elif offsets[0] == 42.0:
        assert mol_result.mae == pytest.approx(14.0)
        assert mol_result.rmse == pytest.approx(np.sqrt((42**2) / 3))
        assert mol_result.spearman_correlation == 0.5
        assert mol_result.spearman_p_value > 0

    # This reverses the order, so that the correlation will be -1
    elif offsets[0] == -200.0:
        assert mol_result.mae == pytest.approx(200.0)
        assert mol_result.rmse == pytest.approx(np.sqrt((200**2 + 400**2) / 3))
        assert mol_result.spearman_correlation == -1
        assert mol_result.spearman_p_value == 0

    # This assert should hold in all cases
    assert mol_result.predicted_energy_profile != pytest.approx(
        mol_result.reference_energy_profile
    )
