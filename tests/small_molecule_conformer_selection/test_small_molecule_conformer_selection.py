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

import numpy as np
import pytest
from ase import units

from mlipaudit.small_molecule_conformer_selection import (
    ConformerSelectionBenchmark,
)
from mlipaudit.small_molecule_conformer_selection.conformer_selection import (
    ConformerSelectionModelOutput,
    ConformerSelectionMoleculeModelOutput,
    ConformerSelectionMoleculeResult,
    ConformerSelectionResult,
)


@pytest.mark.parametrize("fast_dev_run", [True, False])
def test_small_molecule_conformer_selection_benchmark_runs_through(
    load_force_field, get_data_input_dir, fast_dev_run
):
    """This tests runs through the conformer selection benchmark end-to-end."""
    force_field = load_force_field
    data_input_dir = get_data_input_dir
    benchmark = ConformerSelectionBenchmark(
        force_field=force_field,
        data_input_dir=data_input_dir,
        fast_dev_run=fast_dev_run,
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


@pytest.mark.parametrize("constant_offset", [0.0, 7.1234])
def test_small_molecule_conformer_selection_benchmark_outputs_perfect_agreement(
    load_force_field, get_data_input_dir, constant_offset
):
    """This test checks that if the model output matches the reference data,
    we get perfect metrics as output.
    """
    force_field = load_force_field
    data_input_dir = get_data_input_dir
    benchmark = ConformerSelectionBenchmark(
        force_field=force_field, data_input_dir=data_input_dir, fast_dev_run=True
    )

    # This matches the input data
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

    # Hack to only look at the first three conformers in the reference
    benchmark._wiggle150_data[0].dft_energy_profile = benchmark._wiggle150_data[
        0
    ].dft_energy_profile[:3]

    result = benchmark.analyze()
    result = result.molecules[0]

    assert result.mae < 1e-9
    assert result.rmse < 1e-9
    assert result.spearman_correlation == 1
    assert result.spearman_p_value == 0
    assert result.predicted_energy_profile == pytest.approx(
        result.reference_energy_profile
    )


@pytest.mark.parametrize("offsets", [[1.0, 8.0], [42.0, 0], [-200.0, -400.0]])
def test_small_molecule_conformer_selection_benchmark_outputs_bad_agreement(
    load_force_field, get_data_input_dir, offsets
):
    """This test checks the analysis output of the benchmark w.r.t. artificially set
    errors to verify multiple ranking scenarios.
    """
    force_field = load_force_field
    data_input_dir = get_data_input_dir
    benchmark = ConformerSelectionBenchmark(
        force_field=force_field, data_input_dir=data_input_dir
    )

    # This matches the input data
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

    benchmark._wiggle150_data[0].dft_energy_profile = benchmark._wiggle150_data[
        0
    ].dft_energy_profile[:3]

    result = benchmark.analyze()
    result = result.molecules[0]

    # This should lead to small MAE and still perfect correlation
    if offsets[0] == 1.0:
        assert result.mae == pytest.approx(3.0)
        assert result.rmse == pytest.approx(np.sqrt((1 + 64) / 3))
        assert result.spearman_correlation == 1
        assert result.spearman_p_value == 0

    # This changes the order, so that the correlation is not perfect anymore
    elif offsets[0] == 42.0:
        assert result.mae == pytest.approx(14.0)
        assert result.rmse == pytest.approx(np.sqrt((42**2) / 3))
        assert result.spearman_correlation == 0.5
        assert result.spearman_p_value > 0

    # This reverses the order, so that the correlation will be -1
    elif offsets[0] == -200.0:
        assert result.mae == pytest.approx(200.0)
        assert result.rmse == pytest.approx(np.sqrt((200**2 + 400**2) / 3))
        assert result.spearman_correlation == -1
        assert result.spearman_p_value == 0

    # This assert should hold in all cases
    assert result.predicted_energy_profile != pytest.approx(
        result.reference_energy_profile
    )
