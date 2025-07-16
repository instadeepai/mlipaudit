import numpy as np
import pytest
from ase import units

from mlipaudit.small_molecule_conformer_selection import (
    ConformerSelectionBenchmark,
)
from mlipaudit.small_molecule_conformer_selection.conformer_selection import (
    ConformerSelectionBenchmarkResult,
    ConformerSelectionModelOutput,
)


def test_small_molecule_conformer_selection_benchmark_runs_through(
    load_force_field, get_data_input_dir
):
    """This tests runs through the conformer selection benchmark end-to-end."""
    force_field = load_force_field
    data_input_dir = get_data_input_dir
    benchmark = ConformerSelectionBenchmark(
        force_field=force_field, data_input_dir=data_input_dir
    )

    benchmark.run_model()
    result = benchmark.analyze()

    assert len(result) == 1
    assert type(result[0]) is ConformerSelectionBenchmarkResult
    assert len(result[0].predicted_energy_profile) == 3
    assert len(result[0].reference_energy_profile) == 3


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
        force_field=force_field, data_input_dir=data_input_dir
    )

    # This matches the input data
    benchmark.model_output = [
        ConformerSelectionModelOutput(
            molecule_name="ado",
            predicted_energy_profile=[
                (-603984.7444920692 + constant_offset) * (units.kcal / units.mol),
                (-603878.2798760363 + constant_offset) * (units.kcal / units.mol),
                (-603844.3333408346 + constant_offset) * (units.kcal / units.mol),
            ],
        )
    ]

    result = benchmark.analyze()

    assert result[0].mae < 1e-9
    assert result[0].rmse < 1e-9
    assert result[0].spearman_correlation == 1
    assert result[0].spearman_p_value == 0
    assert result[0].predicted_energy_profile == pytest.approx(
        result[0].reference_energy_profile
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
    benchmark.model_output = [
        ConformerSelectionModelOutput(
            molecule_name="ado",
            predicted_energy_profile=[
                -603984.7444920692 * (units.kcal / units.mol),
                (-603878.2798760363 + offsets[0]) * (units.kcal / units.mol),
                (-603844.3333408346 + offsets[1]) * (units.kcal / units.mol),
            ],
        )
    ]

    result = benchmark.analyze()

    # This should lead to small MAE and still perfect correlation
    if offsets[0] == 1.0:
        assert result[0].mae == pytest.approx(3.0)
        assert result[0].rmse == pytest.approx(np.sqrt((1 + 64) / 3))
        assert result[0].spearman_correlation == 1
        assert result[0].spearman_p_value == 0

    # This changes the order, so that the correlation is not perfect anymore
    elif offsets[0] == 42.0:
        assert result[0].mae == pytest.approx(14.0)
        assert result[0].rmse == pytest.approx(np.sqrt((42**2) / 3))
        assert result[0].spearman_correlation == 0.5
        assert result[0].spearman_p_value > 0

    # This reverses the order, so that the correlation will be -1
    elif offsets[0] == -200.0:
        assert result[0].mae == pytest.approx(200.0)
        assert result[0].rmse == pytest.approx(np.sqrt((200**2 + 400**2) / 3))
        assert result[0].spearman_correlation == -1
        assert result[0].spearman_p_value == 0

    # This assert should hold in all cases
    assert result[0].predicted_energy_profile != pytest.approx(
        result[0].reference_energy_profile
    )
