from mlipaudit.small_molecule_conformer_selection import (
    ConformerSelectionBenchmark,
)
from mlipaudit.small_molecule_conformer_selection.conformer_selection import (
    ConformerSelectionBenchmarkResult,
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
    assert len(result[0].predicted_energy_profile) == 51
