"""Conformer selection integration test."""

from mlipaudit.small_molecule_conformer_selection.conformer_selection import (
    SmallMoleculeConformerSelectionBenchmark,
)


def test_small_molecule_conformer_selection_benchmark(
    setup_system_and_force_field,
):
    """Test for small molecule conformer selection benchmark."""
    _, _, mace_ff = setup_system_and_force_field
    benchmark = SmallMoleculeConformerSelectionBenchmark(
        force_field=mace_ff, fast_dev_run=True
    )

    benchmark.run_model()
    benchmark.analyze()
