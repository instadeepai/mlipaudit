from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase.io import read as ase_read
from mlip.simulation import SimulationState

from mlipaudit.folding import FoldingBenchmark
from mlipaudit.folding.folding import (
    FoldingModelOutput,
    FoldingMoleculeResult,
    FoldingResult,
)

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def folding_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> FoldingBenchmark:
    """Assembles a fully configured and isolated FoldingBenchmark instance.

    This fixture is parameterized to handle the `fast_dev_run` flag.

    Returns:
        An initialized FoldingBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)

    return FoldingBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        fast_dev_run=is_fast_run,
    )


@pytest.mark.parametrize("folding_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_simulation_with_static_and_random_trajectory(
    folding_benchmark,
    mock_jaxmd_simulation_engine,
):
    """Integration test using the modular fixture for fast_dev_run."""
    benchmark = folding_benchmark

    atoms = ase_read(INPUT_DATA_DIR / "folding" / "chignolin_1uao_xray.xyz")

    num_steps = 10

    # Case 1: we set the trajectory to be just 10 identical structures
    if benchmark.fast_dev_run:
        traj = np.array([atoms.positions] * num_steps)
        forces = np.zeros(shape=traj.shape)
    # Case 2: we use random structures
    else:
        np.random.seed(42)
        traj = np.random.rand(num_steps, len(atoms), 3)
        forces = np.random.rand(num_steps, len(atoms), 3)

    mock_engine = mock_jaxmd_simulation_engine(
        SimulationState(
            atomic_numbers=atoms.numbers,
            positions=traj,
            forces=forces,
            temperature=np.zeros(num_steps),
        )
    )

    with patch(
        "mlipaudit.folding.folding.JaxMDSimulationEngine",
        return_value=mock_engine,
    ) as mock_engine_class:
        if benchmark.fast_dev_run:  # Case 1
            benchmark.run_model()
        else:  # Case 2
            with pytest.raises(FileNotFoundError):
                benchmark.run_model()

        assert mock_engine_class.call_count == 1
        assert mock_engine.run.call_count == 1

    assert type(benchmark.model_output) is FoldingModelOutput
    assert benchmark.model_output.structure_names == ["chignolin_1uao_xray"]

    result = benchmark.analyze()

    assert type(result) is FoldingResult
    assert type(result.molecules[0]) is FoldingMoleculeResult
    assert len(result.molecules) == 1

    # Case 1: values will all predictable as trajectory is static
    if benchmark.fast_dev_run:
        for i in range(num_steps):
            assert result.molecules[0].rmsd_trajectory[i] < 1e-6
            assert result.molecules[0].tm_score_trajectory[i] == pytest.approx(1.0)
            assert len(set(result.molecules[0].radius_of_gyration)) == 1
            assert len(set(result.molecules[0].proportion_folded_amino_acid)) == 1
            assert len(set(result.molecules[0].match_secondary_structure)) == 1
            assert result.molecules[0].radius_of_gyration_fluctuation == 0.0
        assert result.avg_min_rmsd < 1e-6
        assert result.avg_max_tm_score == pytest.approx(1.0)
    # Case 2: values will fluctuate between frames
    else:
        for i in range(num_steps):
            assert result.molecules[0].rmsd_trajectory[i] > 1e-6
            assert 0.0 < result.molecules[0].tm_score_trajectory[i] < 1.0
            assert len(set(result.molecules[0].radius_of_gyration)) > 1
            assert len(set(result.molecules[0].proportion_folded_amino_acid)) > 1
            assert len(set(result.molecules[0].match_secondary_structure)) > 1
            assert result.molecules[0].radius_of_gyration_fluctuation > 0.0
        assert result.avg_min_rmsd > 1e-6
        assert 0.0 < result.avg_max_tm_score < 1.0
        assert result.avg_min_rmsd == result.molecules[0].min_rmsd
        assert result.avg_max_tm_score == result.molecules[0].max_tm_score
