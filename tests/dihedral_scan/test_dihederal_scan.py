from pathlib import Path

import pytest

from mlipaudit.dihedral_scan.dihedral_scan import (
    DihedralScanBenchmark,
    DihedralScanModelOutput,
    DihedralScanResult,
    FragmentModelOutput,
)

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def dihedral_scan_benchmark(
    request, mocked_benchmark_init, mock_force_field
) -> DihedralScanBenchmark:
    """Assembles a fully configured and isolated DihedralScanBenchmark instance.

    This fixture is parameterized to handle the `fast_dev_run` flag.

    Returns:
        An initialized DihedralScanBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)

    return DihedralScanBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        fast_dev_run=is_fast_run,
    )


@pytest.mark.parametrize("dihedral_scan_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_inference(
    dihedral_scan_benchmark, mocked_batched_inference, mocker
):
    """Integration test using the modular fixture for fast_dev_run."""
    benchmark = dihedral_scan_benchmark

    _mocked_batched_inference = mocker.patch(
        "mlipaudit.dihedral_scan.dihedral_scan.run_batched_inference",
        side_effect=mocked_batched_inference,
    )

    benchmark.run_model()

    assert type(benchmark.model_output) is DihedralScanModelOutput
    assert type(benchmark.model_output.fragments[0] is FragmentModelOutput)
    assert len(benchmark.model_output.fragments[0].energy_predictions) == len(
        benchmark._torsion_net_500[0].conformer_coordinates
    )

    result = benchmark.analyze()

    assert type(result) is DihedralScanResult
