import re
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
        benchmark._torsion_net_500["fragment_001"].conformer_coordinates
    )

    result = benchmark.analyze()

    assert type(result) is DihedralScanResult
    assert len(result.fragments) == len(benchmark._torsion_net_500)

    maes = [frag.mae for frag in result.fragments]
    assert result.avg_mae == sum(maes) / len(maes)

    rmses = [frag.rmse for frag in result.fragments]
    assert result.avg_rmse == sum(rmses) / len(rmses)

    expected_call_count = 1
    assert _mocked_batched_inference.call_count == expected_call_count


def test_analyze_raises_error_if_run_first(dihedral_scan_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        dihedral_scan_benchmark.analyze()


@pytest.mark.parametrize(
    "dihedral_scan_benchmark, expected_fragments",
    [(True, 1), (False, 2)],
    indirect=["dihedral_scan_benchmark"],
)
def test_data_loading(dihedral_scan_benchmark, expected_fragments):
    """Unit test for the _torsion_net_500 property, parameterized for fast_dev_run."""
    data = dihedral_scan_benchmark._torsion_net_500
    assert len(data) == expected_fragments
    assert list(data.keys())[0] == "fragment_001"
    if not dihedral_scan_benchmark.fast_dev_run:
        assert list(data.keys())[1] == "fragment_002"
