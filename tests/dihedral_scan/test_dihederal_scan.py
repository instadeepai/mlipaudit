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

from mlipaudit.benchmarks.dihedral_scan.dihedral_scan import (
    DihedralScanBenchmark,
    DihedralScanModelOutput,
    DihedralScanResult,
    FragmentModelOutput,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def dihedral_scan_benchmark(
    request, mocked_benchmark_init, mock_force_field
) -> DihedralScanBenchmark:
    """Assembles a fully configured and isolated DihedralScanBenchmark instance.

    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized DihedralScanBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return DihedralScanBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize("dihedral_scan_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_inference(
    dihedral_scan_benchmark, mocked_batched_inference, mocker
):
    """Integration test using the modular fixture for fast dev run."""
    benchmark = dihedral_scan_benchmark

    _mocked_batched_inference = mocker.patch(
        "mlipaudit.utils.inference.run_batched_inference",
        side_effect=mocked_batched_inference,
    )

    # Needs to be mocked because the limited test data will produce NaNs
    _mocked_spearman_r = mocker.patch(
        "mlipaudit.benchmarks.dihedral_scan.dihedral_scan.pearsonr",
        side_effect=lambda x, y: (1.0, 0.0),
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
    [(True, 2), (False, 2)],
    indirect=["dihedral_scan_benchmark"],
)
def test_data_loading(dihedral_scan_benchmark, expected_fragments):
    """Unit test for the _torsion_net_500 property, parameterized for fast dev run."""
    data = dihedral_scan_benchmark._torsion_net_500
    assert len(data) == expected_fragments
    assert list(data.keys())[0] == "fragment_001"
    assert list(data.keys())[1] == "fragment_002"


def test_analyze(dihedral_scan_benchmark):
    """Test analysis."""
    benchmark = dihedral_scan_benchmark

    benchmark.model_output = DihedralScanModelOutput(
        fragments=[
            FragmentModelOutput(
                fragment_name="fragment_001",  # Perfect agreement
                energy_predictions=[  # Convert from kcal/mol to eV
                    11.081099561648443 * (units.kcal / units.mol),
                    0.0,
                    10.79156492796028 * (units.kcal / units.mol),
                ],
            ),
            FragmentModelOutput(
                fragment_name="fragment_002",  # Some deviation
                energy_predictions=[
                    3.0 * (units.kcal / units.mol),
                    2.0 * (units.kcal / units.mol),
                ],
            ),
        ]
    )
    result = benchmark.analyze()

    assert result.fragments[0].mae < 1e-9
    assert result.fragments[0].rmse < 1e-9
    assert result.fragments[0].pearson_r == pytest.approx(1.0)
    assert result.fragments[0].pearson_p == pytest.approx(0.0)
    assert result.fragments[0].barrier_height_error < 1e-9

    assert (
        result.fragments[0].predicted_energy_profile
        == result.fragments[0].reference_energy_profile
    )

    # Align the profiles
    min_ref_idx = 1

    predicted_energy_profile = np.array([3.0, 2.0]) * (units.kcal / units.mol)
    predicted_energy_profile_aligned_2 = (
        predicted_energy_profile - predicted_energy_profile[min_ref_idx]
    )

    predicted_energy_profile_aligned_2 /= units.kcal / units.mol
    assert result.fragments[1].mae == pytest.approx(
        np.mean(
            np.abs(
                predicted_energy_profile_aligned_2 - np.array([2.2851370912976563, 0.0])
            )
        )
    )
    assert result.fragments[1].rmse == pytest.approx(
        np.sqrt(
            np.mean(
                np.square(
                    predicted_energy_profile_aligned_2
                    - np.array([2.2851370912976563, 0.0])
                )
            )
        )
    )
