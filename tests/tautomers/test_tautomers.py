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

import math
import re
from pathlib import Path

import pytest
from ase import units

from mlipaudit.benchmarks.tautomers.tautomers import (
    TautomersBenchmark,
    TautomersModelOutput,
    TautomersMoleculeResult,
    TautomersResult,
)
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def tautomers_benchmark(
    request,
    mocked_benchmark_init,  # Use the generic init mock
    mock_force_field,  # Use the generic force field mock
) -> TautomersBenchmark:
    """Assembles a fully configured and isolated TautomersBenchmark instance.

    This fixture is parameterized to handle the `run_mode` flag.

    Returns:
        An initialized TautomersBenchmark instance.
    """
    is_fast_run = getattr(request, "param", False)
    run_mode = RunMode.DEV if is_fast_run else RunMode.STANDARD

    return TautomersBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=run_mode,
    )


@pytest.mark.parametrize("tautomers_benchmark", [True, False], indirect=True)
def test_full_run_with_mocked_inference(
    tautomers_benchmark, mocked_batched_inference, mocker
):
    """Integration test using the modular fixture for fast dev run."""
    benchmark = tautomers_benchmark

    _mocked_batched_inference = mocker.patch(
        "mlipaudit.utils.inference.run_batched_inference",
        side_effect=mocked_batched_inference,
    )

    benchmark.run_model()
    assert type(benchmark.model_output) is TautomersModelOutput

    result = benchmark.analyze()

    assert type(result) is TautomersResult
    assert len(result.molecules) == len(benchmark._tautomers_data)
    assert len(result.molecules) == 2 if benchmark.run_mode == RunMode.DEV else 3
    assert type(result.molecules[0]) is TautomersMoleculeResult

    deviations = []
    for mol in result.molecules:
        assert abs(mol.predicted_energy_diff - mol.ref_energy_diff) == mol.abs_deviation
        deviations.append(mol.abs_deviation)

    assert result.mae == pytest.approx(sum(deviations) / len(deviations))

    assert _mocked_batched_inference.call_count == 1


def test_analyze_raises_error_if_run_first(tautomers_benchmark):
    """Verifies the RuntimeError using the new fixture."""
    expected_message = "Must call run_model() first."
    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        tautomers_benchmark.analyze()


@pytest.mark.parametrize(
    "tautomers_benchmark, expected_molecules",
    [(True, 2), (False, 3)],
    indirect=["tautomers_benchmark"],
)
def test_data_loading(tautomers_benchmark, expected_molecules):
    """Unit test for the _tautomers_data property, parameterized for fast dev run."""
    data = tautomers_benchmark._tautomers_data
    assert len(data) == expected_molecules


@pytest.mark.parametrize("constant_offset", [0.0, 7.1234])
def test_good_agreement(tautomers_benchmark, constant_offset):
    """Tests that the core mathematical properties of the analyze method hold true."""
    benchmark = tautomers_benchmark

    benchmark.model_output = TautomersModelOutput(
        structure_ids=["1069", "0987", "0389"],
        predictions=[
            [-11419.4259769493 + constant_offset, -11419.441686654 + constant_offset],
            [-78839.6852067731 + constant_offset, -78839.7758300742 + constant_offset],
            [-14953.0661039464 + constant_offset, -14953.4743095029 + constant_offset],
        ],
    )

    result = benchmark.analyze()

    for mol in result.molecules:
        assert mol.abs_deviation == 0.0
        assert mol.predicted_energy_diff != 0.0
        assert mol.ref_energy_diff != 0.0

    assert result.mae == 0.0
    assert result.rmse == 0.0


def test_bad_agreement(tautomers_benchmark):
    """Verify analysis outputs in case of bad agreement."""
    benchmark = tautomers_benchmark

    offsets = [1.234, 1.0, 2.345, 3.456]
    benchmark.model_output = TautomersModelOutput(
        structure_ids=["1069", "0987", "0389"],
        predictions=[
            [-11419.4259769493 + offsets[0], -11419.441686654],
            [-78839.6852067731 + offsets[1], -78839.7758300742 + offsets[2]],
            [-14953.0661039464 - offsets[3], -14953.4743095029],
        ],
    )

    result = benchmark.analyze()

    factor = 1.0 / (units.kcal / units.mol)  # unit conversion
    assert result.molecules[0].abs_deviation == pytest.approx(factor * offsets[0])
    assert result.molecules[1].abs_deviation == pytest.approx(
        factor * (offsets[2] - offsets[1])
    )
    assert result.molecules[2].abs_deviation == pytest.approx(factor * offsets[3])

    assert result.mae == pytest.approx(
        factor * (offsets[0] + offsets[2] - offsets[1] + offsets[3]) / 3
    )

    assert result.rmse == pytest.approx(
        math.sqrt(
            (
                factor**2
                * (offsets[0] ** 2 + (offsets[2] - offsets[1]) ** 2 + offsets[3] ** 2)
            )
            / 3
        )
    )
