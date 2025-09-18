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
from pathlib import Path

import pytest

from mlipaudit.benchmarks import TautomersBenchmark
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "run_mode,should_fail",
    [
        (RunMode.DEV, False),
        (RunMode.FAST, False),
        (RunMode.STANDARD, False),
        ("dev", False),
        ("fast", False),
        ("standard", False),
        ("devv", True),
        ("STANDARD", True),
        (5, True),
    ],
)
def test_benchmark_can_be_correctly_initialized_with_run_mode(
    mock_force_field, run_mode, should_fail
):
    """Tests that a benchmark can be initialized with run mode as string or
    enum and raises an error if wrong string or other type is given.
    """
    if should_fail:
        with pytest.raises(ValueError) as exc:
            TautomersBenchmark(mock_force_field, INPUT_DATA_DIR, run_mode=run_mode)
            assert "is not a valid RunMode" in str(exc)
    else:
        benchmark = TautomersBenchmark(
            mock_force_field, INPUT_DATA_DIR, run_mode=run_mode
        )
        assert isinstance(benchmark.run_mode, RunMode)
