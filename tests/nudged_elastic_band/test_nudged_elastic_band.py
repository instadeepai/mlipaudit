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

import pytest
from pathlib import Path

from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import NudgedElasticBandBenchmark
from mlipaudit.run_mode import RunMode

INPUT_DATA_DIR = Path(__file__).parent.parent / "data"

@pytest.fixture
def nudged_elastic_band_benchmark(mocked_benchmark_init, mock_force_field):
    return NudgedElasticBandBenchmark(
        force_field=mock_force_field,
        data_input_dir=INPUT_DATA_DIR,
        run_mode=RunMode.STANDARD,
    )

def test_nudged_elastic_band_benchmark_can_be_run(nudged_elastic_band_benchmark):
    nudged_elastic_band_benchmark.run_model()
    assert nudged_elastic_band_benchmark.model_output is not None