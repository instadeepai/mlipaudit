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

import logging

from mlipaudit.benchmark import Benchmark, BenchmarkResult

logger = logging.getLogger("mlipaudit")


class BondLengthDistributionBenchmark(Benchmark):
    """Benchmark for small organic molecule bond length distribution."""

    def run_model(self) -> None:
        """Run model."""
        raise NotImplementedError

    def analyze(self) -> BenchmarkResult:
        """Analyze results."""
        raise NotImplementedError
