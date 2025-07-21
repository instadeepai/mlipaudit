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

import os
from pathlib import Path

from mlipaudit.benchmark import Benchmark, BenchmarkResult
from mlipaudit.io import (
    load_benchmark_results_from_disk,
    write_benchmark_results_to_disk,
)


class DummyBenchmarkResultLarge(BenchmarkResult):
    """A dummy benchmark result with 5 entries."""

    a: int
    b: str
    c: list[float]
    d: list[tuple[float, float]]


class DummyBenchmarkResultSmallSubclass(BenchmarkResult):
    """A dummy benchmark result subclass."""

    value: float


class DummyBenchmarkResultSmall(BenchmarkResult):
    """A dummy benchmark result with one entry."""

    values: list[DummyBenchmarkResultSmallSubclass]


class DummyBenchmark1(Benchmark):
    """Dummy benchmark 1."""

    name = "benchmark_1"
    result_class = DummyBenchmarkResultLarge

    def run_model(self) -> None:
        """No need to implement this for this test."""
        pass

    def analyze(self) -> DummyBenchmarkResultLarge:  # type:ignore
        """No need to implement this for this test."""
        pass


class DummyBenchmark2(Benchmark):
    """Dummy benchmark 2."""

    name = "benchmark_2"
    result_class = DummyBenchmarkResultSmall

    def run_model(self) -> None:
        """No need to implement this for this test."""
        pass

    def analyze(self) -> list[DummyBenchmarkResultSmall]:  # type:ignore
        """No need to implement this for this test."""
        pass


def test_benchmark_results_io_works(tmpdir):
    """Tests whether results can be saved and loaded again to and from disk."""
    results_model_1 = {
        "benchmark_1": DummyBenchmarkResultLarge(
            a=7, b="test", c=[3.4, 5.6, 7.8], d=[(1.0, 1.1), (1.2, 1.3)]
        ),
        "benchmark_2": DummyBenchmarkResultSmall(
            values=[
                DummyBenchmarkResultSmallSubclass(value=0.1),
                DummyBenchmarkResultSmallSubclass(value=0.2),
            ]
        ),
    }

    results_model_2 = {
        "benchmark_1": DummyBenchmarkResultLarge(
            a=17, b="test", c=[13.4, 15.6, 17.8], d=[(11.0, 11.1), (11.2, 11.3)]
        ),
        "benchmark_2": DummyBenchmarkResultSmall(
            values=[
                DummyBenchmarkResultSmallSubclass(value=10.1),
                DummyBenchmarkResultSmallSubclass(value=10.2),
            ]
        ),
    }

    write_benchmark_results_to_disk(results_model_1, Path(tmpdir) / "model_1")

    assert set(os.listdir(Path(tmpdir) / "model_1")) == {"benchmark_1", "benchmark_2"}
    assert os.listdir(Path(tmpdir) / "model_1" / "benchmark_1") == ["result.json"]

    write_benchmark_results_to_disk(results_model_2, Path(tmpdir) / "model_2")

    assert set(os.listdir(Path(tmpdir) / "model_2")) == {"benchmark_1", "benchmark_2"}
    assert os.listdir(Path(tmpdir) / "model_2" / "benchmark_1") == ["result.json"]
    assert set(os.listdir(Path(tmpdir))) == {"model_1", "model_2"}

    loaded_results = load_benchmark_results_from_disk(
        tmpdir, [DummyBenchmark1, DummyBenchmark2]
    )

    assert set(loaded_results.keys()) == {"model_1", "model_2"}
    assert loaded_results["model_1"] == results_model_1
    assert loaded_results["model_2"] == results_model_2
