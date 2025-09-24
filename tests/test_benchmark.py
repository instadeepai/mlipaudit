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

from mlipaudit.benchmark import Benchmark


def test_missing_attributes(
    mock_force_field, dummy_small_result_class, dummy_model_output_class
):
    """Ensure we cannot create a child class with missing attributes."""
    with pytest.raises(
        NotImplementedError,
        match="DummyBenchmark must override the 'result_class' attribute.",
    ):

        class DummyBenchmark(Benchmark):
            """Dummy benchmark 2."""

            name = "benchmark_2"
            # result_class = dummy_small_result_class
            model_output_class = dummy_model_output_class

            required_elements = {"H", "O"}

            def run_model(self) -> None:
                """No need to implement this for this test."""
                pass

            def analyze(self) -> list[dummy_small_result_class]:  # type:ignore
                """No need to implement this for this test."""
                pass

    with pytest.raises(
        NotImplementedError,
        match="DummyBenchmark must override the 'model_output_class' attribute.",
    ):

        class DummyBenchmark(Benchmark):
            """Dummy benchmark 2."""

            name = "benchmark_2"
            result_class = dummy_small_result_class
            # model_output_class = dummy_model_output_class

            required_elements = {"H", "O"}

            def run_model(self) -> None:
                """No need to implement this for this test."""
                pass

            def analyze(self) -> list[dummy_small_result_class]:  # type:ignore
                """No need to implement this for this test."""
                pass

    with pytest.raises(
        NotImplementedError,
        match="DummyBenchmark must override the 'required_elements' attribute.",
    ):

        class DummyBenchmark(Benchmark):
            """Dummy benchmark 2."""

            name = "benchmark_2"
            result_class = dummy_small_result_class
            model_output_class = dummy_model_output_class

            # required_elements = {"H", "O"}

            def run_model(self) -> None:
                """No need to implement this for this test."""
                pass

            def analyze(self) -> list[dummy_small_result_class]:  # type:ignore
                """No need to implement this for this test."""
                pass
