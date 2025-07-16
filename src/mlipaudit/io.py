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

import json
import os
from pathlib import Path

from mlipaudit.benchmark import Benchmark, BenchmarkResult


def write_benchmark_results_to_disk(
    results: dict[str, BenchmarkResult | list[BenchmarkResult]],
    output_dir: str | os.PathLike,
) -> None:
    """Writes a collection of benchmark results to disk.

    Args:
        results: The results as a dictionary with the benchmark names as keys.
        output_dir: Directory to which to write the results.
    """
    _output_dir = Path(output_dir)
    _output_dir.mkdir(exist_ok=True, parents=True)

    for name, result in results.items():
        (_output_dir / name).mkdir(exist_ok=True)
        with (_output_dir / name / "result.json").open("w") as json_file:
            if type(result) is list:
                json_as_str = [json.loads(item.model_dump_json()) for item in result]
            else:
                json_as_str = json.loads(result.model_dump_json())  # type: ignore
            json.dump(json_as_str, json_file, indent=4)


def load_benchmark_results_from_disk(
    results_dir: str | os.PathLike, benchmark_classes: list[type[Benchmark]]
) -> dict[str, dict[str, BenchmarkResult | list[BenchmarkResult]]]:
    """Loads benchmark results from disk.

    Args:
        results_dir: The path to the directory with the results.
        benchmark_classes: A list of benchmark classes that correspond to those
                           benchmarks to load from disk.

    Returns:
        The loaded results. It is a dictionary of dictionaries. The first key
        corresponds to the model names and the second keys are the benchmark names.
    """
    _results_dir = Path(results_dir)

    results: dict[str, dict[str, BenchmarkResult | list[BenchmarkResult]]] = {}
    for model_subdir in _results_dir.iterdir():
        results[model_subdir.name] = {}
        for benchmark_subdir in model_subdir.iterdir():
            for benchmark_class in benchmark_classes:
                if benchmark_subdir.name != benchmark_class.name:
                    continue
                with (benchmark_subdir / "result.json").open("r") as json_file:
                    json_data = json.load(json_file)

                if type(json_data) is list:
                    result = [
                        benchmark_class.result_class(**item)  # type: ignore
                        for item in json_data
                    ]
                else:
                    result = benchmark_class.result_class(**json_data)  # type: ignore

                results[model_subdir.name][benchmark_subdir.name] = result

    return results
