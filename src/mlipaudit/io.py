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

from mlipaudit.benchmark import BenchmarkResult


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
