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
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.io_helpers import (
    dataclass_to_dict_with_arrays,
    dict_with_arrays_to_dataclass,
)

RESULT_FILENAMES = "result.json"
MODEL_OUTPUT_ZIP_FILENAMES = "model_output.zip"
MODEL_OUTPUT_JSON_FILENAMES = "model_output.json"
MODEL_OUTPUT_ARRAYS_FILENAMES = "arrays.npz"


def write_benchmark_results_to_disk(
    results: dict[str, BenchmarkResult],
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
        with (_output_dir / name / RESULT_FILENAMES).open("w") as json_file:
            json_as_str = json.loads(result.model_dump_json())  # type: ignore
            json.dump(json_as_str, json_file, indent=4)


def load_benchmark_results_from_disk(
    results_dir: str | os.PathLike, benchmark_classes: list[type[Benchmark]]
) -> dict[str, dict[str, BenchmarkResult]]:
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

    results: dict[str, dict[str, BenchmarkResult]] = {}
    for model_subdir in _results_dir.iterdir():
        results[model_subdir.name] = {}
        for benchmark_subdir in model_subdir.iterdir():
            for benchmark_class in benchmark_classes:
                if benchmark_subdir.name != benchmark_class.name:
                    continue
                with (benchmark_subdir / RESULT_FILENAMES).open("r") as json_file:
                    json_data = json.load(json_file)

                result = benchmark_class.result_class(**json_data)  # type: ignore

                results[model_subdir.name][benchmark_subdir.name] = result

    return results


def write_model_outputs_to_disk(
    model_outputs: dict[str, ModelOutput], output_dir: str | os.PathLike
) -> None:
    """Writes a collection of model outputs to disk.

    Args:
        model_outputs: The model outputs as a dictionary with the benchmark names
                       as keys.
        output_dir: Directory to which to write the model outputs.
    """
    _output_dir = Path(output_dir)
    _output_dir.mkdir(exist_ok=True, parents=True)

    for name, model_output in model_outputs.items():
        (_output_dir / name).mkdir(exist_ok=True)
        data, arrays = dataclass_to_dict_with_arrays(model_output)

        with TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / MODEL_OUTPUT_JSON_FILENAMES
            arrays_path = Path(tmpdir) / MODEL_OUTPUT_ARRAYS_FILENAMES

            with json_path.open("w") as json_file:
                json.dump(data, json_file)

            np.savez(arrays_path, **arrays)

            with ZipFile(
                _output_dir / name / MODEL_OUTPUT_ZIP_FILENAMES, "w"
            ) as zip_object:
                zip_object.write(json_path, os.path.basename(json_path))
                zip_object.write(arrays_path, os.path.basename(arrays_path))


def load_model_outputs_from_disk(
    model_outputs_dir: str | os.PathLike, benchmark_classes: list[type[Benchmark]]
) -> dict[str, dict[str, ModelOutput]]:
    """Loads model outputs from disk.

    Args:
        model_outputs_dir: The path to the directory with the model_outputs.
        benchmark_classes: A list of benchmark classes that correspond to those
                           benchmarks to load from disk.

    Returns:
        The loaded model outputs. It is a dictionary of dictionaries. The first key
        corresponds to the model names and the second keys are the benchmark names.
    """
    _model_outputs_dir = Path(model_outputs_dir)

    model_outputs: dict[str, dict[str, ModelOutput]] = {}
    for model_subdir in _model_outputs_dir.iterdir():
        model_outputs[model_subdir.name] = {}
        for benchmark_subdir in model_subdir.iterdir():
            for benchmark_class in benchmark_classes:
                if benchmark_subdir.name != benchmark_class.name:
                    continue

                load_path = benchmark_subdir / MODEL_OUTPUT_ZIP_FILENAMES
                with ZipFile(load_path, "r") as zip_object:
                    with zip_object.open(MODEL_OUTPUT_JSON_FILENAMES, "r") as json_file:
                        json_data = json.load(json_file)
                    with zip_object.open(
                        MODEL_OUTPUT_ARRAYS_FILENAMES, "r"
                    ) as arrays_file:
                        npz = np.load(arrays_file)
                        arrays = {key: npz[key] for key in npz.files}

                model_output = dict_with_arrays_to_dataclass(
                    json_data,
                    arrays,
                    benchmark_class.model_output_class,  # type: ignore
                )

                model_outputs[model_subdir.name][benchmark_subdir.name] = model_output

    return model_outputs
