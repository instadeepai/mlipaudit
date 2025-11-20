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
from mlipaudit.benchmarks import BENCHMARK_NAMES, BENCHMARKS_WITHOUT_SCORES
from mlipaudit.io_helpers import (
    dataclass_to_dict_with_arrays,
    dict_with_arrays_to_dataclass,
)

RESULT_FILENAME = "result.json"
SCORE_FILENAME = "score.json"
OVERALL_SCORE_KEY_NAME = "overall_score"
MODEL_OUTPUT_ZIP_FILENAME = "model_output.zip"
MODEL_OUTPUT_JSON_FILENAME = "model_output.json"
MODEL_OUTPUT_ARRAYS_FILENAME = "arrays.npz"


def write_benchmark_result_to_disk(
    benchmark_name: str,
    result: BenchmarkResult,
    output_dir: str | os.PathLike,
) -> None:
    """Writes a benchmark result to disk.

    Args:
        benchmark_name: The benchmark name.
        result: The benchmark result.
        output_dir: Directory to which to write the result.
    """
    _output_dir = Path(output_dir)
    _output_dir.mkdir(exist_ok=True, parents=True)
    (_output_dir / benchmark_name).mkdir(exist_ok=True)

    with open(
        _output_dir / benchmark_name / RESULT_FILENAME, mode="w", encoding="utf-8"
    ) as json_file:
        json_as_str = json.loads(result.model_dump_json())  # type: ignore
        json.dump(json_as_str, json_file, indent=2)


def load_benchmark_result_from_disk(
    results_dir: str | os.PathLike,
    benchmark_class: type[Benchmark],
) -> BenchmarkResult:
    """Loads a benchmark result from disk.

    Args:
        results_dir: The path to the directory with the results.
        benchmark_class: The benchmark class that corresponds to the
                         benchmark to load from disk.

    Returns:
        The loaded benchmark result.
    """
    _results_dir = Path(results_dir)
    benchmark_subdir = _results_dir / benchmark_class.name

    with open(
        benchmark_subdir / RESULT_FILENAME, mode="r", encoding="utf-8"
    ) as json_file:
        json_data = json.load(json_file)

    return benchmark_class.result_class(**json_data)  # type: ignore


def load_benchmark_results_from_disk(
    results_dir: str | os.PathLike, benchmark_classes: list[type[Benchmark]]
) -> dict[str, dict[str, BenchmarkResult]]:
    """Loads benchmark results from disk.

    Note that we handle hidden files by ignoring them.

    This expects the results to be in our convention of directory structure
    which is `<results_dir>/<model_name>/<benchmark_name>/result.json`, i.e.,
    the individual results for each model and their subdirectories containing the
    individual results for each benchmark in a `result.json` file.

    The results are loaded all together and not only one at a time with this function
    as this corresponds to the most common use case of the UI app, and the results
    are not expected to be too large in memory (in contrast, for example, to the
    model outputs).

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
        if model_subdir.stem.startswith("."):
            continue
        results[model_subdir.name] = {}
        for benchmark_subdir in model_subdir.iterdir():
            if benchmark_subdir.stem.startswith("."):
                continue
            for benchmark_class in benchmark_classes:
                if benchmark_subdir.name != benchmark_class.name:
                    continue

                result_path = benchmark_subdir / RESULT_FILENAME
                if result_path.is_file():
                    with open(result_path, mode="r", encoding="utf-8") as f:
                        result = benchmark_class.result_class.model_validate_json(  # type: ignore
                            f.read()
                        )

                    results[model_subdir.name][benchmark_subdir.name] = result

    return results


def generate_empty_scores_dict() -> dict[str, float | None]:
    """Generate a scores dict with scores of 0.0 assigned to
    benchmarks returning a score and scores of None for those
    that don't.

    Returns:
        The dictionary of 0 or null scores.
    """
    padded_scores: dict[str, float | None] = {}
    benchmarks_without_scores_names = [b.name for b in BENCHMARKS_WITHOUT_SCORES]
    for benchmark_name in BENCHMARK_NAMES:
        if benchmark_name not in benchmarks_without_scores_names:
            padded_scores[benchmark_name] = 0.0
        else:
            padded_scores[benchmark_name] = None

    return padded_scores


def write_scores_to_disk(
    scores: dict[str, float | None],
    output_dir: str | os.PathLike,
) -> None:
    """Writes the scores to disk. This will populate the resulting json
    with the generated scores, as well as scores of 0.0 for benchmarks
    that were skipped and scores of None for benchmarks that don't return
    scores.

    Args:
        scores: The results as a dictionary with the benchmark names as keys
            and their scores as values.
        output_dir: Directory to which to write the results.
    """
    _output_dir = Path(output_dir)
    _output_dir.mkdir(exist_ok=True, parents=True)

    with open(_output_dir / SCORE_FILENAME, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


def load_score_from_disk(
    output_dir: str | os.PathLike,
) -> dict[str, float]:
    """Loads the scores from disk for a single model.

    Args:
        output_dir: Directory from which to load the scores.
            Should point to the folder for the results of
            a single model.

    Returns:
        A dictionary of scores where the keys are the
            benchmark names.
    """
    with open(Path(output_dir) / SCORE_FILENAME, mode="r", encoding="utf-8") as f:
        scores = json.load(f)

    return scores


def load_scores_from_disk(
    scores_dir: str | os.PathLike,
) -> dict[str, dict[str, float]]:
    """Loads the scores from disk for all models.

    Args:
        scores_dir: Directory from which to load the scores.
            Should point to the folder for the results of
            multiple models.

    Returns:
        A dictionary of dictionaries where the first keys
            are the model names and the second keys the
            benchmark names.
    """
    _scores_dir = Path(scores_dir)
    scores = {}
    for model_subdir in _scores_dir.iterdir():
        if model_subdir.stem.startswith("."):
            continue
        with open(model_subdir / SCORE_FILENAME, "r", encoding="utf-8") as json_file:
            model_scores = json.load(json_file)
        scores[model_subdir.name] = model_scores
    return scores


def write_model_output_to_disk(
    benchmark_name: str, model_output: ModelOutput, output_dir: str | os.PathLike
) -> None:
    """Writes a model output to disk.

    Each model output is written to disk as a zip archive.

    Args:
        benchmark_name: The benchmark name.
        model_output: The model output to save.
        output_dir: Directory to which to write the model output.
    """
    _output_dir = Path(output_dir)
    _output_dir.mkdir(exist_ok=True, parents=True)
    (_output_dir / benchmark_name).mkdir(exist_ok=True)

    data, arrays = dataclass_to_dict_with_arrays(model_output)

    with TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / MODEL_OUTPUT_JSON_FILENAME
        arrays_path = Path(tmpdir) / MODEL_OUTPUT_ARRAYS_FILENAME

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file)

        np.savez(arrays_path, **arrays)

        with ZipFile(
            _output_dir / benchmark_name / MODEL_OUTPUT_ZIP_FILENAME, "w"
        ) as zip_object:
            zip_object.write(json_path, os.path.basename(json_path))
            zip_object.write(arrays_path, os.path.basename(arrays_path))


def load_model_output_from_disk(
    model_outputs_dir: str | os.PathLike,
    benchmark_class: type[Benchmark],
) -> ModelOutput:
    """Loads a model output from disk.

    Args:
        model_outputs_dir: The path to the directory with the model_outputs.
        benchmark_class: The benchmark class that corresponds to the
                         benchmark to load from disk.

    Returns:
        The loaded model output.
    """
    _model_outputs_dir = Path(model_outputs_dir)
    benchmark_subdir = _model_outputs_dir / benchmark_class.name
    zip_to_load_path = benchmark_subdir / MODEL_OUTPUT_ZIP_FILENAME

    with ZipFile(zip_to_load_path, "r") as zip_object:
        with zip_object.open(MODEL_OUTPUT_JSON_FILENAME, "r") as json_file:
            json_data = json.load(json_file)
        with zip_object.open(MODEL_OUTPUT_ARRAYS_FILENAME, "r") as arrays_file:
            npz = np.load(arrays_file)
            arrays = {key: npz[key] for key in npz.files}

    model_output = dict_with_arrays_to_dataclass(
        json_data,
        arrays,
        benchmark_class.model_output_class,  # type: ignore
    )

    return model_output
