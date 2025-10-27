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
import os
import runpy
import statistics
import warnings
from pathlib import Path

from ase.calculators.calculator import Calculator as ASECalculator
from mlip.models import ForceField, Mace, Nequip, Visnet
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.model_io import load_model_from_zip

from mlipaudit.benchmark import Benchmark
from mlipaudit.io import (
    write_benchmark_result_to_disk,
    write_scores_to_disk,
)
from mlipaudit.run_mode import RunMode

logger = logging.getLogger("mlipaudit")

EXTERNAL_MODEL_VARIABLE_NAME = "mlipaudit_external_model"


def _model_class_from_name(model_name: str) -> type[MLIPNetwork]:
    if "visnet" in model_name:
        return Visnet
    if "mace" in model_name:
        return Mace
    if "nequip" in model_name:
        return Nequip
    raise NotImplementedError(
        "Name of model zip archive does not contain info about the type of MLIP model."
    )


def _can_run_model_on_benchmark(
    benchmark_class: type[Benchmark], force_field: ForceField
) -> bool:
    """Checks that we can run a force field on a certain benchmark,
    i.e. if it can handle the required element types, logging
    whether the benchmark can run or not.

    Returns:
        Whether the benchmark can run or not.
    """
    if not benchmark_class.check_can_run_model(force_field):
        missing_element_types = benchmark_class.get_missing_element_types(force_field)
        logger.info(
            "Skipping benchmark %s due to missing element types: %s",
            benchmark_class.name,
            missing_element_types,
        )
        return False

    return True


def _load_external_model(py_file: str) -> ASECalculator | ForceField:
    """Loads an external model from a specified Python file.

    This is either an ASE calculator or a `ForceField` object.

    Args:
        py_file: The location of the Python file to load the model from.

    Returns:
        The loaded ASE calculator or force field instance.

    Raises:
        ImportError: If external model not found in file.
        ValueError: If external model found in file has wrong type.
    """
    globals_dict = runpy.run_path(py_file)
    if EXTERNAL_MODEL_VARIABLE_NAME not in globals_dict:
        raise ImportError(
            f"{EXTERNAL_MODEL_VARIABLE_NAME} not found in specified .py file."
        )

    is_ase_calc = isinstance(globals_dict[EXTERNAL_MODEL_VARIABLE_NAME], ASECalculator)
    is_mlip_ff = isinstance(globals_dict[EXTERNAL_MODEL_VARIABLE_NAME], ForceField)
    if not (is_ase_calc or is_mlip_ff):
        raise ValueError(
            f"{EXTERNAL_MODEL_VARIABLE_NAME} must be either of type ASE "
            f"calculator or of the mlip library's 'ForceField' type."
        )

    return globals_dict[EXTERNAL_MODEL_VARIABLE_NAME]


def load_force_field(model: str) -> ASECalculator | ForceField:
    """Loads a force field from a specified model file.

    This is either an ASE calculator or a `ForceField` object.

    Args:
        model: The location of the model file to load the model from.

    Returns:
        The loaded ASE calculator or force field instance.

    Raises:
        ValueError: If model file does not have ending .py or .zip.
    """
    model_name = Path(model).stem
    if Path(model).suffix == ".zip":
        model_class = _model_class_from_name(model_name)
        force_field = load_model_from_zip(model_class, model)
    elif Path(model).suffix == ".py":
        force_field = _load_external_model(model)
    else:
        raise ValueError("Model arguments must be .zip or .py files.")

    return force_field


def run_benchmarks(
    model_paths: list[str],
    benchmarks_to_run: list[type[Benchmark]],
    run_mode: RunMode,
    output_dir: os.PathLike | str,
    data_input_dir: os.PathLike | str,
):
    """Main for the MLIPAudit benchmark.

    Raises:
        ValueError: If specified model files do not have ending .py or .zip.
    """
    output_dir = Path(output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        force=True,
    )
    logger.setLevel(logging.INFO)

    warnings.filterwarnings(
        "ignore",
        message="Explicitly requested dtype .* requested in sum is not available,"
        " and will be truncated to dtype float32.",
        category=UserWarning,
        module="jax._src.numpy.reductions",
    )
    warnings.filterwarnings(
        "ignore",
        message="None encountered in jnp.array(); this is currently treated as NaN."
        " In the future this will result in an error.",
        category=FutureWarning,
        module="jax._src.numpy.lax_numpy",
    )

    logger.info(
        "Will run the following benchmarks: %s",
        ", ".join([b.name for b in benchmarks_to_run]),
    )

    for model in model_paths:
        model_name = Path(model).stem
        logger.info("Running benchmarks for model %s.", model_name)

        force_field = load_force_field(model)

        scores = {}
        for benchmark_class in benchmarks_to_run:
            if not _can_run_model_on_benchmark(benchmark_class, force_field):
                continue

            logger.info("Running benchmark %s.", benchmark_class.name)

            benchmark = benchmark_class(
                force_field=force_field,
                data_input_dir=data_input_dir,
                run_mode=run_mode,
            )
            benchmark.run_model()
            result = benchmark.analyze()

            if result.score is not None:
                scores[benchmark.name] = result.score
                logger.info("Benchmark %s score: %.2f", benchmark.name, result.score)

            write_benchmark_result_to_disk(
                benchmark_class.name, result, output_dir / model_name
            )
            logger.info(
                "Wrote benchmark result to disk at path %s.",
                output_dir / model_name / benchmark_class.name,
            )

        # Compute model score here with results
        model_score = statistics.mean(scores.values())
        scores["overall_score"] = model_score
        logger.info("Model score: %.2f", model_score)

        write_scores_to_disk(scores, output_dir / model_name)
        logger.info(
            "Wrote benchmark results and scores to disk at path %s.",
            output_dir / model_name,
        )

    logger.info("Completed all benchmarks with all models.")
