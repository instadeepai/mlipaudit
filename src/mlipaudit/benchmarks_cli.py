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
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ase.calculators.calculator import Calculator as ASECalculator
from mlip.models import ForceField, Mace, Nequip, Visnet
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.model_io import load_model_from_zip
from pydantic import ValidationError

from mlipaudit.benchmark import Benchmark, ModelOutput
from mlipaudit.exceptions import ModelOutputTransferError
from mlipaudit.io import (
    OVERALL_SCORE_KEY_NAME,
    generate_empty_scores_dict,
    write_benchmark_result_to_disk,
    write_scores_to_disk,
)
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_model_score

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


def fetch_missing_elements(
    benchmark_class: type[Benchmark], force_field: ForceField
) -> set:
    """Checks that we can run a force field on a certain benchmark,
    i.e. if it can handle the required element types.

    Args:
        benchmark_class: The benchmark class to check.
        force_field: The force field to check.

    Returns:
        A set of missing element types if the model cannot run the benchmark,
        otherwise an empty set.
    """
    if not benchmark_class.check_can_run_model(force_field):
        missing_element_types = benchmark_class.get_missing_element_types(force_field)
        return missing_element_types

    return set()


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


def _transfer_model_output(
    src_output: ModelOutput, target_output_class: type[ModelOutput]
) -> ModelOutput:
    """Transfers one model output, the source, to another model output class,
    the target.

    Args:
        src_output: The source model output instance.
        target_output_class: The target model output class.

    Returns:
        The instantiated target model output with the transferred content.

    Raises:
        ModelOutputTransferError: If transfer was not possible.

    """
    try:
        return target_output_class.model_validate(
            {
                f_name: getattr(src_output, f_name)
                for f_name in type(src_output).model_fields
            },
            extra="forbid",
        )
    except ValidationError:
        raise ModelOutputTransferError(
            "Requested model output transfer was impossible due to "
            "incompatibility of 'ModelOutput' classes."
        )


def run_benchmarks(
    model_paths: list[str],
    benchmarks_to_run: list[type[Benchmark]],
    run_mode: RunMode,
    output_dir: os.PathLike | str,
    data_input_dir: os.PathLike | str = "./data",
    verbose: bool = False,
    log_timings: bool = False,
) -> None:
    """Main for the MLIPAudit benchmark.

    Args:
        model_paths: List of model zip archive file paths.
        benchmarks_to_run: List of benchmarks to run.
        run_mode: The run mode in which to run the benchmarks.
        output_dir: The directory to which to save the results
            and scores.
        data_input_dir: The directory from which to load the input
            data. Defaults to `./data`.
        verbose: Whether to enable verbose logging from the `mlip`
            library. Defaults to False.
        log_timings: Whether to additionally log the time required to run
            each benchmark.

    Raises:
        ValueError: If specified model files do not have ending .py or .zip.
    """
    output_dir = Path(output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        force=True,
    )
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        mlip_logger = logging.getLogger("mlip")
        mlip_logger.setLevel(logging.WARNING)

    # Filter out known jax-md warnings
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

    # Get list of benchmark and model names
    benchmark_names = [b.name for b in benchmarks_to_run]
    model_names = [Path(model).stem for model in model_paths]

    logger.info(
        "Preparing to run %d %s (%s) across %d %s (%s).",
        len(benchmarks_to_run),
        "benchmarks" if len(benchmark_names) > 1 else "benchmark",
        ", ".join(benchmark_names),
        len(model_paths),
        "models" if len(model_paths) > 1 else "model, ".join(model_names),
        ", ".join(model_names),
    )

    skipped_benchmarks = defaultdict(list)

    for model_index, (model_to_run, model_name) in enumerate(
        zip(model_paths, model_names), 1
    ):
        logger.info(
            "--- [%d/%d] MODEL %s - Starting ---",
            model_index,
            len(model_paths),
            model_name,
        )

        force_field = load_force_field(model_to_run)

        reusable_model_outputs: dict[tuple[str, ...], ModelOutput] = {}
        scores = generate_empty_scores_dict()

        for benchmark_attempt_idx, benchmark_class in enumerate(benchmarks_to_run, 1):
            # First check we can run the benchmark with the model
            missing_elements = fetch_missing_elements(benchmark_class, force_field)
            if missing_elements:
                logger.info(
                    "[%d/%d] MODEL %s - [%d/%d] BENCHMARK %s - Skipped "
                    " due to missing elements %s.",
                    model_index,
                    len(model_paths),
                    model_name,
                    benchmark_attempt_idx,
                    len(benchmarks_to_run),
                    benchmark_class.name,
                    missing_elements,
                )
                skipped_benchmarks[model_name].append((
                    benchmark_class.name,
                    missing_elements,
                ))
                continue

            logger.info(
                "[%d/%d] MODEL %s - [%d/%d] BENCHMARK %s - Running...",
                model_index,
                len(model_paths),
                model_name,
                benchmark_attempt_idx,
                len(benchmarks_to_run),
                benchmark_class.name,
            )

            benchmark = benchmark_class(
                force_field=force_field,
                data_input_dir=data_input_dir,
                run_mode=run_mode,
            )

            reusable_output_id = benchmark.reusable_output_id
            if reusable_output_id and reusable_output_id in reusable_model_outputs:
                logger.info(
                    "[%d/%d] MODEL %s - [%d/%d] BENCHMARK %s - Loading in "
                    "model outputs from a previous benchmark...",
                    model_index,
                    len(model_paths),
                    model_name,
                    benchmark_attempt_idx,
                    len(benchmarks_to_run),
                    benchmark_class.name,
                )
                benchmark.model_output = _transfer_model_output(
                    reusable_model_outputs[reusable_output_id],
                    benchmark.model_output_class,  # type: ignore
                )

            else:
                try:
                    run_model_start = datetime.now()
                    benchmark.run_model()
                    run_model_end = datetime.now()
                    time_for_model_to_run = (
                        run_model_end - run_model_start
                    ).total_seconds()

                    if reusable_output_id is not None:
                        reusable_model_outputs[reusable_output_id] = (
                            benchmark.model_output  # type: ignore
                        )

                except Exception as e:
                    logger.error(
                        "Error running model %s on benchmark %s: %s",
                        model_name,
                        benchmark.name,
                        str(e),
                    )
                    continue

            # Analyze model outputs
            try:
                analysis_start = datetime.now()
                result = benchmark.analyze()
                analysis_end = datetime.now()
                time_for_analysis = (analysis_end - analysis_start).total_seconds()
            except Exception as e:
                logger.error(
                    "Error analyzing model %s on benchmark %s: %s",
                    model_name,
                    benchmark.name,
                    str(e),
                )
                continue

            if result.score is not None:
                scores[benchmark.name] = result.score
                logger.info(
                    "[%d/%d] MODEL %s - [%d/%d] BENCHMARK %s - Score: %.2f",
                    model_index,
                    len(model_paths),
                    model_name,
                    benchmark_attempt_idx,
                    len(benchmarks_to_run),
                    benchmark.name,
                    result.score,
                )

            write_benchmark_result_to_disk(
                benchmark_class.name, result, output_dir / model_name
            )
            logger.info(
                "Wrote benchmark result to disk at path %s.",
                output_dir / model_name / benchmark_class.name,
            )
            if log_timings:
                logger.info(
                    "[%d/%d] MODEL %s - [%d/%d] BENCHMARK %s -"
                    " Time for model to run: %.2fs",
                    model_index,
                    len(model_paths),
                    model_name,
                    benchmark_attempt_idx,
                    len(benchmarks_to_run),
                    benchmark.name,
                    time_for_model_to_run,
                )
                logger.info(
                    "[%d/%d] MODEL %s - [%d/%d] BENCHMARK %s -"
                    " Time for analysis: %.2fs",
                    model_index,
                    len(model_paths),
                    model_name,
                    benchmark_attempt_idx,
                    len(benchmarks_to_run),
                    benchmark.name,
                    time_for_analysis,
                )

        # Compute mean model score over all benchmarks
        model_score = compute_model_score(scores)

        logger.info(
            "--- [%d/%d] MODEL %s score"
            " (averaged over all available benchmarks): %.2f ---",
            model_index,
            len(model_paths),
            model_name,
            model_score,
        )

        # Also write the overall score to disk
        scores[OVERALL_SCORE_KEY_NAME] = model_score
        write_scores_to_disk(scores, output_dir / model_name)

        logger.info(
            "Wrote benchmark results and scores to disk at path %s.",
            output_dir / model_name,
        )

    # Log skipped benchmarks
    for model_name, skipped in skipped_benchmarks.items():
        for benchmark_name, missing_elements in skipped:
            logger.info(
                "Model %s skipped benchmark %s due to missing elements: %s",
                model_name,
                benchmark_name,
                missing_elements,
            )

    logger.info("Completed all benchmarks with all models.")
