"""This module contains run_model functions for benchmarks
which contain the same methods.
"""

import logging
from pathlib import Path

from ase.calculators.calculator import Calculator as ASECalculator
from ase.io import read as ase_read
from mlip.models import ForceField
from mlip.simulation import SimulationState

from mlipaudit.io import load_model_output_from_disk
from mlipaudit.utils import get_simulation_engine

logger = logging.getLogger("mlipaudit")

COMMON_BENCHMARKS = [["sampling", "folding_stability"]]


def _attempt_reuse_outputs(
    model_outputs_path, benchmark_class_to_load, benchmark_class_to_load_into
):
    if model_outputs_path is None:
        return None

    loaded_model_output = None
    try:
        fs_model_output = load_model_output_from_disk(
            model_outputs_path,
            benchmark_class=benchmark_class_to_load,
        )

        sampling_model_output = benchmark_class_to_load_into.model_output_class(
            structure_names=fs_model_output.structure_names,
            simulation_states=fs_model_output.simulation_states,
        )
        loaded_model_output = sampling_model_output

    except FileNotFoundError:
        logger.info(
            "Could not find or load reusable model output from %s."
            " Proceeding with MD simulation.",
            model_outputs_path,
        )

    except Exception as e:
        # Catch other potential loading/deserialization errors
        logger.error(
            "Error loading model output from %s: %s. Proceeding with MD simulation.",
            model_outputs_path,
            e,
        )

    return loaded_model_output


def run_biomolecules(
    structure_names: list[str],
    data_input_dir: Path,
    benchmark_name: str,
    force_field: ForceField | ASECalculator,
    box_sizes: dict[str, float] | dict[str, list[float]],
    **md_kwargs,
):
    """Common `run_model` method for `sampling` and `folding_stability`.

    Args:
        structure_names: The list of structure names.
        data_input_dir: The input directory towards the files.
        force_field: The force field to run MD with.
        benchmark_name: The name of the benchmark being run.
        box_sizes: The box sizes for the simulations.
        **md_kwargs: Additional keyword arguments passed to `run_model`.

    Returns:
        A dictionary containing a list of `structure_names` and a
            list of `simulation_states`.
    """
    model_output: dict[str, list[str] | list[SimulationState]] = {
        "structure_names": [],
        "simulation_states": [],
    }
    for structure_name in structure_names:
        logger.info("Running MD for %s", structure_name)
        xyz_filename = structure_name + ".xyz"
        atoms = ase_read(
            data_input_dir / benchmark_name / "starting_structures" / xyz_filename
        )

        md_engine = get_simulation_engine(
            atoms, force_field, box=box_sizes[structure_name], **md_kwargs
        )
        md_engine.run()

        final_state = md_engine.state
        model_output["structure_names"].append(structure_name)
        model_output["simulation_states"].append(final_state)

    return model_output
