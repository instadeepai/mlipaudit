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

"""Shared input systems and MD protocol for the biomolecule benchmarks."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator as ASECalculator
from ase.io import read as ase_read
from mlip.models import ForceField
from mlip.simulation import SimulationState

from mlipaudit.benchmark import DEFAULT_SPIN
from mlipaudit.run_mode import RunMode
from mlipaudit.utils.simulation import run_simulation
from mlipaudit.utils.stability import is_simulation_stable

logger = logging.getLogger("mlipaudit")

# Both biomolecule benchmarks read their input structures from this directory (see
# `Benchmark.data_name`), so the shared systems are stored on HuggingFace only once.
BIOMOLECULES_DATA_NAME = "folding_stability"

STRUCTURE_NAMES = [
    "chignolin_1uao_xray",
    "trp_cage_2jof_xray",
    "villin_capped_solvated",
]

BOX_SIZES = {
    "chignolin_1uao_xray": [23.98, 22.45, 20.68],
    "trp_cage_2jof_xray": [29.33, 29.74, 23.59],
    "villin_capped_solvated": [34.199, 34.199, 34.199],
}

STRUCTURE_CHARGES: dict[str, float] = {
    "chignolin_1uao_xray": -2.0,
    "trp_cage_2jof_xray": 0.0,
    "villin_capped_solvated": 2.0,
}

# Energy minimization is run with the JAX-MD FIRE minimizer (GPU-accelerated,
# force-only). The default ASE BFGS engine builds and eigendecomposes a dense
# (3N x 3N) Hessian every step, which is infeasible for the large solvated
# biomolecules in this benchmark (e.g. villin has ~3400 atoms).
MINIMIZATION_CONFIG = {
    "simulation_type": "minimization",
    "use_jax_md_minimization": True,
    "num_steps": 100,
    "snapshot_interval": 10,
    "temperature_kelvin": None,
    "timestep_fs": 0.1,
    # Ignored by the JAX-MD FIRE minimizer; used by the ASE BFGS fallback path.
    "max_force_convergence_threshold": 0.01,
}

MINIMIZATION_CONFIG_DEV = {
    "simulation_type": "minimization",
    "use_jax_md_minimization": True,
    "num_steps": 10,
    "snapshot_interval": 1,
    "temperature_kelvin": None,
    "timestep_fs": 0.1,
    # Ignored by the JAX-MD FIRE minimizer; used by the ASE BFGS fallback path.
    "max_force_convergence_threshold": 0.01,
}

SIMULATION_CONFIG = {
    "num_steps": 250_000,
    "snapshot_interval": 10_000,
    "num_episodes": 25,
    "temperature_kelvin": 300.0,
}

SIMULATION_CONFIG_DEV = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 300.0,
}

NUM_DEV_SYSTEMS = 1
NUM_FAST_SYSTEMS = 2


def get_structure_names(run_mode: RunMode) -> list[str]:
    """Return the systems to run for the given run mode.

    Returns:
        The subset of `STRUCTURE_NAMES` used by the run mode.
    """
    if run_mode == RunMode.DEV:
        return STRUCTURE_NAMES[:NUM_DEV_SYSTEMS]
    if run_mode == RunMode.FAST:
        return STRUCTURE_NAMES[:NUM_FAST_SYSTEMS]
    return STRUCTURE_NAMES


def _get_md_kwargs(run_mode: RunMode) -> dict[str, Any]:
    return SIMULATION_CONFIG_DEV if run_mode == RunMode.DEV else SIMULATION_CONFIG


def _get_minimization_kwargs(run_mode: RunMode) -> dict[str, Any]:
    return MINIMIZATION_CONFIG_DEV if run_mode == RunMode.DEV else MINIMIZATION_CONFIG


def assert_structure_names_in_model_output(
    structure_names: list[str], run_mode: RunMode
) -> None:
    """Assert that model output structure names are correct.

    The model outputs may have been transferred from another benchmark sharing the
    same `reusable_output_id`, so this guards against a mismatch in the run systems.
    """
    assert set(structure_names).issubset(STRUCTURE_NAMES)
    assert len(structure_names) == (
        NUM_DEV_SYSTEMS
        if run_mode == RunMode.DEV
        else (NUM_FAST_SYSTEMS if run_mode == RunMode.FAST else len(STRUCTURE_NAMES))
    )


def iter_biomolecule_simulations(
    force_field: ForceField | ASECalculator,
    data_dir: str | os.PathLike,
    run_mode: RunMode,
) -> Iterator[tuple[str, SimulationState | None]]:
    """Yield energy-minimized production MD for each biomolecule system in turn.

    For every system, an energy minimization (JAX-MD FIRE) is run first, the MD is
    seeded with the minimized coordinates, and the production MD is run. The systems
    and protocol are shared by the `folding_stability` and `sampling` benchmarks.
    Results are yielded per system so callers can populate their model output
    incrementally.

    Args:
        force_field: The force field to run the simulations with.
        data_dir: The directory holding the biomolecule input structures.
        run_mode: The run mode controlling the systems and simulation lengths.

    Yields:
        Each system's name and its final simulation state (`None` if that
        simulation failed), in the run-mode's system order.
    """
    md_kwargs = _get_md_kwargs(run_mode)
    minimization_kwargs = _get_minimization_kwargs(run_mode)

    for structure_name in get_structure_names(run_mode):
        logger.info("Running MD for %s", structure_name)

        atoms = ase_read(Path(data_dir) / f"{structure_name}.xyz")
        atoms.info["charge"] = float(STRUCTURE_CHARGES[structure_name])
        atoms.info["spin"] = DEFAULT_SPIN

        logger.info("Running energy minimization for %s", structure_name)
        minimization_state = run_simulation(
            atoms,
            force_field,
            box=BOX_SIZES[structure_name],
            **minimization_kwargs,
        )
        # The JAX-MD minimizer does not mutate the atoms in place, so seed the MD
        # with the minimized coordinates (final frame of the minimization).
        if minimization_state is None or not is_simulation_stable(minimization_state):
            logger.warning(
                "Energy minimization failed or was unstable for %s; running MD from "
                "the input structure",
                structure_name,
            )
        else:
            atoms.set_positions(np.asarray(minimization_state.positions[-1]))

        simulation_state = run_simulation(
            atoms, force_field, box=BOX_SIZES[structure_name], **md_kwargs
        )

        yield structure_name, simulation_state
