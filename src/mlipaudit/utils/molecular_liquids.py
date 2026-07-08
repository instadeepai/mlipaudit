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
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from mlip.simulation.enums import MDIntegrator

from mlipaudit.benchmark import DEFAULT_CHARGE, DEFAULT_SPIN
from mlipaudit.run_mode import RunMode
from mlipaudit.utils.simulation import run_simulation

logger = logging.getLogger("mlipaudit")

ANGSTROM3_TO_CM3 = 1e-24

DENSITY_RELATIVE_DEVIATION_THRESHOLD = 0.02
DENSITY_SCORE_ALPHA = 0.1

#: Reusable output IDs shared by the RDF and density benchmarks of each system group.
WATER_REUSABLE_OUTPUT_ID = ("water_molecular_liquid",)
SOLVENT_REUSABLE_OUTPUT_ID = ("solvent_molecular_liquid",)

# --------------------------------------------------------------------------------------
# Water
# --------------------------------------------------------------------------------------

#: Input-data directory name shared by the water RDF and water density benchmarks.
WATER_DATA_NAME = "water_radial_distribution"
WATERBOX_N500 = "water_box_n500_eq.pdb"
WATER_MOLECULE_INDICES_PATH = "water_box_n500_molecule_indices.npy"

WATER_MOLECULE_WEIGHT = 18.01528  # g/mol
WATER_ATOMS_PER_MOLECULE = 3
WATER_REFERENCE_DENSITY = 0.997773  # g/cm3

WATER_SIMULATION_CONFIG = {
    "num_steps": 500_000,
    "snapshot_interval": 500,
    "num_episodes": 1000,
    "temperature_kelvin": 295.15,
    "pressure_bar": 1.01325,
}
WATER_SIMULATION_CONFIG_FAST = {
    "num_steps": 250_000,
    "snapshot_interval": 250,
    "num_episodes": 1000,
    "temperature_kelvin": 295.15,
    "pressure_bar": 1.01325,
}
WATER_SIMULATION_CONFIG_DEV = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 295.15,
    "pressure_bar": 1.01325,
}

# --------------------------------------------------------------------------------------
# Solvents
# --------------------------------------------------------------------------------------

#: Input-data directory name shared by the solvent RDF and solvent density benchmarks.
SOLVENT_DATA_NAME = "solvent_radial_distribution"
NUM_DEV_SYSTEMS = 1

#: Per-solvent molecular weight (g/mol) and number of atoms per molecule.
SOLVENT_MOLECULE_CONFIG = {
    "CCl4": {"molecule_weight": 153.823, "atoms_per_molecule": 5},
    "methanol": {"molecule_weight": 32.042, "atoms_per_molecule": 6},
    "acetonitrile": {"molecule_weight": 41.053, "atoms_per_molecule": 6},
}
SOLVENT_REFERENCE_DENSITIES = {  # g/cm3
    "CCl4": 1.594,
    "acetonitrile": 0.786,
    "methanol": 0.791,
}

SOLVENT_SIMULATION_CONFIG = {
    "num_steps": 500_000,
    "snapshot_interval": 500,
    "num_episodes": 1000,
    "temperature_kelvin": 293.15,
    "pressure_bar": 1.01325,
}
SOLVENT_SIMULATION_CONFIG_FAST = {
    "num_steps": 250_000,
    "snapshot_interval": 250,
    "num_episodes": 1000,
    "temperature_kelvin": 293.15,
    "pressure_bar": 1.01325,
}
SOLVENT_SIMULATION_CONFIG_DEV = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 293.15,
    "pressure_bar": 1.01325,
}


def compute_densities(
    simulation_state: SimulationState, molecule_weight: float, atoms_per_molecule: int
) -> np.ndarray:
    """Compute the density (g/cm3) for each frame of the simulation.

    Args:
        simulation_state: The final simulation state.
        molecule_weight: Molecular weight of each solvent molecule.
        atoms_per_molecule: Number of atoms in each solvent molecule.

    Returns:
        densities: Computed density (g/cm3) for each frame of the simulation.
    """
    n_molecules = simulation_state.positions.shape[1] / atoms_per_molecule
    volumes = np.abs(np.linalg.det(simulation_state.cell))

    density_numerator = molecule_weight * n_molecules
    density_denominator = units.mol * volumes * ANGSTROM3_TO_CM3

    densities = density_numerator / density_denominator
    return densities


def average_equilibrated_density(densities: np.ndarray) -> float:
    """Average the density over the final four fifths of the frames.

    The first fifth of the trajectory is treated as equilibration and discarded.

    Args:
        densities: Per-frame densities (g/cm3).

    Returns:
        The average equilibrated density (g/cm3).
    """
    n_frames_equilibration = len(densities) // 5
    return float(np.mean(densities[n_frames_equilibration:]))


def _get_water_md_kwargs(run_mode: RunMode) -> dict[str, Any]:
    """Return the water simulation configuration for the given run mode."""
    if run_mode == RunMode.DEV:
        return WATER_SIMULATION_CONFIG_DEV
    if run_mode == RunMode.FAST:
        return WATER_SIMULATION_CONFIG_FAST
    return WATER_SIMULATION_CONFIG


def load_water_system(data_dir: str | os.PathLike) -> Atoms:
    """Load the water box structure from the given data directory.

    Returns:
        The water box as an `ase.Atoms` object with charge and spin set.
    """
    atoms = ase_read(Path(data_dir) / WATERBOX_N500)
    atoms.info["charge"] = DEFAULT_CHARGE
    atoms.info["spin"] = DEFAULT_SPIN
    return atoms


def _load_water_molecule_indices(data_dir: str | os.PathLike) -> np.ndarray:
    """Load the per-molecule atom indices used by the NPT barostat.

    Returns:
        The per-molecule atom indices.
    """
    return np.load(Path(data_dir) / WATER_MOLECULE_INDICES_PATH)


def run_water_npt_simulation(
    force_field: Any, data_dir: str | os.PathLike, run_mode: RunMode
) -> SimulationState | None:
    """Run the water box NPT simulation.

    The MD simulation is performed using the JAX MD engine and starts from the
    reference structure. The NPT integrator uses Langevin dynamics with a Monte Carlo
    barostat.

    Args:
        force_field: The force field to run the simulation with.
        data_dir: The directory holding the water input data.
        run_mode: The run mode controlling the simulation length.

    Returns:
        The final simulation state, or None if the simulation failed.
    """
    logger.info("Running water box NPT simulation.")
    return run_simulation(
        atoms=load_water_system(data_dir),
        force_field=force_field,
        md_integrator=MDIntegrator.NPT_MC_LANGEVIN,
        molecule_indices=_load_water_molecule_indices(data_dir),
        **_get_water_md_kwargs(run_mode),
    )


def _get_solvent_md_kwargs(run_mode: RunMode) -> dict[str, Any]:
    """Return the solvent simulation configuration for the given run mode."""
    if run_mode == RunMode.DEV:
        return SOLVENT_SIMULATION_CONFIG_DEV
    if run_mode == RunMode.FAST:
        return SOLVENT_SIMULATION_CONFIG_FAST
    return SOLVENT_SIMULATION_CONFIG


def get_solvent_system_names(run_mode: RunMode) -> list[str]:
    """Return the solvent system names to run for the given run mode.

    Returns:
        The solvent system names.
    """
    system_names = list(SOLVENT_MOLECULE_CONFIG.keys())
    if run_mode == RunMode.STANDARD:
        return system_names
    # reduced number of cases for DEV and FAST run mode
    return system_names[:NUM_DEV_SYSTEMS]


def get_solvent_pdb_file_name(system_name: str) -> str:
    """Return the PDB file name for a solvent system."""
    return f"{system_name}_eq.pdb"


def get_solvent_molecule_indices_file_name(system_name: str) -> str:
    """Return the molecule-indices file name for a solvent system."""
    return f"{system_name}_molecule_indices.npy"


def load_solvent_system(data_dir: str | os.PathLike, system_name: str) -> Atoms:
    """Load a solvent structure from the given data directory.

    Returns:
        The solvent structure as an `ase.Atoms` object with charge and spin set.
    """
    atoms = ase_read(Path(data_dir) / get_solvent_pdb_file_name(system_name))
    atoms.info["charge"] = DEFAULT_CHARGE
    atoms.info["spin"] = DEFAULT_SPIN
    return atoms


def _load_solvent_molecule_indices(
    data_dir: str | os.PathLike, system_name: str
) -> np.ndarray:
    """Load the per-molecule atom indices for a solvent system.

    Returns:
        The per-molecule atom indices for the given solvent system.
    """
    return np.load(Path(data_dir) / get_solvent_molecule_indices_file_name(system_name))


def run_solvent_npt_simulations(
    force_field: Any, data_dir: str | os.PathLike, run_mode: RunMode
) -> tuple[list[str], list[SimulationState | None]]:
    """Run one NPT simulation per solvent system.

    Args:
        force_field: The force field to run the simulations with.
        data_dir: The directory holding the solvent input data.
        run_mode: The run mode controlling the simulation length and system count.

    Returns:
        A tuple of (system names, simulation states) in matching order. A simulation
        state is None if the corresponding simulation failed.
    """
    system_names = get_solvent_system_names(run_mode)
    md_kwargs = _get_solvent_md_kwargs(run_mode)

    simulation_states: list[SimulationState | None] = []
    for system_name in system_names:
        logger.info("Running NPT simulation for %s.", system_name)
        simulation_state = run_simulation(
            atoms=load_solvent_system(data_dir, system_name),
            force_field=force_field,
            md_integrator=MDIntegrator.NPT_MC_LANGEVIN,
            molecule_indices=_load_solvent_molecule_indices(data_dir, system_name),
            **md_kwargs,
        )
        simulation_states.append(simulation_state)

    return system_names, simulation_states
