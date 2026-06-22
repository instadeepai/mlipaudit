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

import numpy as np
from ase import units
from mlip.simulation import SimulationState

ANGSTROM3_TO_CM3 = 1e-24


def compute_densities(
    simulation_state: SimulationState, molecule_weight: float, atoms_per_molecule: int
) -> np.ndarray:
    """Compute the density (g/cm3) for each frame of the simulation.

    Used by `WaterRadialDistributionFunction` and `SolventRadialDistributionFunction`.
    TODO: Should be moved into benchmark file when a density benchmark is created.

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
