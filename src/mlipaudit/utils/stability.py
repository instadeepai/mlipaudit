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
from jax import numpy as jnp
from mlip.simulation import SimulationState
from scipy.spatial.distance import pdist, squareform

HYDROGEN_BOND_CUTOFF_ANGSTROM = 2.5


def is_frame_stable(
    positions: np.ndarray, cutoff: float = HYDROGEN_BOND_CUTOFF_ANGSTROM
) -> bool:
    """Check if a position in a simulation is stable or whether at least
    one atom has drifted beyond the cutoff.

    Args:
        positions: The positions of the atoms.
        cutoff: If an atom's distance to all other atoms
            exceeds the cutoff, the frame will be flagged
            as unstable. The unit is Angstrom. Defaults to 2.5

    Returns:
        Whether the position is stable or not.
    """
    pairwise_distances = pdist(positions, metric="euclidean")
    pairwise_distances_squareform = squareform(pairwise_distances)
    exceed_distance = pairwise_distances_squareform > cutoff
    np.fill_diagonal(exceed_distance, True)  # As diag all 0s
    any_drifting = np.all(exceed_distance, axis=0)
    return not np.any(any_drifting)


def is_simulation_stable(
    simulation_state: SimulationState, temperature: float = 300.0
) -> bool:
    """Check if a simulation exploded or not, by looking
    at the positions of the final frame.

    Args:
        simulation_state: The final simulation state.
        temperature: The temperature at which the simulation was run.

    Returns:
        Whether the simulation was stable.
    """
    # Checks for NaNs
    if np.isnan(simulation_state.positions).any():
        return False

    # Check temp if we have access, i.e. MD and not minimization
    if (
        simulation_state.temperature is not None
        and find_explosion_frame(simulation_state, temperature) > -1
    ):
        return False

    # Check the last frame's atom positions
    return is_frame_stable(simulation_state.positions[-1])


def find_explosion_frame(simulation_state: SimulationState, temperature: float) -> int:
    """Find the frame where a simulation exploded or return -1.

    Given a trajectory and the temperature at which it was run, assuming that it
    used a constant schedule, checks whether the simulation exploded by seeing if
    the temperature increases dramatically.

    Args:
        simulation_state: The state containing the trajectory.
        temperature: The constant temperature at which the simulation was run.

    Returns:
        The frame at which the simulation exploded or -1 if it remained stable.

    Raises:
        ValueError: If the simulation state does not contain temperature information.
    """
    if simulation_state.temperature is None:
        raise ValueError("Simulation state does not contain temperature information.")

    temperatures = simulation_state.temperature
    threshold = temperature + 10_000.0

    exceed_indices = jnp.nonzero(temperatures > threshold)[0]
    if exceed_indices.shape[0] > 0:
        return int(exceed_indices[0])

    jump_indices = jnp.nonzero(temperatures[1:] > 100.0 * temperatures[:-1])[0]
    if jump_indices.shape[0] > 0:
        return int(jump_indices[0] + 1)

    return -1
