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

import os
import tempfile
from pathlib import Path

import mdtraj
import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from mlip.simulation import SimulationState


def create_mdtraj_trajectory_from_simulation_state(
    simulation_state: SimulationState,
    topology_path: str | os.PathLike,
    cell_lengths: tuple[float, float, float] | None = None,
    cell_angles: tuple[float, float, float] = (90.0, 90.0, 90.0),
) -> mdtraj.Trajectory:
    """Create an mdtraj trajectory from a simulation state and topology.

    This function uses a temporary directory as it temporarily writes to disk in order
    to save the trajectory as an xyz file.

    Args:
        simulation_state: The state containing the trajectory.
        topology_path: The path towards the topology file. Typically, a pdb file.
        cell_lengths: The lengths of the unit cell. Default is `None`.
        cell_angles: The angles of the unit cell. Default is `(90, 90, 90)`.

    Returns:
        The converted trajectory.
    """
    ase_traj = create_ase_trajectory_from_simulation_state(simulation_state)
    with tempfile.TemporaryDirectory() as tmpdir:
        _tmp_path = Path(tmpdir)
        ase_write(_tmp_path / "traj.xyz", ase_traj)
        traj = mdtraj.load(_tmp_path / "traj.xyz", top=topology_path)
        if cell_lengths is not None:
            traj.unitcell_lengths = np.tile(cell_lengths, (traj.n_frames, 1))
            traj.unitcell_angles = np.tile(cell_angles, (traj.n_frames, 1))
    return traj


def create_ase_trajectory_from_simulation_state(
    simulation_state: SimulationState,
) -> list[Atoms]:
    """Create an ASE trajectory from the mlip library's simulation state.

    Args:
        simulation_state: The state containing the trajectory.

    Returns:
        An ASE trajectory as a list of `ase.Atoms`.
    """
    num_frames = simulation_state.positions.shape[0]
    trajectory = [
        Atoms(
            numbers=simulation_state.atomic_numbers,
            positions=simulation_state.positions[frame],
        )
        for frame in range(num_frames)
    ]
    return trajectory
