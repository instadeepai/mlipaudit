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

import mdtraj as md
import numpy as np
from ase import Atoms, units
from ase.io import write as ase_write
from mlip.simulation import SimulationState


def create_mdtraj_trajectory_from_simulation_state(
    simulation_state: SimulationState,
    topology_path: str | os.PathLike,
    cell_lengths: tuple[float, float, float] | None = None,
    cell_angles: tuple[float, float, float] = (90.0, 90.0, 90.0),
) -> md.Trajectory:
    """Create an mdtraj trajectory from a simulation state and topology.

    This function uses a temporary directory as it temporarily writes to disk in order
    to save the trajectory as an xyz file. All input values should be in Angstrom
    units. Note that the resulting trajectory uses nm as units.

    Args:
        simulation_state: The state containing the trajectory.
        topology_path: The path towards the topology file. Typically, a pdb file.
        cell_lengths: The lengths of the unit cell in Angstrom. Default is `None`.
        cell_angles: The angles of the unit cell in degrees. Default is `(90, 90, 90)`.

    Returns:
        The converted trajectory.
    """
    ase_traj = create_ase_trajectory_from_simulation_state(simulation_state)
    with tempfile.TemporaryDirectory() as tmpdir:
        _tmp_path = Path(tmpdir)
        ase_write(_tmp_path / "traj.xyz", ase_traj)
        traj = md.load(_tmp_path / "traj.xyz", top=topology_path)
        if cell_lengths is not None:
            # converting length units to nm for mdtraj
            cell_lengths_converted = [
                cell_length * (units.Angstrom / units.nm)
                for cell_length in cell_lengths
            ]
            traj.unitcell_lengths = np.tile(cell_lengths_converted, (traj.n_frames, 1))
            traj.unitcell_angles = np.tile(cell_angles, (traj.n_frames, 1))
    return traj


def create_ase_trajectory_from_simulation_state(
    simulation_state: SimulationState,
) -> list[Atoms]:
    """Create an ASE trajectory from the mlip library's simulation state.

    Note that the positions of the simulation state is in
    Angstrom, as are the positions of the resulting ASE
    trajectory.

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


def create_mdtraj_trajectory_from_positions(
    positions: np.ndarray, atom_symbols: list[str]
) -> md.Trajectory:
    """Load a simulation state into an MDTraj trajectory.

    Note that the units will therefore be converted
    from Angstrom to nm.

    Args:
        positions: Atomic positions from a simulation state.
        atom_symbols: List of atom symbols.

    Returns:
        trajectory: MDTraj trajectory.
    """
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("MOL", chain)

    for name in atom_symbols:
        topology.add_atom(
            name=name, element=md.element.get_by_symbol(name), residue=residue
        )

    if positions.ndim == 2:
        positions = positions.reshape(1, -1, 3)

    positions = positions * (units.Angstrom / units.nm)  # convert to nm
    trajectory = md.Trajectory(topology=topology, xyz=positions)

    return trajectory
