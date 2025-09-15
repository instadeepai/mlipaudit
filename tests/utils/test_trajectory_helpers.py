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

from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from mlip.simulation import SimulationState

from mlipaudit.utils import (
    create_ase_trajectory_from_simulation_state,
    create_mdtraj_trajectory_from_simulation_state,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "utils"


def test_create_mdtraj_trajectory_from_simulation_state():
    """Tests the creation of an mdtraj trajectory from a simulation state."""
    pdb_filepath = DATA_DIR / "two_alanines_capped.pdb"
    atoms = ase_read(pdb_filepath)

    traj_pos = np.stack([atoms.positions] * 25)
    state = SimulationState(
        atomic_numbers=atoms.numbers,
        positions=traj_pos,
        forces=np.zeros(shape=(25, len(atoms), 3)),
        temperature=np.zeros(25),
    )

    mdtraj_traj = create_mdtraj_trajectory_from_simulation_state(state, pdb_filepath)

    assert mdtraj_traj.n_frames == 25
    assert mdtraj_traj.n_atoms == len(atoms)
    assert mdtraj_traj.n_residues == 4
    assert mdtraj_traj.n_chains == 1
    assert mdtraj_traj.unitcell_lengths is None
    assert mdtraj_traj.unitcell_angles is None

    mdtraj_traj_with_box = create_mdtraj_trajectory_from_simulation_state(
        state, pdb_filepath, cell_lengths=(20.7, 20.7, 20.7)
    )

    traj_unitcell_lengths = [float(x) for x in mdtraj_traj_with_box.unitcell_lengths[0]]
    traj_unitcell_angles = [float(x) for x in mdtraj_traj_with_box.unitcell_angles[0]]

    assert all(np.isclose(traj_unitcell_lengths, [2.07, 2.07, 2.07]))
    assert all(np.isclose(traj_unitcell_angles, [90.0, 90.0, 90.0]))


def test_create_ase_trajectory_from_simulation_state():
    """Tests the creation of an ase trajectory from a simulation state."""
    state = SimulationState(
        atomic_numbers=[8, 6, 1, 1, 1],
        positions=np.full(shape=(25, 5, 3), fill_value=1.234),
        forces=np.zeros(shape=(25, 5, 3)),
        temperature=np.zeros(25),
    )

    traj = create_ase_trajectory_from_simulation_state(state)
    assert len(traj) == 25

    atoms = traj[0]
    assert str(atoms.symbols) == "OCH3"
    assert list(atoms.numbers) == state.atomic_numbers
    assert np.array_equal(atoms.positions, state.positions[0])
