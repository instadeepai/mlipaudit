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

import ase
import mdtraj
import numpy as np
import tmtools
from ase import units

_NM_TO_ANGSTROM = units.nm / units.Angstrom


def assert_matching_topologies(traj: mdtraj.Trajectory, ref: mdtraj.Trajectory) -> None:
    """Assert that a trajectory and a reference structure share an atom ordering.

    The metrics in this module compare a trajectory against the experimental
    reference of the same molecule, so the atom correspondence is known 1:1 and
    is used directly rather than being re-derived.

    Args:
        traj: The trajectory object from the `mdtraj` library, with solvent
            already removed.
        ref: The reference structure, loaded with the `mdtraj` library.

    Raises:
        ValueError: If the two topologies do not describe the same atoms in the
            same order.
    """

    def _atom_keys(trajectory: mdtraj.Trajectory) -> list[tuple[int, str, str]]:
        return [
            (atom.residue.index, atom.residue.name, atom.name)
            for atom in trajectory.topology.atoms
        ]

    if _atom_keys(traj) != _atom_keys(ref):
        raise ValueError(
            "The trajectory and the reference structure do not describe the same "
            "atoms in the same order."
        )


def compute_ca_rmsd_values(
    traj: mdtraj.Trajectory, ref: mdtraj.Trajectory
) -> list[float]:
    """Compute the carbon alpha RMSD of each frame of a trajectory, in Angstrom.

    The RMSD is computed using the known 1:1 atom correspondence between the
    trajectory and the reference structure, under an RMSD-optimal (Kabsch)
    superposition. Note that a sequence aligner must not be used for this:
    since both structures are the same molecule, there is no correspondence to
    infer, and an aligner is free to drop badly displaced residues as gaps,
    which makes the result non-monotonic in the actual structural deviation.

    Args:
        traj: The trajectory object from the `mdtraj` library, with coordinates
            in nm as per the `mdtraj` convention.
        ref: The reference structure, loaded with the `mdtraj` library.

    Returns:
        The carbon alpha RMSD of each frame relative to the reference structure,
        converted to Angstrom.
    """
    carbon_alpha_indices = traj.topology.select("name CA")
    carbon_alpha_indices_ref = ref.topology.select("name CA")

    rmsd_values_nm = mdtraj.rmsd(
        traj,
        ref,
        frame=0,
        atom_indices=carbon_alpha_indices,
        ref_atom_indices=carbon_alpha_indices_ref,
    )

    return (rmsd_values_nm * _NM_TO_ANGSTROM).tolist()


def compute_tm_scores(
    traj: mdtraj.Trajectory, ref: mdtraj.Trajectory, stride: int = 1
) -> list[float]:
    """Compute the TM-score of each frame of a trajectory.

    Args:
        traj: The trajectory object from the `mdtraj` library, with coordinates
            in nm as per the `mdtraj` convention.
        ref: The reference structure.
        stride: Stride when moving through the trajectory frames. Default is 1.

    Returns:
        The TM-score of each frame relative to the reference structure.
    """
    # get the amino acid sequences of the reference and the trajectory
    seq_ref = ref.topology.to_fasta()[0]
    seq = traj.topology.to_fasta()[0]

    # Get the indices of the carbon alpha atoms of the reference and the trajectory
    carbon_alpha_indices_ref = ref.topology.select("name CA")
    carbon_alpha_indices = traj.topology.select("name CA")

    # Get the coordinates of the carbon alpha atoms of the reference
    # (same reference point for the TM-score)
    coords_ref = ref.xyz[0][carbon_alpha_indices_ref] * _NM_TO_ANGSTROM

    tm_scores = []

    for frame in range(0, traj.n_frames, stride):
        # get the coordinates of the carbon alpha atoms of the trajectory
        coords = traj.xyz[frame][carbon_alpha_indices] * _NM_TO_ANGSTROM
        results = tmtools.tm_align(
            coords_ref,
            coords,
            seq_ref,
            seq,
        )
        tm_scores.append(results.tm_norm_chain2)

    return tm_scores


def compute_radius_of_gyration_for_ase_atoms(atoms: ase.Atoms) -> float:
    """Compute the radius of gyration for ase Atoms.

    Args:
        atoms: The atoms representing a structure.

    Returns:
        The radius of gyration.
    """
    center_of_mass = atoms.get_center_of_mass()

    squared_distances = np.sum((atoms.positions - center_of_mass) ** 2, axis=1)

    mass = atoms.get_masses()
    sum_mass = np.sum(mass)
    sum_mass_squared_distances = np.sum(mass * squared_distances)

    radius_gyration = np.sqrt(sum_mass_squared_distances / sum_mass)

    return radius_gyration


def get_match_secondary_structure(
    traj: mdtraj.Trajectory, ref: mdtraj.Trajectory, simplified: bool = True
) -> np.ndarray:
    """Get the match secondary structure of the trajectory.

    Args:
        traj: The trajectory to use.
        ref: The reference structure, loaded with the `mdtraj` library.
        simplified: Whether to use the simplified DSSP.

    Returns:
        An array containing the percentage of matches for each frame.
        Each value represents the percentage of residues that match the reference
        structure's secondary structure assignment for that frame.
    """
    dssp = mdtraj.compute_dssp(traj, simplified=simplified)
    dssp_ref = mdtraj.compute_dssp(ref, simplified=simplified)[0]

    # Calculate matches per frame by comparing each frame with reference
    matches = np.array([np.sum(frame == dssp_ref) for frame in dssp])
    return matches / dssp.shape[1]
