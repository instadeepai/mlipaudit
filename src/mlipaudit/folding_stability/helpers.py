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
from pathlib import Path

import ase
import mdtraj
import numpy as np
import tmtools


def compute_tm_scores_and_rmsd_values(
    traj: mdtraj.Trajectory, ref_path: str | os.PathLike, stride: int = 1
) -> tuple[list[float], list[float]]:
    """Compute TM-scores and RMSD values for a trajectory.

    Computes TM-scores between each frame of a trajectory and a
    reference structure.

    Args:
        traj: The trajectory object from the `mdtraj` library.
        ref_path: Path to a reference PDB file.
        stride: Stride when moving through the trajectory frames. Default is 1.

    Returns:
        tm_scores: The TM-scores of the alignment.
        rmsd: The RMSDs of the alignment.
    """
    traj_ref = mdtraj.load(Path(ref_path))

    # get the amino acid sequences of the reference and the trajectory
    seq_ref = traj_ref.topology.to_fasta()[0]
    seq = traj.topology.to_fasta()[0]

    # Get the indices of the carbon alpha atoms of the reference and the trajectory
    carbon_alpha_indices_ref = traj_ref.topology.select("name CA")
    carbon_alpha_indices = traj.topology.select("name CA")

    # Get the coordinates of the carbon alpha atoms of the reference
    # (same reference point for the TM-score)
    coords_ref = traj_ref.xyz[0][carbon_alpha_indices_ref]

    # initialise the lists to store the TM-scores and the RMSDs
    tm_scores = []
    rmsds = []

    for frame in range(0, traj.n_frames, stride):
        # get the coordinates of the carbon alpha atoms of the trajectory
        coords = traj[frame].xyz[0][carbon_alpha_indices]
        # compute the TM-score and the RMSD
        results = tmtools.tm_align(
            coords_ref,
            coords,
            seq_ref,
            seq,
        )
        # append the TM-score and the RMSD to the lists
        tm_scores.append(results.tm_norm_chain2)
        rmsds.append(results.rmsd)

    return tm_scores, rmsds


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
    traj: mdtraj.Trajectory, ref_path: str | os.PathLike, simplified: bool = True
) -> np.ndarray:
    """Get the match secondary structure of the trajectory.

    Args:
        traj: The trajectory to use.
        ref_path: The path to the reference structure.
        simplified: Whether to use the simplified DSSP.

    Returns:
        An array containing the percentage of matches for each frame.
        Each value represents the percentage of residues that match the reference
        structure's secondary structure assignment for that frame.
    """
    dssp = mdtraj.compute_dssp(traj, simplified=simplified)
    traj_ref = mdtraj.load(ref_path)
    dssp_ref = mdtraj.compute_dssp(traj_ref, simplified=simplified)[0]

    # Calculate matches per frame by comparing each frame with reference
    matches = np.array([np.sum(frame == dssp_ref) for frame in dssp])
    return matches / dssp.shape[1]
