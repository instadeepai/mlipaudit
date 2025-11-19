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

import mdtraj as md
import numpy as np
from mdtraj.core.topology import Residue
from scipy.spatial import KDTree
from scipy.stats import entropy


def calculate_multidimensional_distribution(
    points: np.ndarray,
    bins: int = 50,
    normalize: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute the normalized multidimensional histogram.

    Args:
        points: Array of points to analyze, shape (n_points, n_dimensions)
        bins: Number of bins per dimension (default: 50)
        normalize: Whether to normalize the histogram (default: True)

    Returns:
        Tuple containing:
            - Histogram array
            - List of bin edges for each dimension
    """
    ranges = [(-180, 180) for _ in range(points.shape[1])]

    hist, edges = np.histogramdd(points, bins=bins, range=ranges)
    if normalize:
        hist_normalized = hist / np.sum(hist)
        return hist_normalized, edges

    return hist, edges


def calculate_distribution_rmsd(
    hist1: np.ndarray, hist2: np.ndarray, normalize: bool = True
) -> float:
    """Calculate RMSD between two distributions.

    Args:
        hist1: First histogram array
        hist2: Second histogram array
        normalize: Whether to normalize histograms before comparison
            (default: True)

    Returns:
        float: Root mean square deviation between the distributions

    Raises:
        ValueError: If the histograms have different shapes.
    """
    if hist1.shape != hist2.shape:
        raise ValueError("Histograms must have the same shape")

    if normalize:
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)

    return np.sqrt(np.mean((hist1 - hist2) ** 2))


def calculate_distribution_kl_divergence(
    reference_hist: np.ndarray,
    sampled_hist: np.ndarray,
    normalize: bool = True,
) -> float:
    """Calculate KL divergence between two distributions.

    Note: The distributions are flattened prior to the calculation of the KL divergence.

    Args:
        reference_hist: Reference histogram array
        sampled_hist: Sampled histogram array
        normalize: Whether to normalize histograms before comparison

    Raises:
        ValueError: If the histograms have different shapes.

    Returns:
        float: KL divergence between the two distributions
    """
    if reference_hist.shape != sampled_hist.shape:
        raise ValueError("Histograms must have the same shape")

    if normalize:
        reference_hist = reference_hist / np.sum(reference_hist)
        sampled_hist = sampled_hist / np.sum(sampled_hist)

    reference_hist = reference_hist.flatten()
    sampled_hist = sampled_hist.flatten()

    return entropy(pk=reference_hist, qk=sampled_hist)


def calculate_distribution_hellinger_distance(
    reference_hist: np.ndarray,
    sampled_hist: np.ndarray,
    normalize: bool = True,
) -> float:
    """Calculate Hellinger distance between two distributions.

    Args:
        reference_hist: Reference histogram array
        sampled_hist: Sampled histogram array
        normalize: Whether to normalize histograms before comparison (only
            set this to False if the histograms are already normalized).

    Raises:
        ValueError: If the histograms have different shapes.

    Returns:
        The Hellinger distance between the two distributions
    """
    if reference_hist.shape != sampled_hist.shape:
        raise ValueError("Histograms must have the same shape")

    if normalize:
        reference_hist = reference_hist / np.sum(reference_hist)
        sampled_hist = sampled_hist / np.sum(sampled_hist)

    reference_hist_sqrt = np.sqrt(reference_hist)
    sampled_hist_sqrt = np.sqrt(sampled_hist)

    b_coeff = np.sum(reference_hist_sqrt * sampled_hist_sqrt)

    return np.sqrt(max(0.0, 1.0 - b_coeff))


def get_all_dihedrals_from_trajectory(
    traj: md.Trajectory,
    only_backbone: bool = False,
) -> dict[Residue, dict[str, np.ndarray]]:
    """Get all dihedrals from a trajectory.

    Args:
        traj: The trajectory to analyze.
        only_backbone: Whether to only return backbone dihedrals (default: False).

    Returns:
        dict[Residue, dict[str, np.ndarray]]: A dictionary of residues
        and their dihedrals. E.g.
        {
            Residue(1, "ALA"): {
                "phi": np.array([10.0, 20.0, 30.0]),
                "psi": np.array([40.0, 50.0, 60.0]),
            }
        }
    """
    dihedral_functions = {
        "phi": md.compute_phi,
        "psi": md.compute_psi,
    }

    if not only_backbone:
        dihedral_functions.update({
            "chi1": md.compute_chi1,
            "chi2": md.compute_chi2,
            "chi3": md.compute_chi3,
            "chi4": md.compute_chi4,
            "chi5": md.compute_chi5,
        })

    dihedrals: dict[Residue, dict[str, np.ndarray]] = {}

    for dihedral_name, dihedral_function in dihedral_functions.items():
        # atom_indices has shape (n_dihedrals, 4)
        # angles_rad has shape (n_frames, n_dihedrals)
        atom_indices, angles_rad = dihedral_function(traj)
        residues = [traj.top.atom(ids[1]).residue for ids in atom_indices]
        angles_deg = np.degrees(angles_rad)

        for i, residue in enumerate(residues):
            if residue not in dihedrals:
                dihedrals[residue] = {}
            dihedrals[residue][dihedral_name] = angles_deg[:, i]

    # Drop residues which don't contain both backbone dihedrals phi and psi
    filtered_dihedrals = {}
    for residue, dihedrals in dihedrals.items():
        if not ("phi" in dihedrals) ^ ("psi" in dihedrals):
            filtered_dihedrals[residue] = dihedrals

    return filtered_dihedrals


def identify_outlier_data_points(
    sampled_dihedrals: np.ndarray,
    reference_dihedrals: np.ndarray,
    threshold: float = 10.0,
    period: float = 360.0,
) -> list[bool]:
    """Identify outlier data points in a sampled dihedral distribution.

    Args:
        sampled_dihedrals: Sampled dihedrals. Has shape (n_frames, n_dihedrals).
        reference_dihedrals: Reference dihedrals. Has shape (n_frames, n_dihedrals).
        threshold: Threshold for identifying outlier data points (default: 10.0).
        period: Period of the dihedral angle (default: 360.0).

    Returns:
        list[bool]: A list of length n_frames indicating whether each data point
        is an outlier.
    """
    sampled_dihedrals = sampled_dihedrals % period
    reference_dihedrals = reference_dihedrals % period

    n_dihedrals = sampled_dihedrals.shape[1]
    box_sizes = np.full(n_dihedrals, period)

    tree = KDTree(reference_dihedrals, boxsize=box_sizes)
    distances, _ = tree.query(sampled_dihedrals, k=1)

    return list(distances > threshold)
