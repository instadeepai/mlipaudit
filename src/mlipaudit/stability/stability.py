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
import functools
import logging
from typing import TypedDict

import jax.numpy as jnp
import mdtraj
import numpy as np
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from mlip.simulation.configs import JaxMDSimulationConfig
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.utils import create_mdtraj_trajectory_from_simulation_state

logger = logging.getLogger("mlipaudit")

SIMULATION_CONFIG = {
    "num_steps": 100_000,
    "snapshot_interval": 100,
    "num_episodes": 100,
    "temperature_kelvin": 300.0,
}

SIMULATION_CONFIG_FAST = {
    "num_steps": 10,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 300.0,
}

TEMPERATURE_THRESHOLD = 10_000


class StructureMetadata(TypedDict):
    """Docstring."""

    xyz: str
    pdb: str
    description: str


STRUCTURES: dict[str, StructureMetadata] = {
    # Small proteins/peptides
    "1JRS_Leupeptin": {
        "xyz": "71_1jrs_leupeptin.xyz",
        "pdb": "71_1jrs_leupeptin.pdb",
        "description": "Leupeptin inhibitor (71 atoms)",
    },
    "Chignolin": {
        "xyz": "138_1uao_chignolin.xyz",
        "pdb": "138_1uao_chignolin.pdb",
        "description": "Chignolin peptide (138 atoms)",
    },
    "RNA fragment": {
        "xyz": "168_1p79_RNA.xyz",
        "pdb": "168_1p79_RNA.pdb",
        "description": "RNA fragment (168 atoms)",
    },
    # Medium proteins
    "5KGZ": {
        "xyz": "634_5kgz.xyz",
        "pdb": "634_5kgz.pdb",
        "description": "Protein structure (634 atoms)",
    },
    "1AB7": {
        "xyz": "1432_1ab7.xyz",
        "pdb": "1432_1ab7.pdb",
        "description": "Protein structure (1,432 atoms)",
    },
    "1BIP": {
        "xyz": "1818_1bip.xyz",
        "pdb": "1818_1bip.pdb",
        "description": "Protein structure (1,818 atoms)",
    },
    "1A5E": {
        "xyz": "2301_1a5e.xyz",
        "pdb": "2301_1a5e.pdb",
        "description": "Protein structure (2,301 atoms)",
    },
    "1A7M": {
        "xyz": "2803_1a7m.xyz",
        "pdb": "2803_1a7m.pdb",
        "description": "Protein structure (2,803 atoms)",
    },
    "2BQV": {
        "xyz": "3346_2bqv.xyz",
        "pdb": "3346_2bqv.pdb",
        "description": "Protein structure (3,346 atoms)",
    },
}

STRUCTURE_NAMES = list(STRUCTURES.keys())


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
    """
    temperatures = simulation_state.temperature
    threshold = temperature + 10_000.0

    exceed_indices = jnp.nonzero(temperatures > threshold)[0]
    if exceed_indices.shape[0] > 0:
        return int(exceed_indices[0])

    jump_indices = jnp.nonzero(temperatures[1:] > 100.0 * temperatures[:-1])[0]
    if jump_indices.shape[0] > 0:
        return int(jump_indices[0] + 1)

    return -1


def find_heavy_to_hydrogen_starting_bonds(
    traj: mdtraj.Trajectory, solvents=None
) -> np.ndarray:
    """Find all initial bonds between heavy atoms and hydrogen atoms in a trajectory.

    Exclude bonds involving solvent molecules. Computes the bonds
    from the starting frame and effectively ignores the rest of the trajectory.

    Args:
        traj: The trajectory to analyze.
        solvents: The names of solvent molecules to ignore.
         Defaults to ["WAT", "HOH"].

    Returns:
        An array of shape (npairs, 2) where each row is of the form
        (heavy_atom_index, hydrogen_atom_index) representing
        bonds where the first atom is a non-hydrogen atom and the second is a
        hydrogen atom.
    """
    if solvents is None:
        solvents = ["WAT", "HOH"]
    bonds = []
    for bond in traj.topology.bonds:
        res1, res2 = bond.atom1.residue.name, bond.atom2.residue.name
        elem1, elem2 = bond.atom1.element.symbol, bond.atom2.element.symbol

        if res1 in solvents or res2 in solvents:
            continue

        if elem1 == "H" and elem2 != "H":
            bonds.append((bond.atom2.index, bond.atom1.index))
        elif elem1 != "H" and elem2 == "H":
            bonds.append((bond.atom1.index, bond.atom2.index))

    return np.array(bonds)


def find_first_broken_frames_hydrogen_exchange(
    traj: mdtraj.Trajectory, cutoff: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Find the first frames where proton bonds are broken.

    Given a trajectory, first finds all the heavy-to-hydrogen bonds.
    Then computes the distances between the heavy atoms and the hydrogen
    atoms for each bond for each frame. If the distance is greater than
    the cutoff, the bond is considered broken. Returns the frames at which
    the bonds are broken
    and the bond index.

    Args:
        traj: The trajectory to analyze.
        cutoff: The cutoff in nanometers. Defaults to 0.2.

    Returns:
        A tuple of two arrays (frames, bonds). The first
        is of shape (nbrokenbonds,) where each element corresponds
        to the frame index at which the bond was broken. The second
        is of shape (nbrokenbonds, 2) where each row is of the
        form (heavy_atom_index, hydrogen_atom_index).
        Therefore, the triplet ``(frames[i], bonds[i][0], bonds[i][1])``
        tells you at which frame a bond is broken.
    """
    heavy_to_hydrogen_bonds = find_heavy_to_hydrogen_starting_bonds(traj)
    bond_distances = mdtraj.compute_distances(
        traj, heavy_to_hydrogen_bonds
    )  # (nframes, nbonds)
    broken_bonds = bond_distances > cutoff  # (nframes, nbonds)
    any_break = np.any(broken_bonds, axis=0)  # (nbonds,)

    first_broken_frames = np.argmax(broken_bonds, axis=0)  # (nbonds,)
    # Mask those which never break
    first_broken_frames = np.where(any_break, first_broken_frames, -1)  # (nbonds,)

    # Discard bonds which don't break
    first_broken_frames = np.where(first_broken_frames != -1)

    return first_broken_frames, heavy_to_hydrogen_bonds[first_broken_frames]


def find_first_drifting_frames(
    drifting_hydrogens_by_frame: np.ndarray[bool],
) -> np.ndarray:
    """Find the first frames where hydrogens drift away.

    First accumulates the boolean values in a row-wise fashion
    on the reversed array. Then finds the first True for each row.

    Args:
        drifting_hydrogens_by_frame: A boolean array of shape (nframes, nhydrogens).

    Returns:
        An array of shape (nhydrogens,) where each element corresponds to
        the frame index at which the protons started drifting. If a hydrogen
        does not drift, then `nframes` is returned instead.
    """
    nframes, nhydrogens = drifting_hydrogens_by_frame.shape
    suffix_all_true = np.logical_and.accumulate(
        drifting_hydrogens_by_frame[::-1, :],
        axis=0,  # Accumulate over frames
    )[::-1, :]
    first_true_indices = suffix_all_true.argmax(axis=0)

    # Check which columns actually contain at least one True value.
    has_true_in_column = np.any(suffix_all_true, axis=0)

    # Initialize the result array with a sentinel value (nframes)
    # for columns that might not have any True values.
    result = np.full(nhydrogens, nframes, dtype=int)

    # For columns that do have at least one True, update the result
    # with the indices found by argmax.
    result[has_true_in_column] = first_true_indices[has_true_in_column]

    return result


def detect_hydrogen_drift(
    traj: mdtraj.Trajectory, cutoff: float = 0.2
) -> tuple[int, int]:
    """Detect whether hydrogens are drifting away from a system.

    Given a trajectory, first finds all the heavy-to-hydrogen bonds.
    Then computes the distances between the heavy atoms and the hydrogen
    atoms for each bond for each frame. If the distance is greater than
    the cutoff, the bond is considered broken. Then computes the distance
    between all heavy atoms and all hydrogen atoms that break their bond
    at some point. If the distance to all the heavy atoms is greater than
    the cutoff for a hydrogen, it is considered drifting. Returns the frames
    at which the hydrogens start drifting for the remainder of the trajectory.

    Args:
        traj: The trajectory to analyze.
        cutoff: The cutoff in nanometers to consider a bond broken
            and the distance to exceed to all heavy atoms to be considered drifting.
            Defaults to 0.2.

    Returns:
        A tuple of (frame_index, hydrogen_index), corresponding to the first
        frame at which a hydrogen atom drifts. If no drifting, returns (-1, -1).
    """
    heavy_to_hydrogen_bonds = find_heavy_to_hydrogen_starting_bonds(traj)
    bond_distances = mdtraj.compute_distances(
        traj, heavy_to_hydrogen_bonds
    )  # (nframes, nbonds)
    broken_bonds = bond_distances > cutoff  # (nframes, nbonds)
    any_break = np.any(broken_bonds, axis=0)  # (nbonds,)

    if not np.any(any_break):
        return -1, -1

    # Array of all hydrogen indices which break their bond at some point
    broken_hydrogen_indices = heavy_to_hydrogen_bonds[any_break][:, 1]

    # Compute distances between heavy atoms and relevant hydrogens
    heavy_atoms = traj.top.select("! symbol H")
    all_heavy_to_broken_hydrogen_indices = np.array([
        (heavy_index, hydrogen_index)
        for hydrogen_index in broken_hydrogen_indices
        for heavy_index in heavy_atoms
    ])  # (nhydrogen * nheavy, 2)
    heavy_to_broken_hydrogen_distances = mdtraj.compute_distances(
        traj, all_heavy_to_broken_hydrogen_indices
    )  # (nframes, nhydrogen * nheavy)

    # Reshape so that the second dimension corresponds to hydrogens
    heavy_to_broken_hydrogen_distances = heavy_to_broken_hydrogen_distances.reshape(
        -1, len(broken_hydrogen_indices), len(heavy_atoms)
    )  # (nframes, nhydrogen, nheavy)

    # See if distance exceeded to all other heavy atoms
    distances_cutoff = (
        heavy_to_broken_hydrogen_distances > cutoff
    )  # (nframes, nhydrogen, nheavy)
    drifting_hydrogens_by_frame = np.all(
        distances_cutoff, axis=2
    )  # (nframes, nhydrogens)

    first_drifting_frames = find_first_drifting_frames(
        drifting_hydrogens_by_frame
    )  # (nhydrogens,)

    if np.all(first_drifting_frames == drifting_hydrogens_by_frame.shape[0]):
        return -1, -1

    first_drifting_frame = np.min(first_drifting_frames)
    first_drifting_hydrogen = np.argmin(  # only fetches one - could be many
        first_drifting_frames
    )
    first_drifting_hydrogen_index = broken_hydrogen_indices[first_drifting_hydrogen]

    return int(first_drifting_frame), int(first_drifting_hydrogen_index)


class StabilityStructureResult(BaseModel):
    """Docstring."""

    structure_name: str
    description: str
    num_frames: PositiveInt
    num_steps: PositiveInt
    exploded_frame: int
    drift_frame: int
    score: float = Field(ge=0, le=1)


class StabilityResult(BenchmarkResult):
    """Docstring."""

    structure_results: list[StabilityStructureResult]


class StabilityModelOutput(ModelOutput):
    """Docstring.

    Attributes:
        structure_names: The list of structure names.
        simulation_states: The list of final simulation states.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure_names: list[str]
    simulation_states: list[SimulationState]


class StabilityBenchmark(Benchmark):
    """Benchmark for running stability tests.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `stability`.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            ``StabilityResult``.
    """

    name = "stability"
    result_class = StabilityResult

    def run_model(self) -> None:
        """Run MD for each structure.

        The simulation results are stored in the `model_output` attribute.
        """
        self.model_output = StabilityModelOutput(
            structure_names=[],
            simulation_states=[],
        )
        structure_names = STRUCTURE_NAMES[:2] if self.fast_dev_run else STRUCTURE_NAMES
        for structure_name in structure_names:
            logger.info("Running MD for %s", structure_name)
            xyz_filename = STRUCTURES[structure_name]["xyz"]
            atoms = ase_read(self.data_input_dir / self.name / xyz_filename)

            md_engine = JaxMDSimulationEngine(atoms, self.force_field, self._md_config)
            md_engine.run()

            final_state = md_engine.state

            self.model_output.structure_names.append(structure_name)
            self.model_output.simulation_states.append(final_state)

    def analyze(self) -> StabilityResult:
        """Checks whether the trajectories exploded.

        Loads the trajectory from the simulation state and first
        checks whether the trajectory exploded. If not, then loads
        in the corresponding pdb file to access bond
        information and checks whether hydrogens are drifting.

        Returns:
            A `StabilityResult` object with the benchmark results.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        structure_results = []
        for structure_name, simulation_state in zip(
            self.model_output.structure_names, self.model_output.simulation_states
        ):
            explosion_frame = find_explosion_frame(
                simulation_state, self._md_config.temperature_kelvin
            )

            structure_result = {
                "structure_name": structure_name,
                "description": STRUCTURES[structure_name]["description"],
                "num_frames": simulation_state.positions.shape[0],
                "num_steps": self._md_config.num_steps,
                "exploded_frame": explosion_frame,
                "drift_frame": -1,
            }
            if explosion_frame == -1:
                # Check for H drift
                topology_file_name = STRUCTURES[structure_name]["pdb"]
                traj = create_mdtraj_trajectory_from_simulation_state(
                    simulation_state,
                    topology_path=self.data_input_dir / self.name / topology_file_name,
                )
                first_drifting_frame, first_drifting_hydrogen_index = (
                    detect_hydrogen_drift(traj)
                )
                structure_result["drift_frame"] = first_drifting_frame

            structure_result["score"] = self._calculate_score(
                structure_result["drift_frame"],
                structure_result["exploded_frame"],
                structure_result["num_frames"],
            )
            structure_results.append(StabilityStructureResult(**structure_result))

        return StabilityResult(structure_results=structure_results)

    @functools.cached_property
    def _md_config(self) -> JaxMDSimulationConfig:
        if self.fast_dev_run:
            return JaxMDSimulationConfig(**SIMULATION_CONFIG_FAST)
        else:
            return JaxMDSimulationConfig(**SIMULATION_CONFIG)

    @staticmethod
    def _calculate_score(
        drift_frame: int, explosion_frame: int, num_frames: int
    ) -> float:
        if drift_frame == -1 and explosion_frame == -1:
            score = 1.0
        elif explosion_frame == -1:
            score = 0.5 + 0.5 * (drift_frame / num_frames)
        else:
            score = 0.5 * (explosion_frame / num_frames)

        return score
