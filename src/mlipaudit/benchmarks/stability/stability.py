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
import statistics
from typing import Any, TypedDict

import mdtraj
import numpy as np
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode
from mlipaudit.utils import (
    create_mdtraj_trajectory_from_simulation_state,
    run_simulation,
)
from mlipaudit.utils.stability import (
    HYDROGEN_BOND_CUTOFF_ANGSTROM,
    find_explosion_frame,
)

logger = logging.getLogger("mlipaudit")

SIMULATION_CONFIG = {
    "num_steps": 100_000,
    "snapshot_interval": 100,
    "num_episodes": 100,
    "temperature_kelvin": 300.0,
}

SIMULATION_CONFIG_DEV = {
    "num_steps": 10,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 300.0,
}
NUM_DEV_SYSTEMS = 2
NUM_FAST_SYSTEMS = 5

TEMPERATURE_THRESHOLD = 10_000


class StructureMetadata(TypedDict):
    """Docstring."""

    xyz: str
    pdb: str
    description: str


STRUCTURES: dict[str, StructureMetadata] = {
    # Small molecules in vacuum
    "Small_molecule_HCNO": {
        "xyz": "small_molecule_HCNO.xyz",
        "pdb": "small_molecule_HCNO.pdb",
        "description": "SMall molecule (HCNO)",
    },
    "Small_molecule_Sulfur": {
        "xyz": "small_molecule_S.xyz",
        "pdb": "small_molecule_S.pdb",
        "description": "Small molecule (contains Sulfur)",
    },
    "Small_molecule_Halogen": {
        "xyz": "small_molecule_Hal.xyz",
        "pdb": "small_molecule_Hal.pdb",
        "description": "Small molecule (contains Halogens)",
    },
    # peptides in vacuum
    "Peptide_HCNO": {
        "xyz": "peptide_HCNO.xyz",
        "pdb": "peptide_HCNO.pdb",
        "description": "Neurotensin in vacuum (PDB: 2LNF)",
    },
    "Peptide_cys": {
        "xyz": "peptide_cys.xyz",
        "pdb": "peptide_cys.pdb",
        "description": "Cyclic peptide with cysteines in vacuum (Oxytocin; PDB: 7OFG)",
    },
    # Medium protein in vacuum
    "Protein": {
        "xyz": "protein_1a7m.xyz",
        "pdb": "protein_1a7m.pdb",
        "description": "Protein structure in vacuum (PDB: 1A7M)",
    },
    # solvated systems
    "Peptide_solvated": {
        "xyz": "peptide_solv.xyz",
        "pdb": "peptide_solv.pdb",
        "description": "Solvated Oxytocin (PDB: 7OFG)",
    },
    "Peptide_solvated_ions": {
        "xyz": "peptide_solv_ion.xyz",
        "pdb": "peptide_solv_ion.pdb",
        "description": "Solvated Neurotensin with counter-ions (PDB: 2LNF)",
    },
}

BOX_SIZES = {
    "Peptide_solvated": [23.43, 28.96, 20.90],
    "Peptide_solvated_ions": [25.62, 27.89, 37.36],
}

STRUCTURE_NAMES = list(STRUCTURES.keys())


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
    traj: mdtraj.Trajectory, cutoff: float = HYDROGEN_BOND_CUTOFF_ANGSTROM / 10
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
        cutoff: The cutoff in nanometers. Defaults to 0.25.

    Returns:
        A tuple of two arrays (frames, bonds). The first
        is of shape (nbrokenbonds,) where each element corresponds
        to the frame index at which the bond was broken. The second
        is of shape (nbrokenbonds, 2) where each row is of the
        form (heavy_atom_index, hydrogen_atom_index).
        Therefore, the triplet `(frames[i], bonds[i][0], bonds[i][1])`
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
    traj: mdtraj.Trajectory, cutoff: float = HYDROGEN_BOND_CUTOFF_ANGSTROM / 10
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
            Defaults to 0.25.

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
    """Result object for a single structure.

    Attributes:
        structure_name: The name of the structure.
        description: The description of the structure.
        num_frames: The number of frames in the trajectory.
        num_steps: The number of steps the simulation was run for.
        exploded_frame: The frame at which the simulation exploded.
            -1 if it did not explode.
        drift_frame: The first frame at which a hydrogen atom started
            to drift. -1 if there is no drift.
        failed: Whether the simulation failed with an error, without
            necessarily not being stable.
        score: The final score for the structure.
    """

    structure_name: str
    description: str
    num_frames: PositiveInt = 0
    num_steps: PositiveInt
    exploded_frame: int = 0
    drift_frame: int = 0
    failed: bool = False
    score: float = Field(ge=0, le=1)


class StabilityResult(BenchmarkResult):
    """Result object for the stability benchmark.

    Attributes:
        structure_results: A list of individual results
            per structure.
        failed: Whether all the simulations failed and no analysis could be
            performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    structure_results: list[StabilityStructureResult]


class StabilityModelOutput(ModelOutput):
    """Stores model outputs for the stability benchmark.

    Attributes:
        structure_names: The list of structure names.
        simulation_states: The list of final simulation states
            for the corresponding structures. None if the si
            simulation failed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure_names: list[str]
    simulation_states: list[SimulationState | None]


class StabilityBenchmark(Benchmark):
    """Benchmark for running stability tests.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `stability`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category matches the default ("General").
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `StabilityResult`.
        model_output_class: A reference to the `StabilityModelOutput` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
    """

    name = "stability"
    category = "General"
    result_class = StabilityResult
    model_output_class = StabilityModelOutput

    required_elements = {"N", "H", "O", "S", "C", "Cl", "F"}

    def run_model(self) -> None:
        """Run MD for each structure.

        The simulation results are stored in the `model_output` attribute.
        """
        self.model_output = StabilityModelOutput(
            structure_names=[],
            simulation_states=[],
        )

        structure_names = STRUCTURE_NAMES
        if self.run_mode == RunMode.DEV:
            structure_names = STRUCTURE_NAMES[:NUM_DEV_SYSTEMS]
        elif self.run_mode == RunMode.FAST:
            structure_names = STRUCTURE_NAMES[:NUM_FAST_SYSTEMS]

        for structure_name in structure_names:
            logger.info("Running MD for %s", structure_name)
            xyz_filename = STRUCTURES[structure_name]["xyz"]
            atoms = ase_read(self.data_input_dir / self.name / xyz_filename)

            if structure_name in BOX_SIZES:
                simulation_state = run_simulation(
                    atoms,
                    self.force_field,
                    box=BOX_SIZES[structure_name],
                    **self._md_kwargs,
                )
            else:
                simulation_state = run_simulation(
                    atoms, self.force_field, **self._md_kwargs
                )

            self.model_output.structure_names.append(structure_name)
            self.model_output.simulation_states.append(simulation_state)

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
            if simulation_state is None:
                structure_results.append(
                    StabilityStructureResult(
                        structure_name=structure_name,
                        description=STRUCTURES[structure_name]["description"],
                        num_steps=self._md_kwargs["num_steps"],
                        failed=True,
                        score=0.0,
                    )
                )
                continue

            explosion_frame = find_explosion_frame(
                simulation_state, self._md_kwargs["temperature_kelvin"]
            )
            first_drifting_frame = -1
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

            num_frames = simulation_state.positions.shape[0]
            structure_results.append(
                StabilityStructureResult(
                    structure_name=structure_name,
                    description=STRUCTURES[structure_name]["description"],
                    num_frames=num_frames,
                    num_steps=self._md_kwargs["num_steps"],
                    exploded_frame=explosion_frame,
                    drift_frame=first_drifting_frame,
                    score=self._calculate_score(
                        drift_frame=first_drifting_frame,
                        explosion_frame=explosion_frame,
                        num_frames=num_frames,
                    ),
                )
            )

        if all(structure.failed for structure in structure_results):
            return StabilityResult(structure_results=structure_results, failed=True)

        return StabilityResult(
            structure_results=structure_results,
            score=statistics.mean(
                structure_result.score if not structure_result.failed else 0.0
                for structure_result in structure_results
            ),
        )

    @functools.cached_property
    def _md_kwargs(self) -> dict[str, Any]:
        if self.run_mode == RunMode.DEV:
            return SIMULATION_CONFIG_DEV

        return SIMULATION_CONFIG

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
