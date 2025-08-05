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

import logging
import statistics

import numpy as np
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import BaseModel, ConfigDict

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.folding.helpers import (
    compute_radius_of_gyration_for_ase_atoms,
    compute_tm_scores_and_rmsd_values,
    get_match_secondary_structure,
    get_proportion_folded_amino_acid,
)
from mlipaudit.utils import (
    create_ase_trajectory_from_simulation_state,
    create_mdtraj_trajectory_from_simulation_state,
)

logger = logging.getLogger("mlipaudit")

STRUCTURE_NAMES = [
    "chignolin_1uao_xray",
    "trp_cage_2jof_xray",
    "amyloid_beta_1ba6_nmr",
    "orexin_beta_1cq0_nmr",
]

SIMULATION_CONFIG = {
    "num_steps": 100_000,
    "snapshot_interval": 100,
    "num_episodes": 100,
    "temperature_kelvin": 300.0,
}

SIMULATION_CONFIG_FAST = {
    "num_steps": 20,
    "snapshot_interval": 10,
    "num_episodes": 1,
    "temperature_kelvin": 300.0,
}


class FoldingMoleculeResult(BaseModel):
    """Stores the result for one molecule of the folding benchmark.

    Attributes:
        structure_name: The name of the structure.
        rmsd_trajectory: The RMSD values for each frame of the trajectory.
        tm_score_trajectory: The TM scores for each frame of the trajectory.
        radius_of_gyration: Radius of gyration for each frame of the trajectory.
        proportion_folded_amino_acid: Proportion of folded amino acids for each frame
                                      of the trajectory.
        match_secondary_structure: Percentage of matches for each frame. Match means
                                   for a residue that the reference structure's
                                   secondary structure assignment is the same.
        min_rmsd: Minimum RMSD value.
        best_frame_rmsd: Frame index for the minimum RMSD value.
        max_tm_score: Maximum TM score.
        best_frame_tm_score: Frame index for the maximum TM score.
        radius_of_gyration_fluctuation: Standard deviation of radius of gyration
                                        throughout trajectory.
    """

    structure_name: str
    rmsd_trajectory: list[float]
    tm_score_trajectory: list[float]
    radius_of_gyration: list[float]
    proportion_folded_amino_acid: list[float]
    match_secondary_structure: list[float]
    min_rmsd: float
    best_frame_rmsd: int
    max_tm_score: float
    best_frame_tm_score: int
    radius_of_gyration_fluctuation: float


class FoldingResult(BenchmarkResult):
    """Stores the result of the folding benchmark.

    Attributes:
        molecules: A list of `FoldingMoleculeResult` for each molecule processed
                   in the benchmark.
        avg_min_rmsd: Average minimum RMSD value (averaged across molecules).
        avg_max_tm_score: Average maximum TM score (averaged across molecules).
    """

    molecules: list[FoldingMoleculeResult]
    avg_min_rmsd: float
    avg_max_tm_score: float


class FoldingModelOutput(ModelOutput):
    """Stores model outputs for the folding benchmark.

    Attributes:
        structure_names: Names of structures.
        simulation_states: `SimulationState` object for each structure
                           in the same order.
    """

    structure_names: list[str]
    simulation_states: list[SimulationState]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FoldingBenchmark(Benchmark):
    """Benchmark for folding of biosystems."""

    name = "folding"
    result_class = FoldingResult

    def run_model(self) -> None:
        """Run an MD simulation for each biosystem.

        The simulation results are stored in the `model_output` attribute.
        """
        self.model_output = FoldingModelOutput(
            structure_names=[],
            simulation_states=[],
        )

        structure_names = STRUCTURE_NAMES[:1] if self.fast_dev_run else STRUCTURE_NAMES
        if self.fast_dev_run:
            md_config = JaxMDSimulationEngine.Config(**SIMULATION_CONFIG_FAST)
        else:
            md_config = JaxMDSimulationEngine.Config(**SIMULATION_CONFIG)

        for structure_name in structure_names:
            logger.info("Running MD for %s", structure_name)
            xyz_filename = structure_name + ".xyz"
            atoms = ase_read(self.data_input_dir / self.name / xyz_filename)

            md_engine = JaxMDSimulationEngine(atoms, self.force_field, md_config)
            md_engine.run()

            final_state = md_engine.state
            self.model_output.structure_names.append(structure_name)
            self.model_output.simulation_states.append(final_state)

    def analyze(self) -> FoldingResult:
        """Analyzes the folding trajectories.

        Loads the trajectory from the simulation state and computes the TM-score
        and RMSD between the trajectory and the reference structure.
        Note that the reference structure for the TM-score may be the same or
        a different structure than the one used for the simulation.

        Returns:
            A `FoldingResult` object with the benchmark results.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        molecule_results = []

        for idx in range(len(self.model_output.structure_names)):
            structure_name = self.model_output.structure_names[idx]
            simulation_state = self.model_output.simulation_states[idx]

            topology_filename = structure_name + ".pdb"
            ref_filename = structure_name + "_ref.pdb"

            mdtraj_traj = create_mdtraj_trajectory_from_simulation_state(
                simulation_state,
                topology_path=self.data_input_dir / self.name / topology_filename,
            )
            ase_traj = create_ase_trajectory_from_simulation_state(simulation_state)

            # 1. Radius of gyration
            rg_values = [
                compute_radius_of_gyration_for_ase_atoms(frame) for frame in ase_traj
            ]

            # 2. Percentage of folded amino acid (from DSSP) and match
            # in secondary structure
            proportion_folded_amino_acid = get_proportion_folded_amino_acid(mdtraj_traj)
            match_secondary_structure = get_match_secondary_structure(
                mdtraj_traj,
                ref_path=self.data_input_dir / self.name / ref_filename,
                simplified=False,
            )

            # 3. TM-score and RMSD
            tm_scores, rmsd_values = compute_tm_scores_and_rmsd_values(
                mdtraj_traj,
                self.data_input_dir / self.name / ref_filename,
            )

            molecule_result = FoldingMoleculeResult(
                structure_name=structure_name,
                rmsd_trajectory=rmsd_values,
                tm_score_trajectory=tm_scores,
                radius_of_gyration=rg_values,
                proportion_folded_amino_acid=proportion_folded_amino_acid.tolist(),
                match_secondary_structure=match_secondary_structure.tolist(),
                min_rmsd=min(rmsd_values),
                best_frame_rmsd=np.argmin(rmsd_values),
                max_tm_score=max(tm_scores),
                best_frame_tm_score=np.argmax(tm_scores),
                radius_of_gyration_fluctuation=np.std(rg_values),
            )
            molecule_results.append(molecule_result)

        return FoldingResult(
            molecules=molecule_results,
            avg_min_rmsd=statistics.mean(r.min_rmsd for r in molecule_results),
            avg_max_tm_score=statistics.mean(r.max_tm_score for r in molecule_results),
        )
