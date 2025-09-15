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
from mlipaudit.folding_stability.helpers import (
    compute_radius_of_gyration_for_ase_atoms,
    compute_tm_scores_and_rmsd_values,
    get_match_secondary_structure,
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


class FoldingStabilityMoleculeResult(BaseModel):
    """Stores the result for one molecule of the folding stability benchmark.

    Attributes:
        structure_name: The name of the structure.
        rmsd_trajectory: The RMSD values for each frame of the trajectory.
        tm_score_trajectory: The TM scores for each frame of the trajectory.
        radius_of_gyration: Radius of gyration for each frame of the trajectory.
        match_secondary_structure: Percentage of matches for each frame. Match means
            for a residue that the reference structure's
            secondary structure assignment is the same.
        avg_rmsd: Average RMSD value.
        avg_tm_score: Average TM score.
        avg_match: Average of `match_secondary_structure` metric across trajectory.
        radius_of_gyration_fluctuation: Standard deviation of radius of gyration
            throughout trajectory.
        max_abs_deviation_radius_of_gyration: Maximum absolute deviation of
            radius of gyration from `t = 0`` in state in trajectory.
    """

    structure_name: str
    rmsd_trajectory: list[float]
    tm_score_trajectory: list[float]
    radius_of_gyration_deviation: list[float]
    match_secondary_structure: list[float]
    avg_rmsd: float
    avg_tm_score: float
    avg_match: float
    radius_of_gyration_fluctuation: float
    max_abs_deviation_radius_of_gyration: float


class FoldingStabilityResult(BenchmarkResult):
    """Stores the result of the folding stability benchmark.

    Attributes:
        molecules: A list of `FoldingStabilityMoleculeResult` for each molecule
            processed in the benchmark.
        avg_rmsd: Average RMSD value (averaged across molecules).
        avg_tm_score: Average TM score (averaged across molecules).
        avg_match: Average of averaged `match_secondary_structure` metric
            across molecules.
        max_abs_deviation_radius_of_gyration: Maximum absolute deviation of
            radius of gyration from `t = 0` in state in trajectory.
            Maximum absolute deviation across molecules.
    """

    molecules: list[FoldingStabilityMoleculeResult]
    avg_rmsd: float
    avg_tm_score: float
    avg_match: float
    max_abs_deviation_radius_of_gyration: float


class FoldingStabilityModelOutput(ModelOutput):
    """Stores model outputs for the folding stability benchmark.

    Attributes:
        structure_names: Names of structures.
        simulation_states: `SimulationState` object for each structure
            in the same order.
    """

    structure_names: list[str]
    simulation_states: list[SimulationState]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FoldingStabilityBenchmark(Benchmark):
    """Benchmark for folding stability of biosystems.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `folding_stability`.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class is
            `FoldingStabilityResult`.
        model_output_class: A reference to
                            the `FoldingStabilityModelOutput` class.
    """

    name = "folding_stability"
    result_class = FoldingStabilityResult
    model_output_class = FoldingStabilityModelOutput

    atomic_species = {"H", "N", "O", "S", "C"}

    def run_model(self) -> None:
        """Run an MD simulation for each biosystem.

        The simulation results are stored in the `model_output` attribute.
        """
        self.model_output = FoldingStabilityModelOutput(
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

    def analyze(self) -> FoldingStabilityResult:
        """Analyzes the folding stability trajectories.

        Loads the trajectory from the simulation state and computes the TM-score
        and RMSD between the trajectory and the reference structure.
        Note that the reference structure for the TM-score may be the same or
        a different structure than the one used for the simulation.

        Returns:
            A `FoldingStabilityResult` object with the benchmark results.

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

            # 2. Match in secondary structure (from DSSP)
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

            initial_rg = rg_values[0]
            rg_values_deviation = [(rg - initial_rg) for rg in rg_values]

            molecule_result = FoldingStabilityMoleculeResult(
                structure_name=structure_name,
                rmsd_trajectory=rmsd_values,
                tm_score_trajectory=tm_scores,
                radius_of_gyration_deviation=rg_values_deviation,
                match_secondary_structure=match_secondary_structure.tolist(),
                avg_rmsd=statistics.mean(rmsd_values),
                avg_tm_score=statistics.mean(tm_scores),
                avg_match=statistics.mean(match_secondary_structure),
                radius_of_gyration_fluctuation=np.std(rg_values),
                max_abs_deviation_radius_of_gyration=max(map(abs, rg_values_deviation)),
            )
            molecule_results.append(molecule_result)

        return FoldingStabilityResult(
            molecules=molecule_results,
            avg_rmsd=statistics.mean(r.avg_rmsd for r in molecule_results),
            avg_tm_score=statistics.mean(r.avg_tm_score for r in molecule_results),
            avg_match=statistics.mean(r.avg_match for r in molecule_results),
            max_abs_deviation_radius_of_gyration=max(
                r.max_abs_deviation_radius_of_gyration for r in molecule_results
            ),
        )
