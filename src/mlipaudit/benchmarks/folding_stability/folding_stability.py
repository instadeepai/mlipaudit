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
from pydantic import BaseModel, ConfigDict

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.benchmarks.folding_stability.helpers import (
    compute_radius_of_gyration_for_ase_atoms,
    compute_tm_scores_and_rmsd_values,
    get_match_secondary_structure,
)
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils import (
    create_ase_trajectory_from_simulation_state,
    create_mdtraj_trajectory_from_simulation_state,
    run_simulation,
)
from mlipaudit.utils.simulation import REUSABLE_BIOMOLECULES_OUTPUTS_ID
from mlipaudit.utils.stability import is_simulation_stable

logger = logging.getLogger("mlipaudit")

STRUCTURE_NAMES = [
    "chignolin_1uao_xray",
    "trp_cage_2jof_xray",
    "orexin_beta_1cq0_nmr",
]

BOX_SIZES = {
    "chignolin_1uao_xray": [23.98, 22.45, 20.68],
    "trp_cage_2jof_xray": [29.33, 29.74, 23.59],
    "orexin_beta_1cq0_nmr": [40.30, 29.56, 33.97],
}

SIMULATION_CONFIG = {
    "num_steps": 250_000,
    "snapshot_interval": 10_000,
    "num_episodes": 25,
    "temperature_kelvin": 300.0,
}

SIMULATION_CONFIG_DEV = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 300.0,
}
NUM_DEV_SYSTEMS = 1
NUM_FAST_SYSTEMS = 2

RMSD_SCORE_THRESHOLD = 2.0
TM_SCORE_THRESHOLD = 0.5


class FoldingStabilityMoleculeResult(BaseModel):
    """Stores the result for one molecule of the folding stability benchmark.

    Attributes:
        structure_name: The name of the structure.
        rmsd_trajectory: The RMSD values for each frame of the trajectory.
        tm_score_trajectory: The TM scores for each frame of the trajectory.
        radius_of_gyration_deviation: Radius of gyration for each frame
            of the trajectory.
        match_secondary_structure: Percentage of matches for each frame. Match means
            for a residue that the reference structure's
            secondary structure assignment is the same.
        avg_rmsd: Average RMSD value.
        avg_tm_score: Average TM score.
        avg_match: Average of `match_secondary_structure` metric across trajectory.
        radius_of_gyration_fluctuation: Standard deviation of radius of gyration
            throughout trajectory.
        max_abs_deviation_radius_of_gyration: Maximum absolute deviation of
            radius of gyration from `t = 0` in state in trajectory.
        failed: Whether the simulation was stable or failed. If not stable, the other
            attributes will default to None.
    """

    structure_name: str
    rmsd_trajectory: list[float] | None = None
    tm_score_trajectory: list[float] | None = None
    radius_of_gyration_deviation: list[float] | None = None
    match_secondary_structure: list[float] | None = None
    avg_rmsd: float | None = None
    avg_tm_score: float | None = None
    avg_match: float | None = None
    radius_of_gyration_fluctuation: float | None = None
    max_abs_deviation_radius_of_gyration: float | None = None

    failed: bool = False


class FoldingStabilityResult(BenchmarkResult):
    """Stores the result of the folding stability benchmark.

    Attributes:
        molecules: A list of `FoldingStabilityMoleculeResult` for each molecule
            processed in the benchmark.
        avg_rmsd: Average RMSD value (averaged across molecules).
        min_rmsd: Minimum RMSD value (minimum across molecules).
        avg_tm_score: Average TM score (averaged across molecules).
        max_tm_score: Maximum TM score (maximum across molecules).
        avg_match: Average of averaged `match_secondary_structure` metric
            across molecules.
        max_abs_deviation_radius_of_gyration: Maximum absolute deviation of
            radius of gyration from `t = 0` in state in trajectory.
            Maximum absolute deviation across molecules.
        failed: Whether all the simulations or inferences failed
            and no analysis could be performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    molecules: list[FoldingStabilityMoleculeResult]
    avg_rmsd: float | None = None
    min_rmsd: float | None = None
    avg_tm_score: float | None = None
    max_tm_score: float | None = None
    avg_match: float | None = None
    max_abs_deviation_radius_of_gyration: float | None = None


class FoldingStabilityModelOutput(ModelOutput):
    """Stores model outputs for the folding stability benchmark.

    Attributes:
        structure_names: Names of structures.
        simulation_states: `SimulationState` or `None` object for
            each structure in the same order as the structure
            names. `None` if the simulation failed.
    """

    structure_names: list[str]
    simulation_states: list[SimulationState | None]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FoldingStabilityBenchmark(Benchmark):
    """Benchmark for folding stability of biosystems.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `folding_stability`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Biomolecules".
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class is
            `FoldingStabilityResult`.
        model_output_class: A reference to
            the `FoldingStabilityModelOutput` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
        reusable_output_id: An optional ID that references other benchmarks with
            identical input systems and `ModelOutput` signatures (in form of a tuple).
            If present, a user or the CLI can make use of this information to reuse
            cached model outputs from another benchmark carrying the same ID instead of
            rerunning simulations or inference.
    """

    name = "folding_stability"
    category = "Biomolecules"
    result_class = FoldingStabilityResult
    model_output_class = FoldingStabilityModelOutput

    required_elements = {"H", "N", "O", "S", "C"}

    reusable_output_id = REUSABLE_BIOMOLECULES_OUTPUTS_ID

    def run_model(self) -> None:
        """Run an MD simulation for each biosystem.

        The simulation results are stored in the `model_output` attribute.
        """
        if self.run_mode == RunMode.DEV:
            structure_names = STRUCTURE_NAMES[:NUM_DEV_SYSTEMS]
        elif self.run_mode == RunMode.FAST:
            structure_names = STRUCTURE_NAMES[:NUM_FAST_SYSTEMS]
        else:
            structure_names = STRUCTURE_NAMES

        if self.run_mode == RunMode.DEV:
            md_kwargs = SIMULATION_CONFIG_DEV
        else:
            md_kwargs = SIMULATION_CONFIG

        self.model_output = FoldingStabilityModelOutput(
            structure_names=[],
            simulation_states=[],
        )

        for structure_name in structure_names:
            logger.info("Running MD for %s", structure_name)

            xyz_filename = structure_name + ".xyz"
            atoms = ase_read(
                self.data_input_dir / self.name / "starting_structures" / xyz_filename
            )

            simulation_state = run_simulation(
                atoms, self.force_field, box=BOX_SIZES[structure_name], **md_kwargs
            )

            self.model_output.structure_names.append(structure_name)
            self.model_output.simulation_states.append(simulation_state)

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

        self._assert_structure_names_in_model_output()

        molecule_results = []
        num_succeeded = 0

        for idx in range(len(self.model_output.structure_names)):
            structure_name = self.model_output.structure_names[idx]
            simulation_state = self.model_output.simulation_states[idx]

            if simulation_state is None or not is_simulation_stable(simulation_state):
                molecule_results.append(
                    FoldingStabilityMoleculeResult(
                        structure_name=structure_name, failed=True
                    )
                )
                continue

            num_succeeded += 1
            box_size = BOX_SIZES[structure_name]

            mdtraj_traj_solv = create_mdtraj_trajectory_from_simulation_state(
                simulation_state,
                topology_path=self.data_input_dir
                / self.name
                / "pdb_reference_structures"
                / f"{structure_name}.pdb",
                cell_lengths=box_size,  # type: ignore
            )
            ase_traj_solv = create_ase_trajectory_from_simulation_state(
                simulation_state
            )

            non_solvent_idx = mdtraj_traj_solv.top.select("not resname HOH")

            mdtraj_traj = mdtraj_traj_solv.atom_slice(non_solvent_idx)
            ase_traj = [atoms[non_solvent_idx] for atoms in ase_traj_solv]

            # 1. Radius of gyration
            rg_values = [
                compute_radius_of_gyration_for_ase_atoms(frame) for frame in ase_traj
            ]

            # 2. Match in secondary structure (from DSSP)
            match_secondary_structure = get_match_secondary_structure(
                mdtraj_traj,
                ref_path=self.data_input_dir
                / self.name
                / "pdb_reference_structures"
                / f"{structure_name}_ref.pdb",
                simplified=False,
            )

            # 3. TM-score and RMSD
            tm_scores, rmsd_values = compute_tm_scores_and_rmsd_values(
                mdtraj_traj,
                self.data_input_dir
                / self.name
                / "pdb_reference_structures"
                / f"{structure_name}_ref.pdb",
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

        if num_succeeded == 0:
            return FoldingStabilityResult(molecules=molecule_results, score=0.0)

        score = compute_benchmark_score(
            [
                [r.avg_rmsd for r in molecule_results],
                [r.avg_tm_score for r in molecule_results],
            ],
            [RMSD_SCORE_THRESHOLD, TM_SCORE_THRESHOLD],
        )

        return FoldingStabilityResult(
            molecules=molecule_results,
            avg_rmsd=statistics.mean(
                r.avg_rmsd for r in molecule_results if r.avg_rmsd is not None
            ),
            min_rmsd=min(
                r.avg_rmsd for r in molecule_results if r.avg_rmsd is not None
            ),
            avg_tm_score=statistics.mean(
                r.avg_tm_score for r in molecule_results if r.avg_tm_score is not None
            ),
            max_tm_score=max(
                r.avg_tm_score for r in molecule_results if r.avg_tm_score is not None
            ),
            avg_match=statistics.mean(
                r.avg_match for r in molecule_results if r.avg_match is not None
            ),
            max_abs_deviation_radius_of_gyration=max(
                r.max_abs_deviation_radius_of_gyration
                for r in molecule_results
                if r.max_abs_deviation_radius_of_gyration is not None
            ),
            score=score,
        )

    def _assert_structure_names_in_model_output(self) -> None:
        """Asserts whether model output structure names are correct as they may
        have been transferred from a different benchmark.
        """
        assert set(self.model_output.structure_names).issubset(STRUCTURE_NAMES)  # type: ignore
        assert len(self.model_output.structure_names) == (  # type: ignore
            NUM_DEV_SYSTEMS
            if self.run_mode == RunMode.DEV
            else (
                NUM_FAST_SYSTEMS
                if self.run_mode == RunMode.FAST
                else len(STRUCTURE_NAMES)
            )
        )
