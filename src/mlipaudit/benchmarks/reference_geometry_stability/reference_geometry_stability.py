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

import mdtraj as md
import numpy as np
from ase import Atoms, units
from mlip.simulation import SimulationState
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    TypeAdapter,
)

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils import run_simulation
from mlipaudit.utils.stability import is_simulation_stable
from mlipaudit.utils.trajectory_helpers import create_mdtraj_trajectory_from_positions

logger = logging.getLogger("mlipaudit")

OPENFF_NEUTRAL_FILENAME = "openff_n200_neutral.json"
OPENFF_CHARGED_FILENAME = "openff_n20_charged.json"
DATASET_PREFIXES = [
    "openff_neutral",
    "openff_charged",
]

EXPLODED_RMSD_THRESHOLD = 100.0
BAD_RMSD_THRESHOLD = 0.3

SIMULATION_CONFIG = {
    "simulation_type": "minimization",
    "num_steps": 1000,
    "snapshot_interval": 10,
    "max_force_convergence_threshold": 0.01,
}

SIMULATION_CONFIG_DEV = {
    "simulation_type": "minimization",
    "num_steps": 10,
    "snapshot_interval": 1,
    "max_force_convergence_threshold": 0.01,
}
NUM_DEV_SYSTEMS = 2

RMSD_SCORE_THRESHOLD = 0.075


class Molecule(BaseModel):
    """Molecule class.

    Attributes:
        atom_symbols: The list of chemical symbols for the molecule.
        coordinates: The positional coordinates of the molecule.
        smiles: The SMILES string of the molecule.
        charge: The charge of the molecule.
    """

    atom_symbols: list[str]
    coordinates: list[tuple[float, float, float]]
    smiles: str
    charge: float


Molecules = TypeAdapter(dict[str, Molecule])


class MoleculeModelOutput(BaseModel):
    """Stores the simulation state for a molecule.

    Attributes:
        molecule_name: The name of the molecule.
        simulation_state: The simulation state. None if the
            simulation failed.
        failed: Whether the simulation failed on the molecule.
            Defaults to False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule_name: str
    simulation_state: SimulationState | None = None
    failed: bool = False


class ReferenceGeometryStabilityModelOutput(ModelOutput):
    """ModelOutput object for reference geometry stability benchmark.

    Attributes:
        openff_neutral: A list of simulation states for each molecule in the dataset,
            including those that failed.
        openff_charged: A list of simulation states for each molecule in the dataset,
            including those that failed.
    """

    openff_neutral: list[MoleculeModelOutput]
    openff_charged: list[MoleculeModelOutput]


class ReferenceGeometryStabilityDatasetResult(BaseModel):
    """Result for a single dataset.

    Attributes:
        rmsd_values: The list of rmsd values for each molecule.
        avg_rmsd: The average rmsd across all molecules in the dataset.
        num_exploded: The number of molecules that exploded or failed during
            minimization or that failed the simulation. Defaults to 0.
        num_bad_rmsds: The number of molecules that we consider to
            have a poor rmsd score. Defaults to 0.
        failed: Whether all the simulations or inferences failed
            and no analysis could be performed. Defaults to False.
    """

    rmsd_values: list[NonNegativeFloat | None]
    avg_rmsd: NonNegativeFloat | None = None
    num_exploded: NonNegativeInt = 0
    num_bad_rmsds: NonNegativeInt = 0
    failed: bool = False


class ReferenceGeometryStabilityResult(BenchmarkResult):
    """Results object for reference geometry stability benchmark.

    Attributes:
        openff_neutral: The results for the openff neutral dataset.
        openff_charged: The results for the openff charged dataset.
        avg_rmsd: The average rmsd across all datasets.
        failed: Whether all the simulations failed and no analysis could be
            performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    openff_neutral: ReferenceGeometryStabilityDatasetResult
    openff_charged: ReferenceGeometryStabilityDatasetResult
    avg_rmsd: NonNegativeFloat | None = None


class ReferenceGeometryStabilityBenchmark(Benchmark):
    """Benchmark for reference geometry stability.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `reference_geometry_stability`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Small Molecules".
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `ReferenceGeometryStabilityResult`.
        model_output_class: A reference to the `ReferenceGeometryStabilityModelOutput`
            class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
    """

    name = "reference_geometry_stability"
    category = "Small Molecules"
    result_class = ReferenceGeometryStabilityResult
    model_output_class = ReferenceGeometryStabilityModelOutput

    required_elements = {"N", "Cl", "H", "O", "S", "F", "P", "C", "Br"}

    def run_model(self) -> None:
        """Run an MD simulation for each structure.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. The model output is saved in the `model_output`
        attribute.
        """
        if self.run_mode == RunMode.DEV:
            md_kwargs = SIMULATION_CONFIG_DEV
        else:
            md_kwargs = SIMULATION_CONFIG

        self.model_output = ReferenceGeometryStabilityModelOutput(
            openff_neutral=[],
            openff_charged=[],
        )

        for dataset_prefix in DATASET_PREFIXES:
            property_name = f"_{dataset_prefix}_dataset"
            dataset: dict[str, Molecule] = getattr(self, property_name)

            for molecule_name, molecule in dataset.items():
                logger.info(
                    "Running energy minimization for %s in %s",
                    molecule_name,
                    dataset_prefix,
                )

                atoms = Atoms(
                    symbols=molecule.atom_symbols, positions=molecule.coordinates
                )
                simulation_state = run_simulation(atoms, self.force_field, **md_kwargs)

                if simulation_state is not None:
                    molecule_output = MoleculeModelOutput(
                        molecule_name=molecule_name, simulation_state=simulation_state
                    )
                else:
                    molecule_output = MoleculeModelOutput(
                        molecule_name=molecule_name, failed=True
                    )

                getattr(self.model_output, dataset_prefix).append(molecule_output)

    def analyze(self) -> ReferenceGeometryStabilityResult:
        """Calculates the RMSD between the MLIP and reference structures.

        The RMSD is calculated for each structure in the `inference_results` attribute.
        The results are stored in the `analysis_results` attribute. For every structure,
        the results contain the heavy atom RMSD of the last simulation frame with
        respect to the reference structure.

        Returns:
            A `ReferenceGeometryStabilityResult` object with the benchmark results.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        result = {}

        for dataset_prefix in DATASET_PREFIXES:
            rmsd_values: list[float | None] = []
            dataset_model_output: list[MoleculeModelOutput] = getattr(
                self.model_output, dataset_prefix
            )
            num_failed = 0

            property_name = f"_{dataset_prefix}_dataset"
            for molecule_output in dataset_model_output:
                if molecule_output.failed or not is_simulation_stable(
                    molecule_output.simulation_state
                ):
                    num_failed += 1
                    rmsd_values.append(None)
                    continue

                molecule_name = molecule_output.molecule_name
                simulation_state = molecule_output.simulation_state

                reference_molecule: Molecule = getattr(self, property_name)[
                    molecule_name
                ]
                atom_symbols = reference_molecule.atom_symbols
                reference_positions = np.array(reference_molecule.coordinates)
                t_ref = create_mdtraj_trajectory_from_positions(
                    positions=reference_positions, atom_symbols=atom_symbols
                )

                t_pred = create_mdtraj_trajectory_from_positions(
                    positions=simulation_state.positions,  # type: ignore
                    atom_symbols=atom_symbols,
                )

                # only include heavy atoms in RMSD calculation
                heavy_atom_indices = t_ref.top.select("not element H")

                # Get the rmsd of the final frame
                rmsd = float(
                    md.rmsd(t_pred, t_ref, atom_indices=heavy_atom_indices)[-1]
                )

                # convert back to Angstrom
                rmsd *= units.nm / units.Angstrom

                rmsd_values.append(rmsd)

            if all(rmsd is None for rmsd in rmsd_values):
                dataset_result = ReferenceGeometryStabilityDatasetResult(
                    num_exploded=num_failed, failed=True
                )
            else:
                num_bad_rmsds = sum(
                    1
                    for rmsd in rmsd_values
                    if rmsd is not None and rmsd > BAD_RMSD_THRESHOLD
                )
                avg_rmsd = statistics.mean(
                    rmsd for rmsd in rmsd_values if rmsd is not None
                )

            dataset_result = ReferenceGeometryStabilityDatasetResult(
                rmsd_values=rmsd_values,
                avg_rmsd=avg_rmsd,
                num_exploded=num_failed,
                num_bad_rmsds=num_bad_rmsds,
            )
            result[dataset_prefix] = dataset_result

        all_failed = all(dataset_result.failed for dataset_result in result.values())
        if all_failed:
            return ReferenceGeometryStabilityResult(**result, failed=True, score=0.0)

        # Weight average by structure
        score = compute_benchmark_score(
            [
                [
                    rmsd
                    for dataset_result in result.values()
                    for rmsd in dataset_result.rmsd_values
                ]
            ],
            [RMSD_SCORE_THRESHOLD],
        )

        avg_rmsd = statistics.mean(
            dataset_result.avg_rmsd
            for dataset_result in result.values()
            if dataset_result.avg_rmsd is not None
        )

        return ReferenceGeometryStabilityResult(
            **result, avg_rmsd=avg_rmsd, score=score
        )

    def _load_dataset_from_file(self, filename: str) -> dict[str, Molecule]:
        """Helper method to load, validate, and optionally truncate a dataset.

        Args:
            filename: The filename to load.

        Returns:
            A Molecules dataset.
        """
        filepath = self.data_input_dir / self.name / filename
        with open(filepath, "r", encoding="utf-8") as f:
            dataset = Molecules.validate_json(f.read())

        if self.run_mode == RunMode.DEV:
            dataset = dict(list(dataset.items())[:NUM_DEV_SYSTEMS])

        return dataset

    @functools.cached_property
    def _openff_neutral_dataset(self) -> dict[str, Molecule]:
        return self._load_dataset_from_file(OPENFF_NEUTRAL_FILENAME)

    @functools.cached_property
    def _openff_charged_dataset(self) -> dict[str, Molecule]:
        return self._load_dataset_from_file(OPENFF_CHARGED_FILENAME)
