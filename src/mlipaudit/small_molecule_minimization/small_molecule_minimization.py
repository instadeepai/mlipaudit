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
from ase import Atoms
from mlip.simulation import SimulationState, SimulationType
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    TypeAdapter,
)

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.utils.trajectory_helpers import create_mdtraj_trajectory_from_positions

logger = logging.getLogger("mlipaudit")

QM9_NEUTRAL_FILENAME = "qm9_n100_neutral.json"
QM9_CHARGED_FILENAME = "qm9_n10_charged.json"
OPENFF_NEUTRAL_FILENAME = "openff_n100_neutral.json"
OPENFF_CHARGED_FILENAME = "openff_n10_charged.json"

EXPLODED_RMSD_THRESHOLD = 100.0
BAD_RMSD_THRESHOLD = 0.3


class Molecule(BaseModel):
    """Molecule class."""

    atom_symbols: list[str]
    coordinates: list[tuple[float, float, float]]
    smiles: str
    pattern_atoms: list[int] | None = None
    charge: float


Molecules = TypeAdapter(dict[str, Molecule])


class MoleculeSimulationOutput(BaseModel):
    """Stores the simulation state for a molecule.

    Attributes:
        molecule_name: The name of the molecule.
        simulation_state: The simulation state.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule_name: str
    simulation_state: SimulationState


class SmallMoleculeMinimizationModelOutput(ModelOutput):
    """ModelOutput object for small molecule conformer minimization benchmark."""

    qm9_neutral: list[MoleculeSimulationOutput]
    qm9_charged: list[MoleculeSimulationOutput]
    openff_neutral: list[MoleculeSimulationOutput]
    openff_charged: list[MoleculeSimulationOutput]


class SmallMoleculeMinimizationDatasetResult(BaseModel):
    """Result for a single dataset."""

    rmsds: list[NonNegativeFloat] = []
    avg_rmsd: NonNegativeFloat = 0.0
    num_exploded: NonNegativeInt = 0
    num_bad_rmsds: NonNegativeInt = 0


class SmallMoleculeMinimizationResult(BenchmarkResult):
    """Results object for small molecule minimization benchmark."""

    qm9_neutral: SmallMoleculeMinimizationDatasetResult = (
        SmallMoleculeMinimizationDatasetResult()
    )
    qm9_charged: SmallMoleculeMinimizationDatasetResult = (
        SmallMoleculeMinimizationDatasetResult()
    )
    openff_neutral: SmallMoleculeMinimizationDatasetResult = (
        SmallMoleculeMinimizationDatasetResult()
    )
    openff_charged: SmallMoleculeMinimizationDatasetResult = (
        SmallMoleculeMinimizationDatasetResult()
    )


class SmallMoleculeMinimizationBenchmark(Benchmark):
    """Benchmark for small molecule minimization."""

    name = "small_molecule_minimization"
    result_class = SmallMoleculeMinimizationResult

    model_output: SmallMoleculeMinimizationModelOutput

    dataset_prefixes = [
        "qm9_neutral",
        "qm9_charged",
        "openff_neutral",
        "openff_charged",
    ]

    def run_model(self) -> None:
        """Run an MD simulation for each structure.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. The model output is saved in the ``model_output``
        attribute.
        """
        md_config = JaxMDSimulationEngine.Config(
            simulation_type=SimulationType.MINIMIZATION,
            num_steps=100 if not self.fast_dev_run else 10,
            snapshot_interval=10 if not self.fast_dev_run else 1,
            num_episodes=10 if not self.fast_dev_run else 1,
            timestep_fs=0.1,
        )

        self.model_output = SmallMoleculeMinimizationModelOutput(
            qm9_neutral=[],
            qm9_charged=[],
            openff_neutral=[],
            openff_charged=[],
        )

        for dataset_prefix in self._dataset_prefixes:
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
                md_engine = JaxMDSimulationEngine(atoms, self.force_field, md_config)
                md_engine.run()

                getattr(self.model_output, dataset_prefix).append(
                    MoleculeSimulationOutput(
                        molecule_name=molecule_name, simulation_state=md_engine.state
                    )
                )

    def analyze(self) -> SmallMoleculeMinimizationResult:
        """Calculates the RMSD between the MLIP and reference structures.

        The RMSD is calculated for each structure in the `inference_results` attribute.
        The results are stored in the `analysis_results` attribute. For every structure,
        the results contain the heavy atom RMSD of the last simulation frame with
        respect to the reference structure.

        Returns:
            A `SmallMoleculeMinimizationResult` object with the benchmark results.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        result = SmallMoleculeMinimizationResult()

        for dataset_prefix in self._dataset_prefixes:
            rmsds = []
            dataset_model_output: list[MoleculeSimulationOutput] = getattr(
                self.model_output, dataset_prefix
            )

            property_name = f"_{dataset_prefix}_dataset"
            for molecule_output in dataset_model_output:
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
                    positions=simulation_state.positions,
                    atom_symbols=atom_symbols,
                )

                # only include heavy atoms in RMSD calculation
                heavy_atom_indices = t_ref.top.select("not element H")

                # Get the rmsd of the final frame
                rmsd = float(
                    md.rmsd(t_pred, t_ref, atom_indices=heavy_atom_indices)[-1]
                )

                # convert to angstrom
                rmsd *= 10

                rmsds.append(rmsd)

            getattr(result, dataset_prefix).rmsds = rmsds

            getattr(result, dataset_prefix).avg_rmsd = statistics.mean(
                getattr(result, dataset_prefix).rmsds
            )

            getattr(result, dataset_prefix).num_exploded = sum(
                1 for rmsd in rmsds if rmsd > EXPLODED_RMSD_THRESHOLD
            )

            getattr(result, dataset_prefix).num_bad_rmsds = sum(
                1 for rmsd in rmsds if rmsd > BAD_RMSD_THRESHOLD
            )

        return result

    @property
    def _dataset_prefixes(self) -> list[str]:
        return [
            "qm9_neutral",
            "qm9_charged",
            "openff_neutral",
            "openff_charged",
        ]

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

        if self.fast_dev_run:
            dataset = dict(list(dataset.items())[:2])

        return dataset

    @functools.cached_property
    def _qm9_neutral_dataset(self) -> dict[str, Molecule]:
        return self._load_dataset_from_file(QM9_NEUTRAL_FILENAME)

    @functools.cached_property
    def _qm9_charged_dataset(self) -> dict[str, Molecule]:
        return self._load_dataset_from_file(QM9_CHARGED_FILENAME)

    @functools.cached_property
    def _openff_neutral_dataset(self) -> dict[str, Molecule]:
        return self._load_dataset_from_file(OPENFF_NEUTRAL_FILENAME)

    @functools.cached_property
    def _openff_charged_dataset(self) -> dict[str, Molecule]:
        return self._load_dataset_from_file(OPENFF_CHARGED_FILENAME)
