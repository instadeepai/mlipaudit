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

from ase import Atoms
from mlip.simulation import SimulationType
from mlip.simulation.jax_md import JaxMDSimulationEngine

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.small_molecule_geometrics.utils import (
    Molecules,
    MoleculeSimulationOutput,
)

logger = logging.getLogger("mlipaudit")

QM9_NEUTRAL_FILENAME = "qm9_n100_neutral.json"
QM9_CHARGED_FILENAME = "qm9_n10_charged.json"
OPENFF_NEUTRAL_FILENAME = "openff_n100_neutral.json"
OPENFF_CHARGED_FILENAME = "openff_n10_charged.json"


class SmallMoleculeMinimizationModelOutput(ModelOutput):
    """ModelOutput object for small molecule conformer minimization benchmark."""

    qm_neutral: list[MoleculeSimulationOutput]
    qm_charged: list[MoleculeSimulationOutput]
    openff_neutral: list[MoleculeSimulationOutput]
    openff_charged: list[MoleculeSimulationOutput]


class SmallMoleculeMinimizationResult(BenchmarkResult):
    """Results object for small molecule minimization benchmark."""

    raise NotImplementedError


class SmallMoleculeMinimizationBenchmark(Benchmark):
    """Benchmark for small molecule minimization."""

    name = "small_molecule_minimization"
    result_class = SmallMoleculeMinimizationResult

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
            qm_neutral=[],
            qm_charged=[],
            openff_neutral=[],
            openff_charged=[],
        )
        dataset_prefixes = [
            "qm9_neutral",
            "qm9_charged",
            "openff_neutral",
            "openff_charged",
        ]

        for dataset_prefix in dataset_prefixes:
            property_name = f"_{dataset_prefix}_dataset"
            dataset: Molecules = getattr(self, property_name)
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

    def analyze(self) -> BenchmarkResult:
        """Calculates the RMSD between the MLIP and reference structures.

        The RMSD is calculated for each structure in the `inference_results` attribute.
        The results are stored in the `analysis_results` attribute. For every structure,
        the results contain the heavy atom RMSD of the last simulation frame with
        respect to the reference structure.
        """
        raise NotImplementedError

    def _load_dataset_from_file(self, filename: str) -> Molecules:
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
    def _qm9_neutral_dataset(self) -> Molecules:
        return self._load_dataset_from_file(QM9_NEUTRAL_FILENAME)

    @functools.cached_property
    def _qm9_charged_dataset(self) -> Molecules:
        return self._load_dataset_from_file(QM9_CHARGED_FILENAME)

    @functools.cached_property
    def _openff_neutral_dataset(self) -> Molecules:
        return self._load_dataset_from_file(OPENFF_NEUTRAL_FILENAME)

    @functools.cached_property
    def _openff_charged_dataset(self) -> Molecules:
        return self._load_dataset_from_file(OPENFF_CHARGED_FILENAME)
