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
from pathlib import Path

import numpy as np
from ase import Atoms, units
from mlip.inference import run_batched_inference
from numpy.typing import NDArray
from pydantic import BaseModel, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.tools import skip_unallowed_elements

logger = logging.getLogger("mlipaudit")

CURRENT_DIR = Path(__file__).resolve().parent

NCI_ATLAS_FILENAME = "NCI_Atlas.json"

EV_TO_KCAL_MOL = 23.0609

REPULSIVE_DATASETS = ["NCIA_R739x5"]


class NoncovalentInteractionsSystemResult(BenchmarkResult):
    """Results object for the noncovalent interactions benchmark for a single
    bi-molecular system.

    Attributes:
        dataset: The dataset name.
        group: The group name.
        reference_interaction_energy: The reference interaction energy.
        mlip_interaction_energy: The MLIP interaction energy.
        abs_deviation: The absolute deviation between the reference and MLIP interaction
            energies.
        reference_energy_profile: The reference energy profile.
        energy_profile: The MLIP energy profile.
        distance_profile: The distance profile.
    """

    structure_id: str
    structure_name: str
    dataset: str
    group: str
    reference_interaction_energy: float
    mlip_interaction_energy: float
    abs_deviation: float
    reference_energy_profile: list[float]
    energy_profile: list[float]
    distance_profile: list[float]


class NoncovalentInteractionsResult(BenchmarkResult):
    """Results object for the noncovalent interactions benchmark."""

    systems: list[NoncovalentInteractionsSystemResult]


class MolecularSystem(BaseModel):
    """Dataclass for a bi-molecular system."""

    system_id: str
    system_name: str
    dataset_name: str
    group: str
    atoms: list[str]
    coords: list[list[list[float]]]
    distance_profile: list[float]
    interaction_energy_profile: list[float]


Systems = TypeAdapter(dict[str, MolecularSystem])


class NoncovalentInteractionsSystemModelOutput(ModelOutput):
    """Model output for a bi-molecular system."""

    structure_id: str
    energy_profile: list[float]


class NoncovalentInteractionsModelOutput(ModelOutput):
    """Model output for the noncovalent interactions benchmark."""

    systems: list[NoncovalentInteractionsSystemModelOutput]


def compute_total_interaction_energy(
    distance_profile: NDArray[np.floating],
    interaction_energy_profile: NDArray[np.floating],
    repulsive: bool = False,
) -> float:
    """Compute the total interaction energy.

    This function will use the minimum energy value of the interaction energy profile
    as the bottom of the energy well and the energy value associated with the highest
    distance as the energy of the dissociated structure baseline.

    Args:
        distance_profile: The distance profile of the interaction, meaning a series of
            distances between the two interacting molecules.
        interaction_energy_profile: The interaction energy profile of the interaction,
            meaning a series of interaction energies between the two interacting
            molecules at the distances specified in the distance profile.
        repulsive: Whether to use the maximum energy value of the interaction energy
            profile as the bottom of the energy well. Defaults to False.

    Returns:
        The total interaction energy.
    """
    max_energy = np.max(interaction_energy_profile)
    min_energy = np.min(interaction_energy_profile)
    max_distance_idx = np.argmax(distance_profile)
    dissociated_energy = interaction_energy_profile[max_distance_idx]

    if repulsive:
        return max_energy - dissociated_energy
    else:
        return min_energy - dissociated_energy


class NoncovalentInteractionsBenchmark(Benchmark):
    """Benchmark for noncovalent interactions."""

    name = "noncovalent_interactions"
    result_class = NoncovalentInteractionsResult

    def run_model(self) -> None:
        """Run a single point energy calculation for each structure.

        The calculation is performed as a batched inference using the mlip force field
        directly. The energy profile is stored in the `inference_results` attribute.

        Args:
            force_field: The force field wrapping the model to benchmark.
        """
        skipped_structures = skip_unallowed_elements(
            self.force_field,
            [
                (structure.system_id, structure.atoms)
                for structure in self._nci_atlas_data.values()
            ],
        )

        if self.fast_dev_run:
            skipped_structures = []

        atoms_all: list[Atoms] = []
        atoms_all_idx_map: dict[str, list[int]] = {}
        i = 0

        for structure in self._nci_atlas_data.values():
            if structure.system_id in skipped_structures:
                continue
            else:
                atoms_all_idx_map[structure.system_id] = []
                for coord in structure.coords:
                    atoms = Atoms(
                        symbols=structure.atoms,
                        positions=coord,
                    )
                    atoms_all.append(atoms)
                    atoms_all_idx_map[structure.system_id].append(i)
                    i += 1

        predictions = run_batched_inference(
            atoms_all,
            self.force_field,
            batch_size=128,
        )

        logger.info("Running energy calculations...")
        if skipped_structures:
            logger.info(
                "Skipping %s structures because of unallowed elements.",
                len(skipped_structures),
            )

        model_output_systems = []
        for structure_id, indices in atoms_all_idx_map.items():
            predictions_structure = [predictions[i] for i in indices]
            energy_profile: list[float] = [
                prediction.energy for prediction in predictions_structure
            ]
            model_output_systems.append(
                NoncovalentInteractionsSystemModelOutput(
                    structure_id=structure_id,
                    energy_profile=energy_profile,
                )
            )

        self.model_output = NoncovalentInteractionsModelOutput(
            systems=model_output_systems,
        )

    def analyze(self) -> NoncovalentInteractionsResult:
        """Calculate the total interaction energies and their abs. deviations.

        This calculation will yield the MLIP total interaction energy and energy profile
        and the abs. deviation compared to the reference data.

        Returns:
            A `NoncovalentInteractionsResult` object with the benchmark results.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        results = []
        for system in self.model_output.systems:
            structure_id = system.structure_id
            mlip_energy_profile = [
                energy * units.kcal / units.mol for energy in system.energy_profile
            ]
            distance_profile = self._nci_atlas_data[structure_id].distance_profile
            ref_energy_profile = self._nci_atlas_data[
                structure_id
            ].interaction_energy_profile

            dataset_name = self._nci_atlas_data[structure_id].dataset_name
            repulsive = dataset_name in REPULSIVE_DATASETS

            ref_interaction_energy = compute_total_interaction_energy(
                distance_profile, ref_energy_profile, repulsive=repulsive
            )
            mlip_interaction_energy = compute_total_interaction_energy(
                distance_profile, mlip_energy_profile, repulsive=False
            )
            abs_deviation = np.abs(mlip_interaction_energy - ref_interaction_energy)

            results.append(
                NoncovalentInteractionsSystemResult(
                    structure_id=structure_id,
                    structure_name=self._nci_atlas_data[structure_id].system_name,
                    dataset=dataset_name,
                    group=self._nci_atlas_data[structure_id].group,
                    reference_interaction_energy=ref_interaction_energy,
                    mlip_interaction_energy=mlip_interaction_energy,
                    abs_deviation=abs_deviation,
                    reference_energy_profile=ref_energy_profile,
                    energy_profile=mlip_energy_profile,
                    distance_profile=distance_profile,
                )
            )

        return NoncovalentInteractionsResult(systems=results)

    @functools.cached_property
    def _nci_atlas_data(self) -> dict[str, MolecularSystem]:
        with open(
            self.data_input_dir / self.name / NCI_ATLAS_FILENAME,
            "r",
            encoding="utf-8",
        ) as f:
            nci_atlas_data = Systems.validate_json(f.read())

        if self.fast_dev_run:
            nci_atlas_data = {
                "1.01.01": nci_atlas_data["1.01.01"],
                "1.03.03": nci_atlas_data["1.03.03"],
            }

        return nci_atlas_data
