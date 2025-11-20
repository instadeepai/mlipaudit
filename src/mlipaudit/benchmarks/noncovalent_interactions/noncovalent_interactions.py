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
from collections import defaultdict

import numpy as np
from ase import Atoms, units
from pydantic import BaseModel, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils import run_inference, skip_unallowed_elements

logger = logging.getLogger("mlipaudit")

NCI_ATLAS_FILENAME = "NCI_Atlas.json"

REPULSIVE_DATASETS = ["NCIA_R739x5"]

DATASET_RAW_TO_DESCRIPTIVE = {
    "D442x10": "Dispersion",
    "HB375x10": "Hydrogen bonds",
    "HB300SPXx10": "Hydrogen bonds",
    "IHB100x10": "Ionic hydrogen bonds",
    "R739x5": "Repulsive contacts",
    "SH250x10": "Sigma hole",
}

GROUP_RAW_TO_DESCRIPTIVE = {
    "CH-Oa": "CH-O(-)",
    "CH-Na": "CH-N(-)",
    "CH-Ca": "CH-C(-)",
    "NH-Oa": "NH-O(-)",
    "NH-Na": "NH-N(-)",
    "NH-Ca": "NH-C(-)",
    "OH-Oa": "OH-O(-)",
    "OH-Na": "OH-N(-)",
    "OH-Ca": "OH-C(-)",
    "NHk-O": "NH(+)-O",
    "NHk-C": "NH(+)-C",
    "NHk-N": "NH(+)-N",
    "OHk-O": "OH(+)-O",
    "B": "Boron",
}

INTERACTION_ENERGY_SCORE_THRESHOLD = 1.0


class NoncovalentInteractionsSystemResult(BaseModel):
    """Results object for the noncovalent interactions benchmark for a single
    bi-molecular system.

    Attributes:
        system_id: The system id.
        structure_name: The structure name.
        dataset: The dataset name.
        group: The group name.
        reference_interaction_energy: The reference interaction energy.
        mlip_interaction_energy: The MLIP interaction energy.
        deviation: The deviation between the reference and MLIP interaction
            energies.
        reference_energy_profile: The reference energy profile.
        energy_profile: The MLIP energy profile.
        distance_profile: The distance profile.
        failed: Whether running the model failed on the system. Defaults to False.

    """

    system_id: str
    structure_name: str
    dataset: str
    group: str
    reference_interaction_energy: float | None = None
    mlip_interaction_energy: float | None = None
    deviation: float | None = None
    reference_energy_profile: list[float] | None = None
    energy_profile: list[float] | None = None
    distance_profile: list[float] | None = None
    failed: bool = False


class NoncovalentInteractionsResult(BenchmarkResult):
    """Results object for the noncovalent interactions benchmark.

    Attributes:
        systems: The systems results for those that were successfully run.
        n_skipped_unallowed_elements: The number of structures skipped due to unallowed
            elements.
        num_failed: The number of structures that failed running inference.
        mae_interaction_energy_all: The MAE of the interaction energy over all
            tested systems.
        rmse_interaction_energy_all: The RMSE of the interaction energy over all
            tested systems.
        rmse_interaction_energy_subsets: The RMSE of the interaction energy per subset.
        mae_interaction_energy_subsets: The MAE of the interaction energy per subset.
        rmse_interaction_energy_datasets: The RMSE of the interaction energy per
            dataset.
        mae_interaction_energy_datasets: The MAE of the interaction energy per
            dataset.
        failed: Whether all the simulations or inferences failed
            and no analysis could be performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    systems: list[NoncovalentInteractionsSystemResult]
    n_skipped_unallowed_elements: int = 0
    num_failed: int = 0
    mae_interaction_energy_all: float
    rmse_interaction_energy_all: float
    rmse_interaction_energy_subsets: dict[str, float]
    mae_interaction_energy_subsets: dict[str, float]
    rmse_interaction_energy_datasets: dict[str, float]
    mae_interaction_energy_datasets: dict[str, float]


class MolecularSystem(BaseModel):
    """Dataclass for a bi-molecular system.

    Attributes:
        system_id: The system id.
        system_name: The system name.
        dataset_name: The dataset name.
        group: The group name.
        atom_symbols: The list of atom symbols for the molecule.
        coords: The coordinates of the atoms in the system.
        distance_profile: The distance profile of the interaction.
        interaction_energy_profile: The interaction energy profile of the interaction.
    """

    system_id: str
    system_name: str
    dataset_name: str
    group: str
    atom_symbols: list[str]
    coords: list[list[list[float]]]
    distance_profile: list[float]
    interaction_energy_profile: list[float]


Systems = TypeAdapter(dict[str, MolecularSystem])


class NoncovalentInteractionsSystemModelOutput(ModelOutput):
    """Model output for a bi-molecular system.

    Attributes:
        system_id: The unique id of the system.
        energy_profile: The energy profile of the system.
            Is None if any of the inferences failed when
            calculating the energy profile.
    """

    system_id: str
    energy_profile: list[float] | None = None


class NoncovalentInteractionsModelOutput(ModelOutput):
    """Model output for the noncovalent interactions benchmark.

    Attributes:
        systems: List of model outputs for each successfully run system.
        n_skipped_unallowed_elements: The number of structures skipped due to
            unallowed elements.
        skipped_structures: The list of skipped structures due to
            unallowed elements.
        failed_structures: The list of structures for which inference failed.
    """

    systems: list[NoncovalentInteractionsSystemModelOutput]
    n_skipped_unallowed_elements: int
    skipped_structures: list[str] = []
    failed_structures: list[str] = []


def compute_total_interaction_energy(
    distance_profile: list[float],
    interaction_energy_profile: list[float],
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


def _descriptive_data_subset_name(
    dataset_name: str,
    group: str,
) -> tuple[str, str]:
    """Return a descriptive name for a dataset subset."""
    dataset_name = dataset_name.replace("NCIA_", "")

    if dataset_name in DATASET_RAW_TO_DESCRIPTIVE:
        dataset_name_descriptive = DATASET_RAW_TO_DESCRIPTIVE[dataset_name]
    else:
        dataset_name_descriptive = dataset_name

    if group in GROUP_RAW_TO_DESCRIPTIVE:
        group_descriptive = GROUP_RAW_TO_DESCRIPTIVE[group]
    else:
        group_descriptive = group

    return dataset_name_descriptive, group_descriptive


def _compute_metrics_from_system_results(
    results: list[NoncovalentInteractionsSystemResult],
    n_skipped_structures: int,
    n_failed_structures: int,
) -> NoncovalentInteractionsResult:
    """Compute deviation metrics from the system results.

    Args:
        results: The system results for which the inference was successful.
        n_skipped_structures: The number of structures skipped due to unallowed
            elements,
        n_failed_structures: The number of structures that failed running inference.

    Returns:
        A `NoncovalentInteractionsResult` object with the benchmark results.
    """
    if len(results) == 0:
        return NoncovalentInteractionsResult(
            n_skipped_unallowed_elements=n_skipped_structures,
            num_failed=n_failed_structures,
            failed=True,
            score=0.0,
        )

    deviation_per_subset = defaultdict(list)
    deviation_per_dataset = defaultdict(list)
    for system_results in results:
        if system_results.failed:
            continue

        dataset_name = system_results.dataset
        group = system_results.group
        data_subset_name = f"{dataset_name}: {group}"

        deviation_per_subset[data_subset_name].append(system_results.deviation)
        deviation_per_dataset[dataset_name].append(system_results.deviation)

    rmse_interaction_energy_subsets = {}
    mae_interaction_energy_subsets = {}
    rmse_interaction_energy_datasets = {}
    mae_interaction_energy_datasets = {}
    for data_subset_name, deviations in deviation_per_subset.items():
        rmse_interaction_energy_subsets[data_subset_name] = np.sqrt(
            np.mean(np.array(deviations) ** 2)
        )
        mae_interaction_energy_subsets[data_subset_name] = np.mean(
            np.abs(np.array(deviations))
        )
    for dataset_name_descriptive, deviations in deviation_per_dataset.items():
        rmse_interaction_energy_datasets[dataset_name_descriptive] = np.sqrt(
            np.mean(np.array(deviations) ** 2)
        )
        mae_interaction_energy_datasets[dataset_name_descriptive] = np.mean(
            np.abs(np.array(deviations))
        )

    all_deviations = [
        system_result.deviation for system_result in results if not system_result.failed
    ]
    abs_deviations = [np.abs(dev) for dev in all_deviations]

    score = compute_benchmark_score(
        [abs_deviations + [None] * (n_skipped_structures + n_failed_structures)],
        [INTERACTION_ENERGY_SCORE_THRESHOLD],
    )

    mae_interaction_energy_all = np.mean(abs_deviations)
    rmse_interaction_energy_all = np.sqrt(np.mean(np.array(all_deviations) ** 2))

    return NoncovalentInteractionsResult(
        systems=results,
        n_skipped_unallowed_elements=n_skipped_structures,
        num_failed=n_failed_structures,
        mae_interaction_energy_all=mae_interaction_energy_all,
        rmse_interaction_energy_all=rmse_interaction_energy_all,
        rmse_interaction_energy_subsets=rmse_interaction_energy_subsets,
        mae_interaction_energy_subsets=mae_interaction_energy_subsets,
        rmse_interaction_energy_datasets=rmse_interaction_energy_datasets,
        mae_interaction_energy_datasets=mae_interaction_energy_datasets,
        score=score,
    )


class NoncovalentInteractionsBenchmark(Benchmark):
    """Benchmark for noncovalent interactions.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `noncovalent_interactions`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Small Molecules".
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `NoncovalentInteractionsResult`.
        model_output_class: A reference to the `NoncovalentInteractionsResult` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to False.
    """

    name = "noncovalent_interactions"
    category = "Small Molecules"
    result_class = NoncovalentInteractionsResult
    model_output_class = NoncovalentInteractionsModelOutput

    required_elements = {
        "Xe",
        "N",
        "I",
        "Ar",
        "H",
        "Se",
        "O",
        "S",
        "As",
        "Ne",
        "Br",
        "He",
        "Kr",
        "P",
        "C",
        "Cl",
        "F",
        "B",
    }
    skip_if_elements_missing = False

    def run_model(self) -> None:
        """Run a single point energy calculation for each structure.

        The calculation is performed as a batched inference using the mlip force field
        directly. This benchmark will skip structures with unseen elements.
        """
        skipped_structures = skip_unallowed_elements(
            self.force_field,
            [
                (structure.system_id, structure.atom_symbols)
                for structure in self._nci_atlas_data.values()
            ],
        )

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
                        symbols=structure.atom_symbols,
                        positions=coord,
                    )
                    atoms_all.append(atoms)
                    atoms_all_idx_map[structure.system_id].append(i)
                    i += 1

        logger.info("Running energy calculations...")
        if skipped_structures:
            logger.info(
                "Skipping %s structures because of unallowed elements.",
                len(skipped_structures),
            )

        predictions = run_inference(
            atoms_all,
            self.force_field,
            batch_size=128,
        )

        model_output_systems = []
        failed_structures = []
        for system_id, indices in atoms_all_idx_map.items():
            predictions_structure = [predictions[i] for i in indices]

            if None in predictions_structure:
                failed_structures.append(system_id)

            else:
                energy_profile = [
                    prediction.energy  # type: ignore
                    for prediction in predictions_structure
                ]
                model_output_systems.append(
                    NoncovalentInteractionsSystemModelOutput(
                        system_id=system_id,
                        energy_profile=energy_profile,
                    )
                )

        self.model_output = NoncovalentInteractionsModelOutput(
            systems=model_output_systems,
            n_skipped_unallowed_elements=len(skipped_structures),
            skipped_structures=skipped_structures,
            failed_structures=failed_structures,
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
            system_id = system.system_id

            # Convert to kcal/mol
            mlip_energy_profile = [
                energy / (units.kcal / units.mol) for energy in system.energy_profile
            ]
            distance_profile = self._nci_atlas_data[system_id].distance_profile
            ref_energy_profile = self._nci_atlas_data[
                system_id
            ].interaction_energy_profile

            dataset_name = self._nci_atlas_data[system_id].dataset_name
            repulsive = dataset_name in REPULSIVE_DATASETS

            ref_interaction_energy = compute_total_interaction_energy(
                distance_profile, ref_energy_profile, repulsive=repulsive
            )
            mlip_interaction_energy = compute_total_interaction_energy(
                distance_profile, mlip_energy_profile, repulsive=repulsive
            )
            deviation = mlip_interaction_energy - ref_interaction_energy

            group = self._nci_atlas_data[system_id].group

            results.append(
                NoncovalentInteractionsSystemResult(
                    system_id=system_id,
                    structure_name=self._nci_atlas_data[system_id].system_name,
                    dataset=_descriptive_data_subset_name(
                        dataset_name,
                        group,
                    )[0],
                    group=_descriptive_data_subset_name(
                        dataset_name,
                        group,
                    )[1],
                    reference_interaction_energy=ref_interaction_energy,
                    mlip_interaction_energy=mlip_interaction_energy,
                    deviation=deviation,
                    reference_energy_profile=ref_energy_profile,
                    energy_profile=mlip_energy_profile,
                    distance_profile=distance_profile,
                )
            )

        for system_id in self.model_output.failed_structures:
            dataset_name = self._nci_atlas_data[system_id].dataset_name
            group = self._nci_atlas_data[system_id].group
            results.append(
                NoncovalentInteractionsSystemResult(
                    system_id=system_id,
                    structure_name=self._nci_atlas_data[system_id].system_name,
                    dataset=_descriptive_data_subset_name(
                        dataset_name,
                        group,
                    )[0],
                    group=_descriptive_data_subset_name(
                        dataset_name,
                        group,
                    )[1],
                    failed=True,
                )
            )

        return _compute_metrics_from_system_results(
            results,
            self.model_output.n_skipped_unallowed_elements,
            len(self.model_output.failed_structures),
        )

    @functools.cached_property
    def _nci_atlas_data(self) -> dict[str, MolecularSystem]:
        with open(
            self.data_input_dir / self.name / NCI_ATLAS_FILENAME,
            "r",
            encoding="utf-8",
        ) as f:
            nci_atlas_data = Systems.validate_json(f.read())

        if self.run_mode == RunMode.DEV:
            nci_atlas_data = {
                "1.03.03": nci_atlas_data["1.03.03"],
                "1.01.01": nci_atlas_data["1.01.01"],
            }

        return nci_atlas_data
