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

import numpy as np
from ase import Atoms, units
from mlip.inference import run_batched_inference
from pydantic import BaseModel, TypeAdapter
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.scoring import compute_benchmark_score

logger = logging.getLogger("mlipaudit")

WIGGLE_DATASET_FILENAME = "wiggle150_dataset.json"

CONFORMER_THRESHOLDS = {"avg_mae": 0.5, "avg_rmse": 1.5}


class ConformerSelectionMoleculeResult(BaseModel):
    """Results object for small molecule conformer selection benchmark for a single
    molecule.

    Attributes:
        molecule_name: The molecule's name.
        mae: The MAE between the predicted and reference
            energy profiles of the conformers.
        rmse: The RMSE between the predicted and reference
            energy profiles of the conformers.
        spearman_correlation: The spearman correlation coefficient
            between predicted and reference energy profiles.
        spearman_p_value: The spearman p value between predicted
            and reference energy profiles.
        predicted_energy_profile: The predicted energy profile for each conformer.
        reference_energy_profile: The reference energy profiles for each conformer.
    """

    molecule_name: str
    mae: float
    rmse: float
    spearman_correlation: float
    spearman_p_value: float
    predicted_energy_profile: list[float]
    reference_energy_profile: list[float]


class ConformerSelectionResult(BenchmarkResult):
    """Results object for small molecule conformer selection benchmark.

    Attributes:
        molecules: The individual results for each molecule in a list.
        avg_mae: The MAE values for all molecules averaged.
        avg_rmse: The RMSE values for all molecules averaged.
    """

    molecules: list[ConformerSelectionMoleculeResult]
    avg_mae: float
    avg_rmse: float


class ConformerSelectionMoleculeModelOutput(BaseModel):
    """Stores model outputs for the conformer selection benchmark for a given molecule.

    Attributes:
        molecule_name: The molecule's name.
        predicted_energy_profile: The predicted energy profile for the conformers.
    """

    molecule_name: str
    predicted_energy_profile: list[float]


class ConformerSelectionModelOutput(ModelOutput):
    """Stores model outputs for the conformer selection benchmark.

    Attributes:
        molecules: Results for each molecule.
    """

    molecules: list[ConformerSelectionMoleculeModelOutput]


class Conformer(BaseModel):
    """Conformer dataclass.

    A class to store the data for a single molecular
    system, including its energy profile and coordinates of
    all its conformers.

    Attributes:
        molecule_name: The molecule's name.
        dft_energy_profile: The reference dft energies
            for each conformer.
        atom_symbols: The list of atom symbols for the molecule.
        conformer_coordinates: The coordinates for each conformer.
    """

    molecule_name: str
    dft_energy_profile: list[float]
    atom_symbols: list[str]
    conformer_coordinates: list[list[tuple[float, float, float]]]


Conformers = TypeAdapter(list[Conformer])


class ConformerSelectionBenchmark(Benchmark):
    """Benchmark for small organic molecule conformer selection.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is ``conformer_selection``.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of ``self.analyze()``. The result class type is
            ``ConformerSelectionResult``.
    """

    name = "conformer_selection"
    result_class = ConformerSelectionResult

    atomic_species = {"H", "C", "O", "S", "F", "Cl", "N"}

    def run_model(self) -> None:
        """Run a single point energy calculation for each structure.

        The calculation is performed as a batched inference using the MLIP force field
        directly. The energy profile is stored in the `model_output` attribute.
        """
        molecule_outputs = []
        for structure in self._wiggle150_data:
            logger.info("Running energy calculations for %s", structure.molecule_name)

            atoms_list = []
            for conformer_idx in range(len(structure.conformer_coordinates)):
                atoms = Atoms(
                    symbols=structure.atom_symbols,
                    positions=structure.conformer_coordinates[conformer_idx],
                )
                atoms_list.append(atoms)

            predictions = run_batched_inference(
                atoms_list,
                self.force_field,
                batch_size=16,
            )

            energy_profile_list: list[float] = [
                prediction.energy for prediction in predictions
            ]

            model_output = ConformerSelectionMoleculeModelOutput(
                molecule_name=structure.molecule_name,
                predicted_energy_profile=energy_profile_list,
            )
            molecule_outputs.append(model_output)

        self.model_output = ConformerSelectionModelOutput(molecules=molecule_outputs)

    def analyze(self) -> ConformerSelectionResult:
        """Calculates the MAE, RMSE and Spearman correlation.

        The results are returned. For a correct
        representation of the energy differences, the lowest energy conformer of the
        reference data is set to zero for the reference and inference energy profiles.

        Returns:
            A `ConformerSelectionResult` object with the benchmark results.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        reference_energy_profiles = {
            conformer.molecule_name: np.array(conformer.dft_energy_profile)
            for conformer in self._wiggle150_data
        }

        results = []
        for molecule in self.model_output.molecules:
            molecule_name = molecule.molecule_name
            energy_profile = molecule.predicted_energy_profile
            energy_profile = np.array(energy_profile)
            ref_energy_profile = np.array(reference_energy_profiles[molecule_name])

            min_ref_energy = np.min(ref_energy_profile)
            min_ref_idx = np.argmin(ref_energy_profile)

            # Lowest energy conformation of reference is set to zero
            ref_energy_profile_aligned = ref_energy_profile - min_ref_energy

            # Align predicted energy profile to the lowest reference conformer
            predicted_energy_profile_aligned = (
                energy_profile - energy_profile[min_ref_idx]
            ) / (units.kcal / units.mol)  # convert units to kcal/mol

            mae = mean_absolute_error(
                ref_energy_profile_aligned, predicted_energy_profile_aligned
            )
            rmse = root_mean_squared_error(
                ref_energy_profile_aligned, predicted_energy_profile_aligned
            )
            spearman_corr, spearman_p_value = spearmanr(
                ref_energy_profile_aligned, predicted_energy_profile_aligned
            )

            molecule_result = ConformerSelectionMoleculeResult(
                molecule_name=molecule_name,
                mae=mae,
                rmse=rmse,
                spearman_correlation=spearman_corr,
                spearman_p_value=spearman_p_value,
                predicted_energy_profile=predicted_energy_profile_aligned,
                reference_energy_profile=ref_energy_profile_aligned,
            )

            results.append(molecule_result)

        avg_mae = statistics.mean(r.mae for r in results)
        avg_rmse = statistics.mean(r.rmse for r in results)

        score = compute_benchmark_score(
            [avg_mae, avg_rmse],
            [
                CONFORMER_THRESHOLDS["avg_mae"],
                CONFORMER_THRESHOLDS["avg_rmse"],
            ],
        )

        return ConformerSelectionResult(
            molecules=results,
            avg_mae=statistics.mean(r.mae for r in results),
            avg_rmse=statistics.mean(r.rmse for r in results),
            score=score,
        )

    @functools.cached_property
    def _wiggle150_data(self) -> list[Conformer]:
        with open(
            self.data_input_dir / self.name / WIGGLE_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
        ) as f:
            wiggle150_data = Conformers.validate_json(f.read())

        if self.fast_dev_run:
            wiggle150_data = wiggle150_data[:1]

        return wiggle150_data
