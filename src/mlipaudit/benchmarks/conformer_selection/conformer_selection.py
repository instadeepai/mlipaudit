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
from pydantic import BaseModel, Field, NonNegativeFloat, TypeAdapter
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils import run_inference

logger = logging.getLogger("mlipaudit")

WIGGLE_DATASET_FILENAME = "wiggle150_dataset.json"
NUM_DEV_SYSTEMS = 1

MAE_SCORE_THRESHOLD = 0.5
RMSE_SCORE_THRESHOLD = 1.5


class ConformerSelectionMoleculeResult(BaseModel):
    """Results object for small molecule conformer selection benchmark for a single
    molecule. Will have attributes set to None if the inference failed.

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
        failed: Whether the inference failed on the molecule.
    """

    molecule_name: str
    mae: NonNegativeFloat | None = None
    rmse: NonNegativeFloat | None = None
    spearman_correlation: float | None = Field(ge=-1.0, le=1.0, default=None)
    spearman_p_value: float | None = Field(ge=0.0, le=1.0, default=None)
    predicted_energy_profile: list[float] | None = None
    reference_energy_profile: list[float] | None = None
    failed: bool = False


class ConformerSelectionResult(BenchmarkResult):
    """Results object for small molecule conformer selection benchmark.

    Attributes:
        molecules: The individual results for each molecule in a list.
        avg_mae: The MAE values for all molecules that didn't fail averaged.
            Is None in the case all the inferences failed.
        avg_rmse: The RMSE values for all molecules that didn't fail averaged.
            Is None in the case all the inferences failed.
        failed: Whether all the simulations or inferences failed
            and no analysis could be performed. Defaults to False.
       score: The final score for the benchmark between
            0 and 1.
    """

    molecules: list[ConformerSelectionMoleculeResult]
    avg_mae: NonNegativeFloat | None = None
    avg_rmse: NonNegativeFloat | None = None


class ConformerSelectionMoleculeModelOutput(BaseModel):
    """Stores model outputs for the conformer selection benchmark for a given molecule.

    Attributes:
        molecule_name: The molecule's name.
        predicted_energy_profile: The predicted energy profile for the conformers.
            Is None if the inference failed on the molecule.
        failed: Whether the inference failed on the molecule.
    """

    molecule_name: str
    predicted_energy_profile: list[float] | None = None
    failed: bool = False


class ConformerSelectionModelOutput(ModelOutput):
    """Stores model outputs for the conformer selection benchmark.

    Attributes:
        molecules: Results for each molecule.
        num_failed: The number of molecules on which inference failed.
    """

    molecules: list[ConformerSelectionMoleculeModelOutput]
    num_failed: int = 0


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
            file. The name is `conformer_selection`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Small Molecules".
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `ConformerSelectionResult`.
        model_output_class: A reference to
            the `ConformerSelectionModelOutput` class.
        required_elements: The set of element types that are present in the benchmark's
            input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some element types that the model cannot handle. If False,
            the benchmark must have its own custom logic to handle missing element
            types. For this benchmark, the attribute is set to True.
    """

    name = "conformer_selection"
    category = "Small Molecules"
    result_class = ConformerSelectionResult
    model_output_class = ConformerSelectionModelOutput

    required_elements = {"H", "C", "O", "S", "F", "Cl", "N"}

    def run_model(self) -> None:
        """Run a single point energy calculation for each structure.

        The calculation is performed as a batched inference using the MLIP force field
        directly. The energy profile is stored in the `model_output` attribute.
        """
        molecule_outputs, num_failed = [], 0
        for structure in self._wiggle150_data:
            logger.info("Running energy calculations for %s", structure.molecule_name)

            atoms_list = []
            for conformer_idx in range(len(structure.conformer_coordinates)):
                atoms = Atoms(
                    symbols=structure.atom_symbols,
                    positions=structure.conformer_coordinates[conformer_idx],
                )
                atoms_list.append(atoms)

            predictions = run_inference(
                atoms_list,
                self.force_field,
                batch_size=16,
            )

            if None in predictions:
                model_output = ConformerSelectionMoleculeModelOutput(
                    molecule_name=structure.molecule_name, failed=True
                )
                num_failed += 1

            else:
                energy_profile_list = [prediction.energy for prediction in predictions]  # type: ignore
                model_output = ConformerSelectionMoleculeModelOutput(
                    molecule_name=structure.molecule_name,
                    predicted_energy_profile=energy_profile_list,
                )
            molecule_outputs.append(model_output)

        self.model_output = ConformerSelectionModelOutput(
            molecules=molecule_outputs, num_failed=num_failed
        )

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

            if molecule.failed:
                results.append(
                    ConformerSelectionMoleculeResult(
                        molecule_name=molecule_name, failed=True
                    )
                )
                continue

            energy_profile = np.array(molecule.predicted_energy_profile)

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

        if self.model_output.num_failed == len(self.model_output.molecules):
            return ConformerSelectionResult(molecules=results, failed=True, score=0.0)

        avg_mae = statistics.mean(r.mae for r in results if r.mae is not None)
        avg_rmse = statistics.mean(r.rmse for r in results if r.rmse is not None)

        score = compute_benchmark_score(
            [[r.mae for r in results], [r.rmse for r in results]],
            [MAE_SCORE_THRESHOLD, RMSE_SCORE_THRESHOLD],
        )

        return ConformerSelectionResult(
            molecules=results,
            avg_mae=avg_mae,
            avg_rmse=avg_rmse,
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

        if self.run_mode == RunMode.DEV:
            wiggle150_data = wiggle150_data[:NUM_DEV_SYSTEMS]

        return wiggle150_data
