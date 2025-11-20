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
import math
import statistics

from ase import Atoms, units
from pydantic import BaseModel, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils import run_inference

NUM_DEV_SYSTEMS = 2

TAUTOMERS_DATASET_FILENAME = "tautobase_2792.json"
KCAL_MOL_PER_EV = 1.0 / (units.kcal / units.mol)

MAE_SCORE_THRESHOLD = 0.05


class TautomersMoleculeResult(BaseModel):
    """Results object for one molecule/tautomer pair in the tautomers benchmark.

    All units in kcal/mol.

    Attributes:
        structure_id: ID of the structure pair.
        abs_deviation: Absolute deviation in tautomer energy.
        predicted_energy_diff: Predicted energy difference between the tautomers.
        ref_energy_diff: Reference energy difference between the tautomers.
        failed: Whether the inference failed for the tautomer. defaults
            to false.
    """

    structure_id: str
    abs_deviation: float | None = None
    predicted_energy_diff: float | None = None
    ref_energy_diff: float | None = None

    failed: bool = False


class TautomersResult(BenchmarkResult):
    """Results object for tautomers benchmark.

    Attributes:
        molecules: List of benchmark results for each molecule/tautomer pair,
            including those that failed inference.
        mae: Mean absolute error from the reference for tautomer energies.
        rmse: Root-mean-square error from the refrence for tautomer energies.
        failed: Whether all the inferences failed and no analysis could be
            performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    molecules: list[TautomersMoleculeResult]
    mae: float | None = None
    rmse: float | None = None


class TautomersModelOutput(ModelOutput):
    """Stores model outputs for the conformer selection benchmark.

    Attributes:
        structure_ids: IDs of the structure (i.e. tautomer) pairs.
        predictions: The energy predictions for the tautomer pairs.
            None if the inference failed.
    """

    structure_ids: list[str]
    predictions: list[tuple[float, float] | None]


class TautomerPair(BaseModel):
    """JSON schemas for a single tautomer pair.

    Attributes:
        energies: Energies of the tautomers in eV.
        coordinates: Coordinates of the tautomers in Angstrom.
        atom_symbols: List of atoms in the order they appear in the structure.
               This is duplicated in case the atoms would not be in the same order.
    """

    energies: list[float]
    coordinates: list[list[list[float]]]
    atom_symbols: list[list[str]]


TautomerPairs = TypeAdapter(dict[str, TautomerPair])


class TautomersBenchmark(Benchmark):
    """Benchmark for relative vacuum energy differences of tautomers.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `tautomers`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Small Molecules".
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class is
            `TautomersResult`.
        model_output_class: A reference to the `TautomersModelOutput` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
    """

    name = "tautomers"
    category = "Small Molecules"
    result_class = TautomersResult
    model_output_class = TautomersModelOutput

    required_elements = {"H", "C", "N", "I", "Br", "S", "O", "F", "Cl"}

    def run_model(self) -> None:
        """Run single point energy calculations on tautomer structures.

        Note: The benchmark only runs single point energy calculations
        on the input structures, assuming they are already minimized using xtb.
        """
        atoms_list_all_structures: list[Atoms] = []
        structure_name_indices: dict[str, list[int]] = {}
        i = 0

        for structure_id, tautomer_entry in self._tautomers_data.items():
            structure_name_indices[structure_id] = []

            for j in range(2):
                coords = tautomer_entry.coordinates[j]
                # in case atoms are not in the same order both are present in database:
                atom_symbols = tautomer_entry.atom_symbols[j]
                atoms = Atoms(symbols=atom_symbols, positions=coords)
                atoms_list_all_structures.append(atoms)
                structure_name_indices[structure_id].append(i)
                i += 1

        self.model_output = TautomersModelOutput(
            structure_ids=[],
            predictions=[],
        )
        predictions = run_inference(
            atoms_list_all_structures, self.force_field, batch_size=128
        )

        for structure_id, indices in structure_name_indices.items():
            self.model_output.structure_ids.append(structure_id)
            predictions_structure = [predictions[i] for i in indices]

            if None in predictions_structure:
                self.model_output.predictions.append(None)
            else:
                self.model_output.predictions.append(
                    tuple(predictions[i].energy for i in indices)  # type: ignore
                )

    def analyze(self) -> TautomersResult:
        """Checks the energy of tautomers is in check with the reference data.

        Returns:
            A `TautomersResult` object with the benchmark results.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        molecule_results, num_failed = [], 0
        num_structures = len(self.model_output.structure_ids)

        for i in range(num_structures):
            structure_id = self.model_output.structure_ids[i]
            inference_energies = self.model_output.predictions[i]

            if inference_energies is None:
                molecule_results.append(
                    TautomersMoleculeResult(structure_id=structure_id, failed=True)
                )
                num_failed += 1
                continue

            ref_energies = self._tautomers_data[structure_id].energies
            ref_energy_diff = ref_energies[1] - ref_energies[0]

            # Stay in eV internally (MLIP uses eV)
            predicted_energy_diff = inference_energies[1] - inference_energies[0]

            abs_deviation = abs(predicted_energy_diff - ref_energy_diff)

            molecule_result = TautomersMoleculeResult(
                structure_id=structure_id,
                abs_deviation=float(abs_deviation) * KCAL_MOL_PER_EV,
                predicted_energy_diff=float(predicted_energy_diff) * KCAL_MOL_PER_EV,
                ref_energy_diff=float(ref_energy_diff) * KCAL_MOL_PER_EV,
            )
            molecule_results.append(molecule_result)

        if num_failed == num_structures:
            return TautomersResult(molecules=molecule_results, failed=True, score=0.0)

        mae = statistics.mean(
            r.abs_deviation for r in molecule_results if r.abs_deviation is not None
        )
        mse = statistics.mean(
            r.abs_deviation**2 for r in molecule_results if r.abs_deviation is not None
        )

        score = compute_benchmark_score(
            [[r.abs_deviation for r in molecule_results]], [MAE_SCORE_THRESHOLD]
        )

        return TautomersResult(
            molecules=molecule_results, mae=mae, rmse=math.sqrt(mse), score=score
        )

    @functools.cached_property
    def _tautomers_data(self) -> dict[str, TautomerPair]:
        with open(
            self.data_input_dir / self.name / TAUTOMERS_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
        ) as f:
            tautomers_dataset = TautomerPairs.validate_json(f.read())

        if self.run_mode == RunMode.DEV:
            tautomers_dataset = dict(list(tautomers_dataset.items())[:NUM_DEV_SYSTEMS])

        return tautomers_dataset
