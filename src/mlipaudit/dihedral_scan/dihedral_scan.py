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
from collections import defaultdict

import numpy as np
from ase import Atoms, units
from mlip.inference import run_batched_inference
from pydantic import BaseModel, TypeAdapter
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.scoring import compute_benchmark_score

logger = logging.getLogger("mlipaudit")

TORSIONNET_DATASET_FILENAME = "TorsionNet500.json"

DIHEDRAL_THRESHOLDS = {"avg_barrier_height_error": 1.0}


class Fragment(BaseModel):
    """Fragment dataclass.

    A class to store the data for a single fragment.

    Attributes:
        torsion_atom_indices: The atom indices of the torsion atoms.
        dft_energy_profile: A list of tuples of (torsion angle,
            DFT energy) where the DFT energy corresponds to the energy of
            the system at the given angle.
        atom_symbols: The list of atom symbols for the molecule.
        conformer_coordinates: The coordinates for each conformer.
        smiles: The SMILES string of the molecule.
    """

    torsion_atom_indices: list[int]
    dft_energy_profile: list[tuple[float, float]]
    atom_symbols: list[str]
    conformer_coordinates: list[list[tuple[float, float, float]]]
    smiles: str


Fragments = TypeAdapter(dict[str, Fragment])


class FragmentModelOutput(BaseModel):
    """Stores energy predictions per conformer.

    Attributes:
        fragment_name: The name of the fragment.
        energy_predictions: The list of energy predictions
            for each conformer of the fragment.
    """

    fragment_name: str
    energy_predictions: list[float]


class DihedralScanModelOutput(ModelOutput):
    """Stores energy predictions per fragment per conformer.

    Attributes:
        fragments: A list of predictions per fragment.
    """

    fragments: list[FragmentModelOutput]


class DihedralScanFragmentResult(BaseModel):
    """Stores individual fragment results.

    Attributes:
        fragment_name: The name of the fragment.
        mae: The mean absolute error between the predicted energy and the
            reference energies for all the conformers.
        rmse: The root mean square error between the predicted energy and
            the reference energies for all the conformers.
        pearson_r: The Pearson correlation coefficient between the predicted
            and reference energies for all the conformers.
        pearson_p: The p-value of the Pearson correlation coefficient between
            the predicted and reference energies for all the conformers.
        barrier_height_error: The absolute difference between the predicted
            and reference barrier height.
        predicted_energy_profile: The aligned predicted energies for each conformer
            in kcal/mol.
        reference_energy_profile: The reference energies for each conformer
            in kcal/mol.
        distance_profile: The torsion angle for each conformer.
    """

    fragment_name: str
    mae: float
    rmse: float
    pearson_r: float
    pearson_p: float
    barrier_height_error: float
    predicted_energy_profile: list[float]
    reference_energy_profile: list[float]
    distance_profile: list[float]


class DihedralScanResult(BenchmarkResult):
    """Results object for the dihedral scan benchmark.

    Attributes:
        avg_mae: The avg mae across all fragments.
        avg_rmse: The avg rmse across all fragments.
        avg_pearson_r: The avg Pearson correlation coefficient across all fragments.
        avg_pearson_p: The avg Pearson p-value across all fragments.
        avg_barrier_height_error: The avg barrier height error across all fragments.
        fragments: A list of results objects per fragment.
        score: The final score for the benchmark between
            0 and 1.
    """

    avg_mae: float
    avg_rmse: float
    avg_pearson_r: float
    avg_pearson_p: float
    avg_barrier_height_error: float

    fragments: list[DihedralScanFragmentResult]


class DihedralScanBenchmark(Benchmark):
    """Benchmark for small organic molecule dihedral scan.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is ``dihedral_scan``.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of ``self.analyze()``. The result class is
            ``DihedralScanResult``.
    """

    name = "dihedral_scan"
    result_class = DihedralScanResult

    def run_model(self) -> None:
        """Run a single point energy calculation for each conformer for each fragment.

        The calculation is performed as a batched inference using the MLIP force field
        directly. The predicted energies are then stored as a list corresponding to
        the conformers for each fragment.
        """
        atoms_list_all_structures = []
        structure_indices_map = defaultdict(list)

        index = 0

        for fragment_name, fragment in self._torsion_net_500.items():
            for conf_coord in fragment.conformer_coordinates:
                atoms = Atoms(symbols=fragment.atom_symbols, positions=conf_coord)
                atoms_list_all_structures.append(atoms)
                structure_indices_map[fragment_name].append(index)
                index += 1

        predictions = run_batched_inference(
            atoms_list_all_structures, self.force_field, batch_size=128
        )

        fragment_outputs = []

        for fragment_name, indices in structure_indices_map.items():
            fragment_output = FragmentModelOutput(
                fragment_name=fragment_name,
                energy_predictions=[predictions[i].energy for i in indices],
            )
            fragment_outputs.append(fragment_output)

        self.model_output = DihedralScanModelOutput(fragments=fragment_outputs)

    def analyze(self) -> DihedralScanResult:
        """Calculates the RMSD between the MLIP and reference structures.

        The MAE and RMSE are calculated for each structure in the `inference_results`
        attribute. The results are stored in the `analysis_results` attribute. The
        results contain the MAE, RMSE and inference energy profile along the dihedral.

        Returns:
            A `DihedralScanResult` object with the benchmark results.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        results = []
        for fragment_prediction in self.model_output.fragments:
            predicted_energy_profile = np.array(fragment_prediction.energy_predictions)

            ref_fragment = self._torsion_net_500[fragment_prediction.fragment_name]

            distance_profile = np.array([
                state[0] for state in ref_fragment.dft_energy_profile
            ])
            ref_energy_profile = np.array([
                state[1] for state in ref_fragment.dft_energy_profile
            ])

            # Align the profiles
            min_ref_idx = np.argmin(ref_energy_profile)

            predicted_energy_profile_aligned = (
                predicted_energy_profile - predicted_energy_profile[min_ref_idx]
            )

            predicted_energy_profile_aligned /= units.kcal / units.mol

            # Compute relevant metrics
            mae, rmse, r, p, barrier_height_error = self._compute_metrics(
                ref_energy_profile, predicted_energy_profile_aligned
            )

            fragment_result = DihedralScanFragmentResult(
                fragment_name=fragment_prediction.fragment_name,
                mae=mae,
                rmse=rmse,
                pearson_r=r,
                pearson_p=p,
                barrier_height_error=barrier_height_error,
                predicted_energy_profile=list(predicted_energy_profile_aligned),
                reference_energy_profile=list(ref_energy_profile),
                distance_profile=list(distance_profile),
            )

            results.append(fragment_result)

        avg_barrier_height_error = statistics.mean(
            r.barrier_height_error for r in results
        )
        score = compute_benchmark_score(
            [avg_barrier_height_error],
            [DIHEDRAL_THRESHOLDS["avg_barrier_height_error"]],
        )

        return DihedralScanResult(
            avg_mae=statistics.mean(r.mae for r in results),
            avg_rmse=statistics.mean(r.rmse for r in results),
            avg_pearson_r=statistics.mean(r.pearson_r for r in results),
            avg_pearson_p=statistics.mean(r.pearson_p for r in results),
            avg_barrier_height_error=statistics.mean(
                r.barrier_height_error for r in results
            ),
            fragments=results,
            score=score,
        )

    @staticmethod
    def _compute_metrics(
        ref_energy_profile: np.ndarray,
        predicted_energy_profile: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        mae = mean_absolute_error(ref_energy_profile, predicted_energy_profile)
        rmse = root_mean_squared_error(ref_energy_profile, predicted_energy_profile)

        r, p = pearsonr(ref_energy_profile, predicted_energy_profile)

        ref_barrier_height = np.max(ref_energy_profile) - np.min(ref_energy_profile)
        pred_barrier_height = np.max(predicted_energy_profile) - np.min(
            predicted_energy_profile
        )
        barrier_height_error = np.abs(pred_barrier_height - ref_barrier_height)
        return mae, rmse, r, p, barrier_height_error

    @functools.cached_property
    def _torsion_net_500(self) -> dict[str, Fragment]:
        with open(
            self.data_input_dir / self.name / TORSIONNET_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
        ) as f:
            dataset = Fragments.validate_json(f.read())

        if self.fast_dev_run:
            dataset = {
                "fragment_001": dataset["fragment_001"],
                "fragment_002": dataset["fragment_002"],
            }

        return dataset
