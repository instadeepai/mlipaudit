"""Benchmark for highly strained conformer selection."""

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from mlip.inference import run_batched_inference
from mlip.models import ForceField
from pydantic import BaseModel, TypeAdapter
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput

EV_TO_KCAL_MOL = 23.0609

logger = logging.getLogger(__name__)

WIGGLE_DATASET_FILENAME = "wiggle150_dataset.json"


class Conformer(BaseModel):
    """Conformer model.

    A model to store the data for a single molecular
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


class ConformerSelectionResult(BenchmarkResult):
    """Results object for small molecule conformer selection benchmar.

    Attributes:
        molecule_name: The molecule's name.
        mae: The MAE between the predicted and reference
            energy profiles of the conformers.
        rmse: The RMSE between the predicted and reference
            energy profiles of the conformers.
        spearmann_correlation: The spearman correlation coefficient
            between predicted and reference energy profiles.
        spearman_p_value: The spearman p value between predicted
            and reference energy profiles.
        predicted_energy_profile: The predicted energy profile
            for each conformer.
        reference_energies: The reference energy profiles
            for each conformer.
    """

    molecule_name: str
    mae: float
    rmse: float
    spearmann_correlation: float
    spearman_p_value: float
    predicted_energy_profile: list[float]
    reference_energies: list[float]


class ConformerSelectionModelOutput(ModelOutput):
    """Stores model outputs for the conformer selection benchmark.

    Attributes:
        molecule_name: The molecule's name.
        predicted_energy_profile: The predicted energy profile for the
            conformers.
    """

    molecule_name: str
    predicted_energy_profile: list[float]


class ConformerSelectionBenchmark(Benchmark):
    """Benchmark for small organic molecule conformer selection."""

    name = "small_molecule_conformer_selection"

    def __init__(
        self,
        force_field: ForceField,
        fast_dev_run: bool = False,
        data_input_dir: str | Path = "./data/sm_conformer_selection",
    ) -> None:
        """Constructor.

        Args:
            force_field: The force field model to be benchmarked.
            fast_dev_run: Whether to do a fast developer run. Defaults to False.
            data_input_dir: The data input directory.
                Defaults to "./data/sm_conformer_selection".
        """
        super().__init__(
            force_field=force_field,
            fast_dev_run=fast_dev_run,
            data_input_dir=data_input_dir,
        )

        self.model_output: list[ConformerSelectionModelOutput] | None
        self.results: list[ConformerSelectionResult] | None

        # Assume at this point that the data is under self.data_input_dir
        with open(
            self.data_input_dir / WIGGLE_DATASET_FILENAME, "r", encoding="utf-8"
        ) as f:
            wiggle150_data = json.load(f)
            conformer_dataset = TypeAdapter(list[Conformer])
            self.wiggle150_data = conformer_dataset.validate_json(wiggle150_data)

        if self.fast_dev_run:
            self.wiggle150_data = self.wiggle150_data[:1]

        self.structure_names = [
            conformer.molecule_name for conformer in self.wiggle150_data
        ]

        self.reference_energy_profiles = {
            conformer.molecule_name: np.array(conformer.dft_energy_profile)
            for conformer in self.wiggle150_data
        }

    def run_model(self) -> None:
        """Run a single point energy calculation for each structure.

        The calculation is performed as a batched inference using the mlip force field
        directly. The energy profile is stored in the `model_output` attribute.
        """
        model_outputs = []
        for structure in self.wiggle150_data:
            logger.info("Running energy calculations for %s", structure.molecule_name)

            atoms_list = []
            for conformer_idx in range(structure.coordinates.shape[0]):
                atoms = Atoms(
                    symbols=structure.atom_symbols,
                    positions=structure.coordinates[conformer_idx],
                )
                atoms_list.append(atoms)

            predictions = run_batched_inference(
                atoms_list,
                self.force_field,
                batch_size=len(atoms_list),  # Is a batch size of 50 ok?
            )

            energy_profile_list: list[float] = [
                prediction.energy for prediction in predictions
            ]
            energy_profile = np.array(energy_profile_list)

            model_output = ConformerSelectionModelOutput(
                molecule_name=structure.molecule_name,
                predicted_energy_profile=energy_profile,
            )
            model_outputs.append(model_output)

        self.model_output = model_outputs

    def analyze(self) -> list[ConformerSelectionResult]:
        """Calculates the MAE, RMSE and Spearman correlation.

        The results are stored in the `results` attribute. For a correct
        representation of the energy differences, the lowest energy conformer of the
        reference data is set to zero for the reference and inference energy profiles.

        Returns:
            A list of SmallMoleculeConformerSelectionResult

        Raises:
            RuntimeError: If called before `run_model`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")
        results = []
        for molecule in self.model_output:
            molecule_name, energy_profile = (
                molecule.molecule_name,
                molecule.energy_profile,
            )
            ref_energy_profile = self.reference_energy_profiles[molecule_name]

            min_ref_energy = np.min(ref_energy_profile)
            min_ref_idx = np.argmin(ref_energy_profile)

            # Lowest energy conformation of reference is set to zero
            ref_energy_profile_aligned = ref_energy_profile - min_ref_energy

            # Align predicted energy profile to the lowest reference conformer
            predicted_energy_profile_aligned = (
                energy_profile - energy_profile[min_ref_idx]
            ) * EV_TO_KCAL_MOL

            mae = mean_absolute_error(
                ref_energy_profile, predicted_energy_profile_aligned
            )
            rmse = root_mean_squared_error(
                ref_energy_profile, predicted_energy_profile_aligned
            )
            spearman_corr, spearman_p_value = spearmanr(
                ref_energy_profile, predicted_energy_profile_aligned
            )

            molecule_result = ConformerSelectionResult(
                molecule_name=molecule_name,
                mae=mae,
                rmse=rmse,
                spearmann_correlation=spearman_corr,
                spearman_p_value=spearman_p_value,
                predicted_energy_profile=predicted_energy_profile_aligned,
                reference_energies=ref_energy_profile_aligned,
            )

            results.append(molecule_result)

        return results
