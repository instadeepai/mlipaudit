"""Dihedral scan benchmark for small molecules."""

import functools
import logging
from collections import defaultdict

from ase import Atoms
from mlip.inference import run_batched_inference
from pydantic import BaseModel, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput

logger = logging.getLogger("mlipaudit")

TORSIONNET_DATASET_FILENAME = "TorsionNet500.json"


class Fragment(BaseModel):
    """Fragment dataclass."""

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


class DihedralScanResult(BenchmarkResult):
    """Results object for the dihedral scan benchmark."""

    pass


class DihedralScanBenchmark(Benchmark):
    """Benchmark for small organic molecule dihedral scan."""

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
            atoms_list_all_structures, self.force_field, batch_size=16
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
        """
        raise NotImplementedError

    @functools.cached_property
    def _torsion_net_500(self) -> dict[str, Fragment]:
        with open(
            self.data_input_dir / self.name / TORSIONNET_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
        ) as f:
            dataset = Fragments.validate_json(f.read())

        if self.fast_dev_run:
            dataset = {"fragment_001": dataset["fragment_001"]}

        return dataset
