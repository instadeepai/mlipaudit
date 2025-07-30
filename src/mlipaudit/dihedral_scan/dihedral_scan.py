"""Dihedral scan benchmark for small molecules."""

import functools
import logging

from pydantic import BaseModel, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult

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


class DihedralScanResult(BenchmarkResult):
    """Results object for the dihedral scan benchmark."""

    pass


class DihedralScanBenchmark(Benchmark):
    """Benchmark for small organic molecule dihedral scan."""

    name = "dihedral_scan"
    result_class = DihedralScanResult

    def run_model(self) -> None:
        """Run a single point energy calculation for each structure.

        The calculation is performed as a batched inference using the MLIP force field
        directly. The energy profile is stored in the `model_output` attribute.
        """
        pass

    def analyze(self) -> DihedralScanResult:
        """Calculates the RMSD between the MLIP and reference structures.

        The MAE and RMSE are calculated for each structure in the `inference_results`
        attribute. The results are stored in the `analysis_results` attribute. The
        results contain the MAE, RMSE and inference energy profile along the dihedral.
        """
        raise NotImplementedError

    @functools.cached_property
    def _torsion_net_500(self) -> list[Fragment]:
        with open(
            self.data_input_dir / self.name / TORSIONNET_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
        ) as f:
            dataset = Fragments.validate_json(f.read())

        if self.fast_dev_run:
            dataset = {"fragment_001": dataset["fragment_001"]}

        return dataset
