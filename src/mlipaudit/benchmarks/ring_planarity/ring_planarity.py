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
from ase import Atoms
from mlip.simulation import SimulationState
from pydantic import BaseModel, ConfigDict, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils import run_simulation
from mlipaudit.utils.stability import is_simulation_stable

logger = logging.getLogger("mlipaudit")

RING_PLANARITY_DATASET = "ring_planarity_data.json"

SIMULATION_CONFIG = {
    "num_steps": 1_000_000,
    "snapshot_interval": 1000,
    "num_episodes": 1000,
    "temperature_kelvin": 300.0,
}

SIMULATION_CONFIG_DEV = {
    "num_steps": 10,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 300.0,
}

DEVIATION_SCORE_THRESHOLD = 0.05


def deviation_from_plane(coords: np.ndarray) -> float:
    """Calculate the deviation of a molecule from a plane.

    Args:
        coords: numpy array of shape (n, 3) containing the coordinates of the atoms.

    Returns:
        rmsd: Root mean square deviation of points from the mean plane.
    """
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # Use PCA to find principal components
    # The last component will be normal to the mean plane
    _, _, vh = np.linalg.svd(centered_coords)
    normal = vh[-1]
    distances = np.abs(np.dot(centered_coords, normal))

    rmsd = float(np.sqrt(np.mean(distances**2)))
    return rmsd


class Molecule(BaseModel):
    """Molecule class.

    Attributes:
        atom_symbols: The list of atom symbols for the molecule.
        coordinates: The coordinate positions.
        smiles: The molecule smiles pattern.
        pattern_atoms: The indices of the atoms belonging
            to the ring.
        charge: The charge of the molecule.
    """

    atom_symbols: list[str]
    coordinates: list[tuple[float, float, float]]
    smiles: str
    pattern_atoms: list[int]
    charge: float


Molecules = TypeAdapter(dict[str, Molecule])


class RingPlanarityMoleculeResult(BaseModel):
    """Results object for a single molecule.

    Attributes:
        molecule_name: The name of the molecule.
        deviation_trajectory: A list of floats with the entry at index
            i representing the deviation at frame i of the trajectory,
            with each frame corresponding to 1ps of simulation time.
        avg_deviation: The average deviation of the molecule over the
            whole trajectory.
        failed: Whether the simulation succeeded and was stable.
            If not, the other attributes will be not be set.
            Defaults to False.
    """

    molecule_name: str
    deviation_trajectory: list[float] | None = None
    avg_deviation: float | None = None

    failed: bool = False


class RingPlanarityResult(BenchmarkResult):
    """Results object for the ring planarity benchmark.

    Attributes:
        molecules: The individual results for each molecule in a list,
            including potentially those that failed.
        mae_deviation: The MAE of the avg deviations for each molecule.
        failed: Whether all the simulations or inferences failed
            and no analysis could be performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    molecules: list[RingPlanarityMoleculeResult]
    mae_deviation: float | None = None


class MoleculeModelOutput(BaseModel):
    """Stores the simulation state for a molecule.

    Attributes:
        molecule_name: The name of the molecule.
        simulation_state: The simulation state. Defaults to None
            if the simulation failed.
        failed: Whether the simulation failed on the molecule.
            Defaults to False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule_name: str
    simulation_state: SimulationState | None = None
    failed: bool = False


class RingPlanarityModelOutput(ModelOutput):
    """Stores model outputs for the ring planarity benchmark.

    Attributes:
        molecules: A list of simulation states for each molecule, including
            those that failed.
        num_failed: The number of molecules for which simulation failed.
    """

    molecules: list[MoleculeModelOutput]
    num_failed: int = 0


class RingPlanarityBenchmark(Benchmark):
    """Benchmark for small organic molecule ring planarity.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `ring_planarity`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Small Molecules".
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `RingPlanarityResult`.
        model_output_class: A reference to the `RingPlanarityModelOutput` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
    """

    name = "ring_planarity"
    category = "Small Molecules"
    result_class = RingPlanarityResult
    model_output_class = RingPlanarityModelOutput

    required_elements = {"H", "C", "O", "N", "F"}

    def run_model(self) -> None:
        """Run an MD simulation for each structure.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. The model output is saved in the `model_output`
        attribute.
        """
        molecule_outputs, num_failed = [], 0

        if self.run_mode == RunMode.DEV:
            md_kwargs = SIMULATION_CONFIG_DEV
        else:
            md_kwargs = SIMULATION_CONFIG

        for molecule_name, molecule in self._qm9_structures.items():
            logger.info("Running MD for %s", molecule_name)

            atoms = Atoms(
                symbols=molecule.atom_symbols,
                positions=molecule.coordinates,
            )
            simulation_state = run_simulation(atoms, self.force_field, **md_kwargs)

            if simulation_state is not None:
                molecule_output = MoleculeModelOutput(
                    molecule_name=molecule_name, simulation_state=simulation_state
                )
            else:
                molecule_output = MoleculeModelOutput(
                    molecule_name=molecule_name, failed=True
                )
                num_failed += 1

            molecule_outputs.append(molecule_output)

        self.model_output = RingPlanarityModelOutput(
            molecules=molecule_outputs, num_failed=num_failed
        )

    def analyze(self) -> RingPlanarityResult:
        """Calculate how much the ring atoms deviate from a perfect plane.

        The deviation of the ring atoms from a perfect plane is expressed as
        an RMSD (see utils).

        Returns:
            A `RingPlanarityResult` object.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        results = []
        num_succeeded = 0

        for molecule_output in self.model_output.molecules:
            trajectory = molecule_output.simulation_state.positions

            if molecule_output.failed or not is_simulation_stable(
                molecule_output.simulation_state
            ):
                molecule_result = RingPlanarityMoleculeResult(
                    molecule_name=molecule_output.molecule_name, failed=True
                )
                results.append(molecule_result)
                continue

            num_succeeded += 1

            ring_atom_trajectory = trajectory[
                :, self._qm9_structures[molecule_output.molecule_name].pattern_atoms
            ]
            deviation_trajectory = [
                deviation_from_plane(frame) for frame in ring_atom_trajectory
            ]

            molecule_result = RingPlanarityMoleculeResult(
                molecule_name=molecule_output.molecule_name,
                deviation_trajectory=deviation_trajectory,
                avg_deviation=statistics.mean(deviation_trajectory),
            )
            results.append(molecule_result)

        if num_succeeded == 0:
            return RingPlanarityResult(molecules=results, failed=True, score=0.0)

        mae_deviation = statistics.mean(
            r.avg_deviation for r in results if r.avg_deviation is not None
        )
        score = compute_benchmark_score(
            [[r.avg_deviation for r in results]], [DEVIATION_SCORE_THRESHOLD]
        )

        return RingPlanarityResult(
            molecules=results, mae_deviation=mae_deviation, score=score
        )

    @functools.cached_property
    def _qm9_structures(self) -> dict[str, Molecule]:
        with open(
            self.data_input_dir / self.name / RING_PLANARITY_DATASET,
            mode="r",
            encoding="utf-8",
        ) as f:
            dataset = Molecules.validate_json(f.read())

        if self.run_mode == RunMode.DEV:
            dataset = {"benzene": dataset["benzene"], "furan": dataset["furan"]}

        return dataset
