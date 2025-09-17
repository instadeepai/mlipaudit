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
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import BaseModel, ConfigDict, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode

logger = logging.getLogger("mlipaudit")

BOND_LENGTH_DISTRIBUTION_DATASET_FILENAME = "bond_length_distribution.json"

SIMULATION_CONFIG = {
    "num_steps": 1_000_000,
    "snapshot_interval": 1000,
    "num_episodes": 1000,
    "temperature_kelvin": 300.0,
}

SIMULATION_CONFIG_FAST = {
    "num_steps": 10,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 300.0,
}


class Molecule(BaseModel):
    """Molecule class.

    Attributes:
        atom_symbols: The list of chemical symbols for the molecule.
        coordinates: The positional coordinates of the molecule.
        pattern_atom_indices: Two integers specifying the indices
            of the two atoms that are bonded together.
        charge: The charge of the molecule.
        smiles: The SMILES string of the molecule.
    """

    atom_symbols: list[str]
    coordinates: list[tuple[float, float, float]]
    pattern_atom_indices: tuple[int, int]
    reference_bond_distance: float
    charge: float
    smiles: str


Molecules = TypeAdapter(dict[str, Molecule])


class MoleculeSimulationOutput(BaseModel):
    """Stores the simulation state for a molecule.

    Attributes:
        molecule_name: The name of the molecule.
        simulation_state: The simulation state.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule_name: str
    simulation_state: SimulationState


class BondLengthDistributionModelOutput(ModelOutput):
    """Stores model outputs for the bond length distribution benchmark,
    consisting of simulation states for every molecule.

    Attributes:
        molecules: A list of simulation states for every molecule.
    """

    molecules: list[MoleculeSimulationOutput]


class BondLengthDistributionMoleculeResult(BaseModel):
    """Results object for a single molecule.

    Attributes:
        molecule_name: The name of the molecule.
        deviation_trajectory: A list of floats with the entry at index
            i representing the deviation at frame i of the trajectory,
            with each frame corresponding to 1ps of simulation time.
        avg_deviation: The average deviation of the molecule over the
            whole trajectory.
    """

    molecule_name: str
    deviation_trajectory: list[float]
    avg_deviation: float


class BondLengthDistributionResult(BenchmarkResult):
    """Results object for the bond length distribution benchmark.

    Attributes:
        molecules: The individual results for each molecule in a list.
    """

    molecules: list[BondLengthDistributionMoleculeResult]


class BondLengthDistributionBenchmark(Benchmark):
    """Benchmark for small organic molecule bond length distribution.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is ``bond_length_distribution``.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of ``self.analyze()``. The result class type is
            ``BondLengthDistributionResult``.
        model_output_class: A reference to
            the `BondLengthDistributionModelOutput` class.
        required_elements: The set of element types that are present in the benchmark's
            input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some element types that the model cannot handle. If False,
            the benchmark must have its own custom logic to handle missing element
            types. For this benchmark, the attribute is set to True.
    """

    name = "bond_length_distribution"
    result_class = BondLengthDistributionResult
    model_output_class = BondLengthDistributionModelOutput

    required_elements = {"N", "H", "O", "F", "C"}

    def run_model(self) -> None:
        """Run an MD simulation for each structure.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. The simulation state is stored in the
        ``model_output`` attribute.
        """
        molecule_outputs = []

        if self.run_mode == RunMode.DEV:
            md_config = JaxMDSimulationEngine.Config(**SIMULATION_CONFIG_FAST)
        else:
            md_config = JaxMDSimulationEngine.Config(**SIMULATION_CONFIG)

        for pattern_name, molecule in self._bond_length_distribution_data.items():
            logger.info("Running MD for %s", pattern_name)

            atoms = Atoms(
                symbols=molecule.atom_symbols,
                positions=molecule.coordinates,
            )
            md_engine = JaxMDSimulationEngine(
                atoms=atoms, force_field=self.force_field, config=md_config
            )
            md_engine.run()

            molecule_output = MoleculeSimulationOutput(
                molecule_name=pattern_name, simulation_state=md_engine.state
            )
            molecule_outputs.append(molecule_output)

        self.model_output = BondLengthDistributionModelOutput(
            molecules=molecule_outputs
        )

    def analyze(self) -> BondLengthDistributionResult:
        """Calculate how much chemical bonds deviate from the equilibrium bond length.

        The deviation of the length of the bond specified by the SMARTS pattern is
        measured throughout the simulation. The equilibrium bond length is taken from
        the reference structure.

        Returns:
            A ``BondLengthDistributionResult`` object.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        results = []
        for molecule_output in self.model_output.molecules:
            trajectory = molecule_output.simulation_state.positions
            pattern_indices = self._bond_length_distribution_data[
                molecule_output.molecule_name
            ].pattern_atom_indices

            reference_bond_distance = self._bond_length_distribution_data[
                molecule_output.molecule_name
            ].reference_bond_distance

            bond_length_trajectory = np.linalg.norm(
                trajectory[:, pattern_indices[0]] - trajectory[:, pattern_indices[1]],
                axis=1,
            )
            deviation_trajectory = list(
                bond_length_trajectory - reference_bond_distance
            )

            molecule_result = BondLengthDistributionMoleculeResult(
                molecule_name=molecule_output.molecule_name,
                deviation_trajectory=deviation_trajectory,
                avg_deviation=statistics.mean(deviation_trajectory),
            )
            results.append(molecule_result)

        return BondLengthDistributionResult(molecules=results)

    @functools.cached_property
    def _bond_length_distribution_data(self) -> dict[str, Molecule]:
        with open(
            self.data_input_dir / self.name / BOND_LENGTH_DISTRIBUTION_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
        ) as f:
            dataset = Molecules.validate_json(f.read())

        if self.run_mode == RunMode.DEV:
            dataset = dict(list(dataset.items())[:2])

        return dataset
