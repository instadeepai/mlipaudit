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

from ase import Atoms
from mlip.simulation import SimulationState
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import BaseModel, ConfigDict, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput

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
        pattern_atoms: Two integers specifying the indices
            of the two atoms that are bonded together.
        charge: The charge of the molecule.
        smiles: The SMILES string of the molecule.
    """

    atom_symbols: list[str]
    coordinates: list[tuple[float, float, float]]
    pattern_atoms: tuple[int, int]
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


class BondLengthDistributionResult(BenchmarkResult):
    """Result object."""

    raise NotImplementedError


class BondLengthDistributionBenchmark(Benchmark):
    """Benchmark for small organic molecule bond length distribution."""

    name = "bond_length_distribution"
    result_class = BondLengthDistributionResult

    def run_model(self) -> None:
        """Run an MD simulation for each structure.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. The simulation state is stored in the
        ``model_output`` attribute.
        """
        molecule_outputs = []

        if self.fast_dev_run:
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

    def analyze(self) -> BenchmarkResult:
        """Analyze results."""
        raise NotImplementedError

    @functools.cached_property
    def _bond_length_distribution_data(self) -> dict[str, Molecule]:
        with open(
            self.data_input_dir / self.name / BOND_LENGTH_DISTRIBUTION_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
        ) as f:
            dataset = Molecules.validate_json(f.read())

        if self.fast_dev_run:
            dataset = dict(list(dataset.items())[:2])

        return dataset
