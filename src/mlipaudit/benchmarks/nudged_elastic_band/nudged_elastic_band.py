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
from mlip.simulation.ase import ASESimulationEngine
from pydantic import BaseModel, ConfigDict, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.benchmarks.nudged_elastic_band.helpers import NEBSimulationEngine, NEBSimulationConfig
from mlipaudit.run_mode import RunMode

logger = logging.getLogger("mlipaudit")

NEB_DATASET_FILENAME = "grambow_dataset_neb.json"

MINIMIZATION_CONFIG = {
    "simulation_type": "minimization",
    "num_steps": 50,
    "snapshot_interval": 1,
    "log_interval": 1,
    "timestep_fs": 5.0,
    "max_force_convergence_threshold": 0.01,
    "edge_capacity_multiplier": 1.25,
}

NEB_CONFIG = {
    "simulation_type": "neb",
    "num_images": 10,
    "num_steps": 500,
    "snapshot_interval": 1,
    "log_interval": 1,
    "edge_capacity_multiplier": 1.25,
    "max_force_convergence_threshold": 0.5,
    "neb_k": 0.1,
    "continue_from_previous_run": False,
    "climb": False,
}

NEB_CONFIG_CLIMB = {
    "simulation_type": "neb",
    "num_images": 8,
    "num_steps": 500,
    "snapshot_interval": 1,
    "log_interval": 1,
    "edge_capacity_multiplier": 1.25,
    "max_force_convergence_threshold": 0.05,
    "neb_k": 0.1,
    "continue_from_previous_run": True,
    "climb": True,
}


class Molecule(BaseModel):
    """Input molecule BaseModel class.

    Attributes:
        energy: The energy of the molecule.
        atom_symbols: The list of chemical symbols for the molecule.
        coordinates: The positional coordinates of the molecule.
    """

    energy: float
    atom_symbols: list[str]
    coordinates: list[tuple[float, float, float]]


class Reaction(BaseModel):
    """Reaction BaseModel class containing the information
    pertaining to the three states of a reaction, from
    reactants through the transition state to produce
    the products.

    Attributes:
        reactants: The reactants of the reaction.
        products: The products of the reaction.
        transition_state: The transition state of the reaction.
    """

    reactants: Molecule
    products: Molecule
    transition_state: Molecule


Reactions = TypeAdapter(dict[str, Reaction])


class NEBReactionResult(BaseModel):
    """Result for a NEB reaction.

    Attributes:
        converged: Whether the NEB calculation converged.
    """

    converged: bool


class NEBResult(BenchmarkResult):
    """Result for a NEB calculation.

    Attributes:
        reaction_results: A dictionary of reaction results where
            the keys are the reaction identifiers.
    """

    reaction_results: list[NEBReactionResult]
    convergence_rate: float


class NEBModelOutput(ModelOutput):
    """Model output for a NEB calculation.

    Attributes:
        simulation_states: A list of simulation states for every NEB reaction.
    """

    simulation_states: list[SimulationState]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NudgedElasticBandBenchmark(Benchmark):
    """Nudged Elastic Band benchmark.

    Attributes:
        name: The name of the benchmark.
        result_class: The class of the result.
        model_output_class: The class of the model output.
    """

    name = "nudged_elastic_band"
    result_class = NEBResult
    model_output_class = NEBModelOutput
    required_elements = {"H", "C", "N", "O"}
    skip_if_elements_missing = True

    def run_model(self) -> None:
        """Run the NEB calculation."""
        self.model_output = NEBModelOutput(
            simulation_states=[],
        )
        for reaction_id in self._reaction_ids:
            reaction_data = self._grambow_data[reaction_id]
            reactant_atoms = Atoms(
                symbols=reaction_data.reactants.atom_symbols,
                positions=reaction_data.reactants.coordinates,
            )
            product_atoms = Atoms(
                symbols=reaction_data.products.atom_symbols,
                positions=reaction_data.products.coordinates,
            )
            transition_atoms = Atoms(
                symbols=reaction_data.transition_state.atom_symbols,
                positions=reaction_data.transition_state.coordinates,
            )

    def analyze(self) -> NEBResult:
        """Analyze the NEB calculation."""
        return NEBResult(
            reaction_results=[],
            convergence_rate=0.0
        )

    def _run_minimization(self, initial_atoms, final_atoms, ff, em_config):
        """Run an energy minimization to obtain initial structures for NEB.

        Args:
            initial_atoms: The initial atoms.
            final_atoms: The final atoms.
            ff: The force field.
            em_config: The configuration for the energy minimization.

        Returns:
            atoms_initial_em: The initial atoms after energy minimization.
            atoms_final_em: The final atoms after energy minimization.
        """
        em_engine_initial = ASESimulationEngine(initial_atoms, ff, em_config)
        em_engine_initial.run()

        em_engine_final = ASESimulationEngine(final_atoms, ff, em_config)
        em_engine_final.run()

        atoms_initial_em = em_engine_initial.atoms
        atoms_final_em = em_engine_final.atoms

        return atoms_initial_em, atoms_final_em


    def _run_neb(self, initial_atoms, final_atoms, ts_atoms, ff, neb_config, neb_config_climb):
        """Run a nudged elastic band calculation.

        Args:
            initial_atoms: The initial atoms.
            final_atoms: The final atoms.
            ts_atoms: The transition state atoms.
            ff: The force field.
            neb_config: The configuration for the nudged elastic band.
            neb_config_climb: The configuration for the nudged elastic band with climbing.

        Returns:
            neb_engine_climb: The nudged elastic band engine with climbing.
        """
        neb_engine = NEBSimulationEngine(
            initial_atoms, final_atoms, ff, neb_config, transition_state=ts_atoms
        )
        neb_engine.run()

        atomic_numers = neb_engine.state.atomic_numbers
        atoms_list = []
        for coords in neb_engine.state.positions:
            atoms = Atoms(atomic_numers, coords)
            atoms_list.append(atoms)

        neb_engine_climb = NEBSimulationEngine(
            initial_atoms,
            final_atoms,
            ff,
            neb_config_climb,
            images=atoms_list,
        )
        neb_engine_climb.run()

        return neb_engine_climb.state


    @functools.cached_property
    def _grambow_data(self) -> dict[str, Reaction]:
        with open(
            self.data_input_dir / self.name / NEB_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
        ) as f:
            dataset = Reactions.validate_json(f.read())

        if self.run_mode == RunMode.DEV:
            dataset = dict(list(dataset.items())[:2])

        return dataset

    @functools.cached_property
    def _reaction_ids(self) -> list[str]:
        return list(self._grambow_data.keys())
