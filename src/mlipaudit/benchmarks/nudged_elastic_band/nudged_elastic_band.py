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

import numpy as np
from ase import Atoms
from mlip.simulation import SimulationState
from mlip.simulation.ase import ASESimulationEngine
from mlip.simulation.configs import ASESimulationConfig
from pydantic import BaseModel, ConfigDict, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.benchmarks.nudged_elastic_band.engine import (
    NEBSimulationConfig,
    NEBSimulationEngine,
)
from mlipaudit.run_mode import RunMode

logger = logging.getLogger("mlipaudit")

NEB_DATASET_FILENAME = "grambow_dataset_neb.json"

FINAL_CONVERGENCE_THRESHOLD = 0.05

MINIMIZATION_CONFIG = {
    "simulation_type": "minimization",
    "num_steps": 50,
    "snapshot_interval": 1,
    "log_interval": 1,
    "timestep_fs": 5.0,
    "max_force_convergence_threshold": 0.01,
    "edge_capacity_multiplier": 1.25,
}

MINIMIZATION_CONFIG_DEV = {
    "simulation_type": "minimization",
    "num_steps": 1,
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

NEB_CONFIG_DEV = {
    "simulation_type": "neb",
    "num_images": 10,
    "num_steps": 1,
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
    "num_images": 10,
    "num_steps": 500,
    "snapshot_interval": 1,
    "log_interval": 1,
    "edge_capacity_multiplier": 1.25,
    "max_force_convergence_threshold": FINAL_CONVERGENCE_THRESHOLD,
    "neb_k": 0.1,
    "continue_from_previous_run": True,
    "climb": True,
}

NEB_CONFIG_CLIMB_DEV = {
    "simulation_type": "neb",
    "num_images": 10,
    "num_steps": 1,
    "snapshot_interval": 1,
    "log_interval": 1,
    "edge_capacity_multiplier": 1.25,
    "max_force_convergence_threshold": FINAL_CONVERGENCE_THRESHOLD,
    "neb_k": 0.1,
    "continue_from_previous_run": True,
    "climb": True,
}

NUM_DEV_SYSTEMS = 2


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
        reaction_id: The reaction identifier.
        converged: Whether the NEB calculation converged.
            None if the simulation failed.
        failed: Whether the simulation failed.
    """

    reaction_id: str
    converged: bool | None = None
    failed: bool = False


class NEBResult(BenchmarkResult):
    """Result for a NEB calculation.

    Attributes:
        reaction_results: A dictionary of reaction results where
            the keys are the reaction identifiers. Inlcudes the
            failed reactions.
        convergence_rate: The fraction of converged reactions.
        failed: Whether all the simulations or inferences failed
            and no analysis could be performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    reaction_results: list[NEBReactionResult]
    convergence_rate: float | None = None


class NEBModelOutput(ModelOutput):
    """Model output for a NEB calculation.

    Attributes:
        simulation_states: A list of simulation states for every NEB reaction.
            None if the simulation failed.
    """

    simulation_states: list[SimulationState | None]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NudgedElasticBandBenchmark(Benchmark):
    """Nudged Elastic Band benchmark.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `nudged_elastic_band`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Small Molecules".
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `NEBResult`.
        model_output_class: A reference to the `NEBModelOutput` class.
        required_elements: The set of element types that are present in the benchmark's
            input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some element types that the model cannot handle. If False,
            the benchmark must have its own custom logic to handle missing element
            types. For this benchmark, the attribute is set to True.
    """

    name = "nudged_elastic_band"
    category = "Small Molecules"
    result_class = NEBResult
    model_output_class = NEBModelOutput
    required_elements = {"H", "C", "N", "O"}
    skip_if_elements_missing = True

    def run_model(self) -> None:
        """Run the NEB calculation."""
        self.model_output = NEBModelOutput(
            simulation_states=[],
        )

        minim_config_kwargs = (
            MINIMIZATION_CONFIG_DEV
            if self.run_mode == RunMode.DEV
            else MINIMIZATION_CONFIG
        )
        minim_config = ASESimulationConfig(**minim_config_kwargs)

        neb_config_kwargs = (
            NEB_CONFIG_DEV if self.run_mode == RunMode.DEV else NEB_CONFIG
        )
        neb_config_climb_kwargs = (
            NEB_CONFIG_CLIMB_DEV if self.run_mode == RunMode.DEV else NEB_CONFIG_CLIMB
        )
        neb_config = NEBSimulationConfig(**neb_config_kwargs)
        neb_config_climb = NEBSimulationConfig(**neb_config_climb_kwargs)

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
            try:
                atoms_minimized_reactant, atoms_minimized_product = (
                    self._run_minimization(
                        reactant_atoms,
                        product_atoms,
                        self.force_field,
                        minim_config,
                    )
                )

                neb_simulation_state = self._run_neb(
                    atoms_minimized_reactant,
                    atoms_minimized_product,
                    transition_atoms,
                    self.force_field,
                    neb_config,
                    neb_config_climb,
                )
                self.model_output.simulation_states.append(neb_simulation_state)

            except Exception as e:
                logger.info(
                    "Error running simulation on atoms %s, %s, %s: %s",
                    str(reactant_atoms),
                    str(product_atoms),
                    str(transition_atoms),
                    str(e),
                )
                self.model_output.simulation_states.append(None)

    def analyze(self) -> NEBResult:
        """Analyze the NEB calculation.

        Returns:
            NEBResult: The result of the NEB calculation.

        Raises:
            RuntimeError: If run_model() has not been called.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        if all(
            simulation_state is None
            for simulation_state in self.model_output.simulation_states
        ):
            return NEBResult(failed=True, score=0.0)

        n_converged = 0
        n_total = len(self.model_output.simulation_states)
        reaction_results = []
        for i, simulation_state in enumerate(self.model_output.simulation_states):
            if simulation_state is None:
                reaction_results.append(
                    NEBReactionResult(reaction_id=self._reaction_ids[i], failed=True)
                )

            neb_final_force = np.sqrt((simulation_state.forces**2).sum(axis=1).max())
            if neb_final_force < FINAL_CONVERGENCE_THRESHOLD:
                n_converged += 1
                reaction_results.append(
                    NEBReactionResult(reaction_id=self._reaction_ids[i], converged=True)
                )
            else:
                reaction_results.append(
                    NEBReactionResult(
                        reaction_id=self._reaction_ids[i], converged=False
                    )
                )

        convergence_rate = n_converged / n_total
        score = convergence_rate
        return NEBResult(
            reaction_results=reaction_results,
            convergence_rate=convergence_rate,
            score=score,
        )

    def _run_minimization(
        self,
        initial_atoms: Atoms,
        final_atoms: Atoms,
        ff,
        em_config: ASESimulationConfig,
    ) -> tuple[Atoms, Atoms]:
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

    def _run_neb(
        self,
        initial_atoms: Atoms,
        final_atoms: Atoms,
        ts_atoms: Atoms,
        ff,
        neb_config: NEBSimulationConfig,
        neb_config_climb: NEBSimulationConfig,
    ) -> SimulationState:
        """Run a nudged elastic band calculation.

        Args:
            initial_atoms: The initial atoms.
            final_atoms: The final atoms.
            ts_atoms: The transition state atoms.
            ff: The force field.
            neb_config: The configuration for the nudged elastic band.
            neb_config_climb: The configuration for the nudged elastic band with
            climbing image method.

        Returns:
            neb_engine_climb: The nudged elastic band engine with climbing image method.
        """
        neb_engine = NEBSimulationEngine(
            initial_atoms, final_atoms, ff, neb_config, transition_state=ts_atoms
        )
        neb_engine.run()

        atomic_numbers = neb_engine.state.atomic_numbers
        atoms_list = []
        for coords in neb_engine.state.positions:
            atoms = Atoms(atomic_numbers, coords)
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
            dataset = dict(list(dataset.items())[:NUM_DEV_SYSTEMS])

        return dataset

    @functools.cached_property
    def _reaction_ids(self) -> list[str]:
        return list(self._grambow_data.keys())
