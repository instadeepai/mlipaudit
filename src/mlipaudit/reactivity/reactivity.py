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

from ase import Atoms, units
from mlip.inference import run_batched_inference
from pydantic import BaseModel, TypeAdapter

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput

logger = logging.getLogger("mlipaudit")

SIMULATION_CONFIG = {
    "num_steps": 500_000,
    "snapshot_interval": 500,
    "num_episodes": 1000,
    "temperature_kelvin": 295.15,
    "box": 24.772,
}

SIMULATION_CONFIG_FAST = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 295.15,
    "box": 24.772,
}
EV_TO_KCAL_MOL = units.mol / units.kcal

GRAMBOW_DATASET_FILENAME = "grambow_dataset.json"


class Molecule(BaseModel):
    """Input molecule BaseModel."""

    energy: float
    atom_symbols: list[str]
    coordinates: list[tuple[float, float, float]]


class Reaction(BaseModel):
    """Reaction."""

    reactants: Molecule
    products: Molecule
    transition_state: Molecule


Reactions = TypeAdapter(dict[str, Reaction])


class ReactionModelOutput(BaseModel):
    """Individual model output."""

    reactants: float
    products: float
    transition_state: float


class ReactivityModelOutput(ModelOutput):
    """Model output."""

    reaction_ids: list[str]
    energy_predictions: list[ReactionModelOutput]


class ReactionResult(BaseModel):
    """Individual reaction result."""

    ea: float
    ea_ref: float
    dh: float
    dh_ref: float


class ReactivityResult(BenchmarkResult):
    """Result."""

    reaction_results: dict[str, ReactionResult]


class ReactivityBenchmark(Benchmark):
    """Benchmark for transition state energies.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `reactivity`.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class is
            `ReactivityResult`.
    """

    name = "reactivity"
    result_class = ReactivityResult

    model_output_class = ReactivityModelOutput

    def run_model(self) -> None:
        """Run energy predictions."""
        atoms_list_all = []
        atoms_list_indices_reactions = {}
        i = 0

        for reaction_id in self._reaction_ids:
            reaction_data = self._granbow_data[reaction_id]
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
            atoms_list_all.append(reactant_atoms)
            atoms_list_all.append(product_atoms)
            atoms_list_all.append(transition_atoms)

            atoms_list_indices_reactions[reaction_id] = i

            i += 3

        predictions = run_batched_inference(
            atoms_list_all,
            self.force_field,
            batch_size=128,
        )
        energy_predictions = []
        for reaction_id in self._reaction_ids:
            reaction_prediction_indices = atoms_list_indices_reactions[reaction_id]
            reaction_model_output = ReactionModelOutput(
                reactants=predictions[reaction_prediction_indices].energy
                * EV_TO_KCAL_MOL,
                products=predictions[reaction_prediction_indices + 1].energy
                * EV_TO_KCAL_MOL,
                transition_state=predictions[reaction_prediction_indices + 2].energy
                * EV_TO_KCAL_MOL,
            )

            energy_predictions.append(reaction_model_output)

        # Save in kcal/mol
        self.model_output = ReactivityModelOutput(
            reaction_ids=list(self._reaction_ids), energy_predictions=energy_predictions
        )

    def analyze(self) -> ReactivityResult:
        """Analysis.

        Returns:
            A `ReactivityResult` object with the benchmark results.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        result = {}
        for reaction_id, energy_prediction in zip(
            self.model_output.reaction_ids, self.model_output.energy_predictions
        ):
            ref_reaction = self._granbow_data[reaction_id]
            ref_reactant = ref_reaction.reactants.energy
            ref_product = ref_reaction.products.energy
            ref_transition_state = ref_reaction.transition_state.energy

            reaction_result = ReactionResult(
                ea=energy_prediction.transition_state - energy_prediction.reactants,
                ea_ref=ref_transition_state - ref_reactant,
                dh=energy_prediction.products - energy_prediction.reactants,
                dh_ref=ref_product - energy_prediction.reactants,
            )
            result[reaction_id] = reaction_result

        return ReactivityResult(reaction_results=result)

    @functools.cached_property
    def _granbow_data(self) -> dict[str, Reaction]:
        with open(
            self.data_input_dir / self.name / GRAMBOW_DATASET_FILENAME,
            mode="r",
            encoding="utf-8",
        ) as f:
            dataset = Reactions.validate_json(f.read())

        if self.fast_dev_run:
            dataset = dict(list(dataset.items())[:2])

        return dataset

    @functools.cached_property
    def _reaction_ids(self) -> list[str]:
        return list(self._granbow_data.keys())
