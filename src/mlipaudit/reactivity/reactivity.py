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
from ase import Atoms, units
from mlip.inference import run_batched_inference
from pydantic import BaseModel, NonNegativeFloat, TypeAdapter

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


class ReactionModelOutput(BaseModel):
    """Stores the predicted energies for the three states
    of a reaction.

    Attributes:
        reactants_energy: The reactants' energy.
        products_energy: The products' energy.
        transition_state_energy: The transition state energy.
    """

    reactants_energy: float
    products_energy: float
    transition_state_energy: float


class ReactivityModelOutput(ModelOutput):
    """Stores the model outputs for the reactivity benchmark,
    consisting of the energy predictions for each reaction.

    Attributes:
        reaction_ids: A list of reaction identifiers.
        energy_predictions: A corresponding list of energy predictions
            for each reaction.
    """

    reaction_ids: list[str]
    energy_predictions: list[ReactionModelOutput]


class ReactionResult(BaseModel):
    """Individual reaction result.

    Attributes:
        activation_energy_pred: The predicted activation energy.
        activation_energy_ref: The reference activation energy.
        activation_energy_abs_error: The absolute error between the
            predicted and reference activation energies.
        enthalpy_of_reaction_pred: The predicted enthalpy of reaction.
        enthalpy_of_reaction_ref: The reference enthalpy of reaction.
        enthalpy_of_reaction_abs_error: The absolute error between the
            predicted and reference enthalpies of reaction.
    """

    activation_energy_pred: float
    activation_energy_ref: float
    activation_energy_abs_error: NonNegativeFloat
    enthalpy_of_reaction_pred: float
    enthalpy_of_reaction_ref: float
    enthalpy_of_reaction_abs_error: NonNegativeFloat


class ReactivityResult(BenchmarkResult):
    """Result object for the reactivity benchmark.

    Attributes:
        reaction_results: A dictionary of reaction results where
            the keys are the reaction identifiers.
        mae_activation_energy: The MAE of the activation energies.
        rmse_activation_energy: The RMSE of the activation energies.
        mae_enthalpy_of_reaction: The MAE of the enthalpies of reactions.
        rmse_enthalpy_of_reaction: The RMSE of the enthalpies of reactions.
    """

    reaction_results: dict[str, ReactionResult]
    mae_activation_energy: NonNegativeFloat
    rmse_activation_energy: NonNegativeFloat
    mae_enthalpy_of_reaction: NonNegativeFloat
    rmse_enthalpy_of_reaction: NonNegativeFloat


class ReactivityBenchmark(Benchmark):
    """Benchmark for transition state energies.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `reactivity`.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class is
            `ReactivityResult`.
        model_output_class: A reference to the `ReactivityModelOutput` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
    """

    name = "reactivity"
    result_class = ReactivityResult
    model_output_class = ReactivityModelOutput

    required_elements = {"H", "C", "O", "N"}

    def run_model(self) -> None:
        """Run energy predictions."""
        atoms_list_all = []
        atoms_list_indices_reactions = {}
        i = 0

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
                reactants_energy=predictions[reaction_prediction_indices].energy
                * EV_TO_KCAL_MOL,
                products_energy=predictions[reaction_prediction_indices + 1].energy
                * EV_TO_KCAL_MOL,
                transition_state_energy=predictions[
                    reaction_prediction_indices + 2
                ].energy
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
            ref_reaction = self._grambow_data[reaction_id]
            ref_reactant = ref_reaction.reactants.energy
            ref_product = ref_reaction.products.energy
            ref_transition_state = ref_reaction.transition_state.energy

            # Activation energy
            ea = (
                energy_prediction.transition_state_energy
                - energy_prediction.reactants_energy
            )
            ea_ref = ref_transition_state - ref_reactant

            # Enthalpy of reaction
            dh = energy_prediction.products_energy - energy_prediction.reactants_energy
            dh_ref = ref_product - ref_reactant

            reaction_result = ReactionResult(
                activation_energy_pred=ea,
                activation_energy_ref=ea_ref,
                activation_energy_abs_error=abs(ea - ea_ref),
                enthalpy_of_reaction_pred=dh,
                enthalpy_of_reaction_ref=dh_ref,
                enthalpy_of_reaction_abs_error=abs(dh - dh_ref),
            )
            result[reaction_id] = reaction_result

        ea_abs_errors = np.array([
            reaction_result.activation_energy_abs_error
            for reaction_result in result.values()
        ])
        dh_abs_errors = np.array([
            reaction_result.enthalpy_of_reaction_abs_error
            for reaction_result in result.values()
        ])

        return ReactivityResult(
            reaction_results=result,
            mae_activation_energy=float(np.mean(ea_abs_errors)),
            rmse_activation_energy=float(np.sqrt(np.mean(ea_abs_errors**2))),
            mae_enthalpy_of_reaction=float(np.mean(dh_abs_errors)),
            rmse_enthalpy_of_reaction=float(np.sqrt(np.mean(dh_abs_errors**2))),
        )

    @functools.cached_property
    def _grambow_data(self) -> dict[str, Reaction]:
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
        return list(self._grambow_data.keys())
