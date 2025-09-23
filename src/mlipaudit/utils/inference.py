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
import ase
from ase.calculators.calculator import Calculator as ASECalculator
from mlip.inference import run_batched_inference
from mlip.models import ForceField
from mlip.typing import Prediction


def run_inference(
    atoms_list: list[ase.Atoms],
    force_field: ForceField | ASECalculator,
    batch_size: int = 16,
) -> list[Prediction]:
    """Runs inference for a list of `ase.Atoms` objects.

    If `ForceField` object is passed, `run_batched_inference` from the mlip library
    is used.

    Args:
        atoms_list: The list of `ase.Atoms` objects.
        force_field: The force field.
        batch_size: Batch size, default 16. Will only be used if force field is passed
                    as a `ForceField` object.

    Returns:
        A list of `Prediction` objects.

    Raises:
        ValueError: If force field type is not compatible.
    """
    if isinstance(force_field, ForceField):
        return run_batched_inference(atoms_list, force_field, batch_size=batch_size)

    elif isinstance(force_field, ASECalculator):
        predictions = []
        for atoms in atoms_list:
            atoms.calc = force_field
            energy = atoms.get_potential_energy()
            predictions.append(Prediction(energy=energy))

    raise ValueError(
        "Provided force field must be either a mlip-compatible "
        "force field object or an ASE calculator."
    )
