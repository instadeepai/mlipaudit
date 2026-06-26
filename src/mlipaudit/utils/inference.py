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
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ase
from ase.calculators.calculator import Calculator as ASECalculator

if TYPE_CHECKING:
    # These pull in the heavy mlip/JAX stack; imported lazily inside `run_inference`
    # so that merely importing this module stays cheap.
    from mlip.models import ForceField
    from mlip.typing import Prediction

logger = logging.getLogger("mlipaudit")


def run_inference(
    atoms_list: list[ase.Atoms],
    force_field: ForceField | ASECalculator,
    batch_size: int = 16,
) -> list[Prediction | None]:
    """Runs inference for a list of `ase.Atoms` objects.

    If `ForceField` object is passed, `run_batched_inference` from the mlip library
    is used.

    Args:
        atoms_list: The list of `ase.Atoms` objects.
        force_field: The force field.
        batch_size: Batch size, default 16. Will only be used if force field is passed
                    as a `ForceField` object.

    Returns:
        A list of `Prediction` or None objects, None when the model failed
            to perform inference on a system for whatever reason.

    Raises:
        ValueError: If force field type is not compatible.
    """
    # Imported lazily to keep importing this module cheap (these pull in the heavy
    # mlip/JAX stack, only needed when inference is actually run).
    from mlip.inference import run_batched_inference  # noqa: PLC0415
    from mlip.models import ForceField  # noqa: PLC0415
    from mlip.typing import Prediction  # noqa: PLC0415

    if isinstance(force_field, ForceField):
        try:
            predictions = run_batched_inference(
                atoms_list, force_field, batch_size=batch_size
            )
            return predictions
        except ValueError as e:
            logger.info("Error running batched inference: %s", str(e))
            return [None] * len(atoms_list)

    elif isinstance(force_field, ASECalculator):
        predictions = []
        for atoms in atoms_list:
            try:
                atoms.calc = force_field
                energy = atoms.get_potential_energy()
                predictions.append(Prediction(energy=energy))
            except Exception as e:
                logger.info(
                    "Error running inference on system %s: %s", str(atoms), str(e)
                )
                predictions.append(None)
        return predictions

    raise ValueError(
        "Provided force field must be either a mlip-compatible "
        "force field object or an ASE calculator."
    )
