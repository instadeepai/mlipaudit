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

import logging
from copy import deepcopy
from typing import Callable

import ase
from ase.calculators.calculator import Calculator as ASECalculator
from mlip.models import ForceField
from mlip.simulation import SimulationState
from mlip.simulation.ase import ASESimulationEngine
from mlip.simulation.configs import ASESimulationConfig
from mlip.simulation.jax_md import JaxMDSimulationEngine
from mlip.simulation.temperature_scheduling import get_temperature_schedule

logger = logging.getLogger("mlipaudit")


class ASESimulationEngineWithCalculator(ASESimulationEngine):
    """Class derived from mlip's ASE simulation engine but allowing for a passed
    ASE calculator object.
    """

    def __init__(
        self,
        atoms: ase.Atoms,
        ase_calculator: ASECalculator,
        config: ASESimulationConfig,
    ) -> None:
        """Overridden constructor that takes in an ASE calculator instead of an
        mlip force field class.

        Args:
            atoms: The ASE atoms.
            ase_calculator: The ASE calculator to use in the simulation.
            config: The simulation config.
        """
        self.state = SimulationState()
        self.loggers: list[Callable[[SimulationState], None]] = []

        logger.debug("Initialization of simulation begins...")
        self._config = config
        self.atoms = atoms
        self.atoms.center()
        positions = atoms.get_positions()
        self._num_atoms = positions.shape[0]
        self.state.atomic_numbers = atoms.numbers

        self._init_box()

        self.model_calculator = ase_calculator

        self._temperature_schedule = get_temperature_schedule(
            self._config.temperature_schedule_config, self._config.num_steps
        )

        logger.debug("Initialization of simulation completed.")


def get_simulation_engine(
    atoms: ase.Atoms, force_field: ForceField | ASECalculator, **kwargs
) -> JaxMDSimulationEngine | ASESimulationEngineWithCalculator | ASESimulationEngine:
    """Returns the correct simulation engine based on the input force field type.

    For MD simulations with `mlip.models.ForceField` objects, we return a
    `JaxMDSimulationEngine`. For energy minimizations with those objects, we return
    a `ASESimulationEngine`. For any type of simulations with ASE calculator objects,
    we return a `ASESimulationEngineWithCalculator`, which is a custom class of
    the `mlipaudit` library.

    Args:
        atoms: The ASE atoms.
        force_field: The force field, either an `mlip.models.ForceField`
                     or an ASE calculator.
        **kwargs: Keyword arguments to be passed to the MD config object. Assumed to
                  be JAX-MD based, will be modified automatically for ASE.

    Returns:
        The simulation engine.

    Raises:
        ValueError: If force field type is not compatible.
    """
    # Case 1: MD simulations with ForceField objects -> use JAX-MD
    if (
        isinstance(force_field, ForceField)
        and kwargs.get("simulation_type", "md") == "md"
    ):
        md_config = JaxMDSimulationEngine.Config(**kwargs)
        return JaxMDSimulationEngine(atoms, force_field, md_config)

    kwargs_copy = deepcopy(kwargs)
    kwargs_copy.pop("num_episodes", None)  # remove this if exists

    # Case 2: Minimization with ForceField objects -> use ASE
    if isinstance(force_field, ForceField):
        minimization_config = ASESimulationEngine.Config(**kwargs_copy)
        return ASESimulationEngine(atoms, force_field, minimization_config)

    # Case 3: MD or minimization with ASECalculator objects -> use ASE
    if isinstance(force_field, ASECalculator):
        sim_config = ASESimulationEngine.Config(**kwargs_copy)
        return ASESimulationEngineWithCalculator(atoms, force_field, sim_config)

    raise ValueError(
        "Provided force field must be either a mlip-compatible "
        "force field object or an ASE calculator."
    )
