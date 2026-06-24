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
"""Custom ASE simulation engine.

This module is kept separate from :mod:`mlipaudit.utils.simulation` because
importing it pulls in the heavy ``mlip.simulation.ase`` (and therefore JAX-MD)
stack. It is only imported lazily, when a simulation is actually run, so that
merely importing the benchmark modules (e.g. for the CLI) stays fast.
"""

import logging
from typing import Callable

import ase
from ase.calculators.calculator import Calculator as ASECalculator
from mlip.simulation import SimulationState
from mlip.simulation.ase import ASESimulationEngine
from mlip.simulation.configs import ASESimulationConfig
from mlip.simulation.enums import SimulationType
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

        self.is_md_simulation = self._config.simulation_type == SimulationType.MD
        self.is_npt_simulation = (
            self.is_md_simulation and self._config.md_integrator.ensemble == "npt"
        )

        self.model_calculator = ase_calculator

        self._temperature_schedule = get_temperature_schedule(
            self._config.temperature_schedule_config, self._config.num_steps
        )

        logger.debug("Initialization of simulation completed.")
