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
import time
from typing import Callable

import ase
import numpy as np
from ase.calculators.calculator import Calculator as ASECalculator
from ase.mep import NEB
from ase.optimize import BFGS
from mlip.models import ForceField
from mlip.simulation import SimulationState
from mlip.simulation.ase.mlip_ase_calculator import MLIPForceFieldASECalculator
from mlip.simulation.configs import ASESimulationConfig
from mlip.simulation.simulation_engine import SimulationEngine

logger = logging.getLogger("mlip")


class NEBSimulationConfig(ASESimulationConfig):
    """Configuration for the NEB simulations.
    Also includes the attributes of the parent class
    :ASESimulationConfig.
    """

    simulation_type: str = "neb"
    num_images: int = 7
    neb_k: float | None = 10.0
    max_force_convergence_threshold: float | None = 0.1
    continue_from_previous_run: bool = False
    climb: bool = False


class NEBSimulationEngine(SimulationEngine):
    """Simulation engine handling NEB simulations with the ASE backend."""

    Config = NEBSimulationConfig

    def __init__(
        self,
        atoms_initial: ase.Atoms,
        atoms_final: ase.Atoms,
        force_field: ForceField | ASECalculator,
        config: NEBSimulationConfig,
        images: list[ase.Atoms] | None = None,
        transition_state: ase.Atoms | None = None,
    ) -> None:
        """Initialize the NEB simulation engine."""
        self._initialize(
            atoms_initial,
            atoms_final,
            force_field,
            config,
            images,
            transition_state,
        )

    def _initialize(
        self,
        atoms_initial: ase.Atoms,
        atoms_final: ase.Atoms,
        force_field: ForceField | ASECalculator,
        config: NEBSimulationConfig,
        images: list[ase.Atoms] | None = None,
        transition_state: ase.Atoms | None = None,
    ) -> None:
        """Initialize the NEB simulation."""
        self.state = SimulationState()
        self.loggers: list[Callable[[SimulationState], None]] = []

        self._config = config
        self.atoms = atoms_initial
        positions = atoms_initial.get_positions()
        self._num_atoms = positions.shape[0]
        self.state.atomic_numbers = atoms_initial.numbers
        self.force_field = force_field

        self.model_calculator = self._get_model_calculator()

        self.atoms_final = atoms_final

        self._init_box_neb(self.atoms)
        self._init_box_neb(self.atoms_final)

        self.atoms.calc = self._get_model_calculator()
        self.atoms_final.calc = self._get_model_calculator()

        self.neb = NEB([])
        self.images = images
        self.transition_state = transition_state

    def run(self) -> None:
        """Run the NEB simulation.

        Raises:
            ValueError: If continue_from_previous_run is True
            and images are not provided.
        """
        if not self._config.continue_from_previous_run:
            self._init_neb()
        else:
            if not self.images:
                raise ValueError(
                    "Images must be provided if continue_from_previous_run is True"
                )

            for image in self.images:
                image.calc = self._get_model_calculator()

            self.neb = NEB(
                self.images,
                k=self._config.neb_k,
                climb=self._config.climb,
                parallel=True,
            )

        dyn = BFGS(self.neb, alpha=70, maxstep=0.03, logfile=None)

        def log_to_console() -> None:
            """Logs info to console."""
            step = dyn.get_number_of_steps()
            compute_time = time.perf_counter() - self.self_start_interval_time
            self._log_to_console(step, compute_time)

        def set_beginning_interval_time() -> None:
            self.self_start_interval_time = time.perf_counter()

        def update_state() -> None:
            """Update the internal SimulationState object."""
            step = dyn.get_number_of_steps()
            compute_time = time.perf_counter() - self.self_start_interval_time
            self._update_state_neb(step, compute_time)

        dyn.attach(log_to_console, interval=self._config.log_interval)
        dyn.attach(self._call_loggers, interval=self._config.log_interval)
        dyn.attach(update_state, interval=self._config.snapshot_interval)
        dyn.attach(set_beginning_interval_time, interval=self._config.log_interval)
        self.self_start_interval_time = time.perf_counter()

        dyn.run(
            steps=self._config.num_steps,
            fmax=self._config.max_force_convergence_threshold,
        )

    def _init_neb(self) -> None:
        if not self.transition_state:
            num_images = max(self._config.num_images, 2)
            images = [self.atoms]
            images.extend([self.atoms.copy() for _ in range(num_images - 2)])
            images.append(self.atoms_final)
        else:
            num_images = max(self._config.num_images, 3)
            num_images_1 = num_images // 2 + 1
            num_images_2 = num_images - num_images_1 + 1

            images_1 = [self.atoms]
            images_1.extend([self.atoms.copy() for _ in range(num_images_1 - 2)])
            images_1.append(self.transition_state)

            images_2 = [self.transition_state.copy()]
            images_2.extend([self.atoms_final.copy() for _ in range(num_images_2 - 2)])
            images_2.append(self.atoms_final)

            for image in images_1:
                image.calc = self._get_model_calculator()
            for image in images_2:
                image.calc = self._get_model_calculator()

            neb1 = NEB(
                images_1, k=self._config.neb_k, climb=self._config.climb, parallel=True
            )
            neb2 = NEB(
                images_2, k=self._config.neb_k, climb=self._config.climb, parallel=True
            )

            neb1.interpolate(method="idpp")
            neb2.interpolate(method="idpp")

            images = neb1.images + neb2.images[1:]

        for image in images:
            image.calc = self._get_model_calculator()

        self.neb = NEB(
            images, k=self._config.neb_k, climb=self._config.climb, parallel=True
        )

        if not self.transition_state:
            self.neb.interpolate(method="idpp")

    def _init_box_neb(self, atoms: ase.Atoms) -> None:
        if isinstance(self._config.box, float):
            atoms.cell = np.eye(3) * self._config.box
            atoms.pbc = True
        elif isinstance(self._config.box, list):
            atoms.cell = np.diag(np.array(self._config.box))
            atoms.pbc = True
        else:
            atoms.cell = None
            atoms.pbc = False

    def _update_state_neb(
        self,
        step: int,
        compute_time: float,
    ) -> None:
        """Update the internal state of the simulation.
        Here, the positions, forces and potential energy for every image
        are updated and not concatenated, as for the MD simulations and energy
        minimizations.

        Args:
            step: The current step.
            compute_time: The compute time.
        """
        self.state.positions = np.zeros((
            len(self.neb.images),
            len(self.neb.images[0].positions),
            3,
        ))
        self.state.potential_energy = np.zeros(len(self.neb.images))

        for i, image in enumerate(self.neb.images):
            self.state.positions[i] = image.positions
            self.state.potential_energy[i] = image.get_potential_energy()

        self.state.forces = self.neb.get_forces()

        self.state.step = step
        self.state.compute_time_seconds += compute_time

    def _get_model_calculator(self) -> MLIPForceFieldASECalculator | ASECalculator:
        if isinstance(self.force_field, ForceField):
            return MLIPForceFieldASECalculator(
                self.atoms,
                self._config.edge_capacity_multiplier,
                self.force_field,
            )
        else:
            return self.force_field

    def _call_loggers(self) -> None:
        for _logger in self.loggers:
            _logger(self.state)

    def _log_to_console(self, step: int, compute_time: float) -> None:
        """Logs timing information to console via our logger."""
        if step == 0:
            logger.debug(
                "Initialization took %.2f seconds.",
                compute_time,
            )
        else:
            logger.info(
                "Steps %s to %s completed in %.2f seconds.",
                self.state.step,
                step,
                compute_time,
            )
