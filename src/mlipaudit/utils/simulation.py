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
from copy import deepcopy
from typing import TYPE_CHECKING

import ase
from ase.calculators.calculator import Calculator as ASECalculator
from mlip.simulation import SimulationState

if TYPE_CHECKING:
    # These pull in the heavy mlip/JAX-MD stack (~1.5s). They are imported lazily
    # below so that importing this module (e.g. via `mlipaudit.utils`) stays cheap.
    from mlip.models import ForceField
    from mlip.simulation.ase import ASESimulationEngine
    from mlip.simulation.jax_md import JaxMDSimulationEngine

    from mlipaudit.utils._ase_engine import ASESimulationEngineWithCalculator

REUSABLE_BIOMOLECULES_OUTPUTS_ID = ("sampling", "folding_stability")

logger = logging.getLogger("mlipaudit")


def __getattr__(name: str):
    # PEP 562 module-level attribute access. Keeps the historical
    # `mlipaudit.utils.simulation.ASESimulationEngineWithCalculator` import path
    # working without pulling in the heavy mlip stack at module import time (the
    # class lives in `_ase_engine`, whose import is expensive).
    if name == "ASESimulationEngineWithCalculator":
        from mlipaudit.utils._ase_engine import (  # noqa: PLC0415
            ASESimulationEngineWithCalculator,
        )

        return ASESimulationEngineWithCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Imported lazily: these pull in the heavy mlip/JAX-MD stack, which we only want
    # to pay for when a simulation is actually run (not at import time).
    from mlip.models import ForceField  # noqa: PLC0415
    from mlip.simulation.ase import ASESimulationEngine  # noqa: PLC0415
    from mlip.simulation.jax_md import JaxMDSimulationEngine  # noqa: PLC0415

    # Case 1: MD simulations with ForceField objects -> use JAX-MD
    if (
        isinstance(force_field, ForceField)
        and kwargs.get("simulation_type", "md") == "md"
    ):
        md_config = JaxMDSimulationEngine.Config(**kwargs)
        # Log the number of steps that will be run and for how many episodes
        logger.info(
            "Running MD simulation for %d steps and %d episodes.",
            md_config.num_steps,
            md_config.num_episodes,
        )
        return JaxMDSimulationEngine(atoms, force_field, md_config)

    kwargs_copy = deepcopy(kwargs)
    kwargs_copy.pop("num_episodes", None)  # remove this if exists

    # Case 2: Minimization with ForceField objects -> use ASE
    if isinstance(force_field, ForceField):
        minimization_config = ASESimulationEngine.Config(**kwargs_copy)
        logger.info(
            "Running energy minimization with ASE for a maximum of %d steps.",
            minimization_config.num_steps,
        )
        return ASESimulationEngine(atoms, force_field, minimization_config)

    # Case 3: MD or minimization with ASECalculator objects -> use ASE
    if isinstance(force_field, ASECalculator):
        sim_config = ASESimulationEngine.Config(**kwargs_copy)
        logger.info(
            "Running ASE-based simulation for a maximum of %d steps.",
            sim_config.num_steps,
        )
        from mlipaudit.utils._ase_engine import (  # noqa: PLC0415
            ASESimulationEngineWithCalculator,
        )

        return ASESimulationEngineWithCalculator(atoms, force_field, sim_config)

    raise ValueError(
        "Provided force field must be either a mlip-compatible "
        "force field object or an ASE calculator."
    )


def run_simulation(
    atoms: ase.Atoms, force_field: ForceField | ASECalculator, **kwargs
) -> SimulationState | None:
    """Run the simulation with the appropriate simulation engine based on the input
    force field type.

    For MD simulations with `mlip.models.ForceField` objects, runs the simulation with
    `JaxMDSimulationEngine`. For energy minimizations with those objects, runs with
    an `ASESimulationEngine`. For any type of simulations with ASE calculator objects,
    runs with an `ASESimulationEngineWithCalculator`, which is a custom class of
    the `mlipaudit` library. If the simulation fails, the error will be caught and
    None will be returned.

    Args:
        atoms: The ASE atoms.
        force_field: The force field, either an `mlip.models.ForceField`
                     or an ASE calculator.
        **kwargs: Keyword arguments to be passed to the MD config object. Assumed to
                  be JAX-MD based, will be modified automatically for ASE.

    Returns:
        The simulation state or None if the simulation failed.

    Raises:
        ValueError: If force field type is not compatible.
    """
    engine = get_simulation_engine(atoms=atoms, force_field=force_field, **kwargs)

    try:
        engine.run()
        return engine.state

    except Exception as e:
        logger.info("Error running simulation on system %s: %s", str(atoms), str(e))
        return None
