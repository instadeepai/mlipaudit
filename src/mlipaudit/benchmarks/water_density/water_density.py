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

import numpy as np
from mlip.simulation import SimulationState
from pydantic import ConfigDict, NonNegativeFloat

from mlipaudit.benchmark import (
    Benchmark,
    BenchmarkResult,
    ModelOutput,
)
from mlipaudit.scoring import ALPHA, compute_metric_score
from mlipaudit.utils.molecular_liquids import (
    DENSITY_RELATIVE_DEVIATION_THRESHOLD,
    WATER_ATOMS_PER_MOLECULE,
    WATER_DATA_NAME,
    WATER_MOLECULE_WEIGHT,
    WATER_REFERENCE_DENSITY,
    WATER_REUSABLE_OUTPUT_ID,
    average_equilibrated_density,
    compute_densities,
    run_water_npt_simulation,
)
from mlipaudit.utils.stability import is_simulation_stable

logger = logging.getLogger("mlipaudit")


class WaterDensityModelOutput(ModelOutput):
    """Model output containing the final simulation state of the water box.

    Shares its field signature with `WaterRadialDistributionModelOutput` so that the
    NPT simulation can be reused between the two benchmarks (see `reusable_output_id`).

    Attributes:
        simulation_state: The final simulation state of the water
            box simulation. None if the simulation failed.
        failed: Whether the simulation failed. Defaults to False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    simulation_state: SimulationState | None = None
    failed: bool = False


class WaterDensityResult(BenchmarkResult):
    """Result object for the water density benchmark.

    Attributes:
        densities: List of per-frame densities in g/cm3.
        average_density: Average density over the final four fifths of the frames.
        density_deviation: Absolute deviation of the average density from the
            experimental reference.
        reference_density: The experimental reference density in g/cm3.
        failed: Whether the simulation failed and no analysis could be
            performed. Defaults to False.
        score: The final score for the benchmark between 0 and 1.
    """

    densities: list[float] | None = None
    average_density: float | None = None
    density_deviation: NonNegativeFloat | None = None
    reference_density: float = WATER_REFERENCE_DENSITY


class WaterDensityBenchmark(Benchmark):
    """Benchmark for the equilibrium density of liquid water.

    Runs the same water box NPT simulation as `WaterRadialDistributionBenchmark`
    (reusing its output when both are run together) and scores how closely the
    equilibrium density matches the experimental reference.

    Attributes:
        name: The unique benchmark name. The name is `water_density`.
        category: The benchmark's category, `"Molecular Liquids"`.
        data_name: The input-data directory shared with the water RDF benchmark.
        result_class: `WaterDensityResult`.
        model_output_class: `WaterDensityModelOutput`.
        required_elements: The set of atomic element types present in the input files.
        reusable_output_id: Shared with `WaterRadialDistributionBenchmark` so the water
            box NPT simulation is only run once when both benchmarks are run together.
    """

    name = "water_density"
    category = "Molecular Liquids"
    data_name = WATER_DATA_NAME
    result_class = WaterDensityResult
    model_output_class = WaterDensityModelOutput

    required_elements = {"H", "O"}

    reusable_output_id = WATER_REUSABLE_OUTPUT_ID

    def run_model(self) -> None:
        """Run an MD simulation for the water box system using the NPT ensemble.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. The NPT integrator uses Langevin dynamics with
        a Monte Carlo barostat.
        """
        simulation_state = run_water_npt_simulation(
            self.force_field, self.data_dir, self.run_mode
        )
        self.model_output = WaterDensityModelOutput(
            simulation_state=simulation_state, failed=simulation_state is None
        )

    def analyze(self) -> WaterDensityResult:
        """Compute the equilibrium density and how much it deviates from the reference.

        Returns:
            A `WaterDensityResult` object.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        simulation_state = self.model_output.simulation_state

        if self.model_output.failed or not is_simulation_stable(simulation_state):
            return WaterDensityResult(failed=True, score=0.0)

        densities = compute_densities(
            simulation_state, WATER_MOLECULE_WEIGHT, WATER_ATOMS_PER_MOLECULE
        )
        average_density = average_equilibrated_density(densities)
        density_deviation = abs(average_density - WATER_REFERENCE_DENSITY)

        relative_deviation = density_deviation / WATER_REFERENCE_DENSITY
        score = compute_metric_score(
            np.array([relative_deviation]),
            DENSITY_RELATIVE_DEVIATION_THRESHOLD,
            ALPHA,
        ).item()

        return WaterDensityResult(
            densities=densities.tolist(),
            average_density=average_density,
            density_deviation=density_deviation,
            score=score,
        )
