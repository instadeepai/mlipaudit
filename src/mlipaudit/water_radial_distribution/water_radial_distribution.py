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

from ase import Atoms
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import ConfigDict

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput

logger = logging.getLogger("mlipaudit")

SIMULATION_CONFIG = {
    "num_steps": 1_000_000,
    "snapshot_interval": 1000,
    "num_episodes": 1000,
    "temperature_kelvin": 300.0,
}

SIMULATION_CONFIG_FAST = {
    "num_steps": 10,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 300.0,
}
WATERBOX_N500 = "water_box_n500_eq.pdb"


class WaterRadialDistributionModelOutput(ModelOutput):
    """Model output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    simulation_state: SimulationState


class WaterRadialDistributionResult(BenchmarkResult):
    """Benchmark result."""

    pass


class WaterRadialDistributionBenchmark(Benchmark):
    """Benchmark for water radial distribution function.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `water_radial_distribution`.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `WaterRadialDistributionResult`.
    """

    name = "water_radial_distribution"
    result_class = WaterRadialDistributionResult

    model_output: WaterRadialDistributionModelOutput

    def run_model(self) -> None:
        """Run an MD simulation for each structure.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. NOTE: This benchmark runs a simulation in the
        NVT ensemble, which is not recommended for a water RDF calculation.
        """
        if self.fast_dev_run:
            md_config = JaxMDSimulationEngine.Config(**SIMULATION_CONFIG_FAST)
        else:
            md_config = JaxMDSimulationEngine.Config(**SIMULATION_CONFIG)

        logger.info("Running MD for for water radial distribution function.")

        md_engine = JaxMDSimulationEngine(
            atoms=self._water_box_n500, force_field=self.force_field, config=md_config
        )
        md_engine.run()

        self.model_output = WaterRadialDistributionModelOutput(
            simulation_state=md_engine.state
        )

    def analyze(self) -> WaterRadialDistributionResult:
        """Calculate how much the radial distribution deviates from the reference.

        Returns:
            A `WaterRadialDistributionResult` object.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        raise NotImplementedError

    @functools.cached_property
    def _water_box_n500(self) -> Atoms:
        return ase_read(self.data_input_dir / self.name / WATERBOX_N500)
