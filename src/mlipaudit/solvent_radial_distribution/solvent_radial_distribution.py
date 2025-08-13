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

from ase import Atoms
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from mlip.simulation.configs import JaxMDSimulationConfig
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import ConfigDict

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput

logger = logging.getLogger("mlipaudit")

SIMULATION_CONFIG = {
    "num_steps": 500_000,
    "snapshot_interval": 500,
    "num_episodes": 1000,
    "temperature_kelvin": 295.15,
}

SIMULATION_CONFIG_FAST = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 295.15,
}
BOX_CONFIG = {
    "CCl4": 28.575,
    "methanol": 29.592,
    "acetonitrile": 27.816,
}


class SolventRadialDistributionModelOutput(ModelOutput):
    """Model output containg the final simulation state of
    the water box.

    Attributes:
        simulation_state: The final simulation state of the water
            box simulation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    structure_names: list[str]
    simulation_states: list[SimulationState]


class SolventRadialDistributionResult(BenchmarkResult):
    """Result object for the water radial distribution benchmark.

    Attributes:
        radii: The radii values in Angstrom.
        rdf: The radial distribution function values at the
            radii.
        mae: The MAE of the radial distribution function values.
        rmse: The RMSE of the radial distribution function values.
    """

    radii: list[float]
    rdf: list[float]
    mae: float
    rmse: float


class SolventRadialDistributionBenchmark(Benchmark):
    """Benchmark for solvent radial distribution function.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `solvent_radial_distribution`.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `SolventRadialDistributionResult`.
    """

    name = "solvent_radial_distribution"
    result_class = SolventRadialDistributionResult

    model_output: SolventRadialDistributionModelOutput

    def run_model(self) -> None:
        """Run an MD simulation for each structure.
        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. NOTE: This benchmark runs a simulation in the
        NVT ensemble, which is not recommended for a water RDF calculation.
        """
        simulation_states = []
        for system_name in self._system_names:
            logger.info("Running MD for %s radial distribution function.", system_name)

            md_config = (
                JaxMDSimulationConfig(**SIMULATION_CONFIG)
                if not self.fast_dev_run
                else JaxMDSimulationConfig(**SIMULATION_CONFIG_FAST)
            )
            md_engine = JaxMDSimulationEngine(
                atoms=self._load_system(system_name),
                force_field=self.force_field,
                config=md_config,
            )
            md_engine.run()
            simulation_states.append(md_engine.state)

        self.model_output = SolventRadialDistributionModelOutput(
            structure_names=self._system_names, simulation_states=simulation_states
        )

    @property
    def _system_names(self) -> list[str]:
        if not self.fast_dev_run:
            return list(BOX_CONFIG.keys())

        return list(BOX_CONFIG.keys())[:1]

    def _load_system(self, system_name) -> Atoms:
        return ase_read(self.data_input_dir / self.name / system_name)
