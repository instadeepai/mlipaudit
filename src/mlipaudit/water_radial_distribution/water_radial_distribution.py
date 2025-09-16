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

import mdtraj as md
import numpy as np
from ase import Atoms, units
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from mlip.simulation.configs import JaxMDSimulationConfig
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import ConfigDict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils.trajectory_helpers import (
    create_mdtraj_trajectory_from_simulation_state,
)

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

WATERBOX_N500 = "water_box_n500_eq.pdb"
REFERENCE_DATA = "experimental_reference.npz"

WATER_RADIAL_DISTRIBUTION_THRESHOLDS = {"rmse": 0.1}


class WaterRadialDistributionModelOutput(ModelOutput):
    """Model output containing the final simulation state of
    the water box.

    Attributes:
        simulation_state: The final simulation state of the water
            box simulation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    simulation_state: SimulationState


class WaterRadialDistributionResult(BenchmarkResult):
    """Result object for the water radial distribution benchmark.

    Attributes:
        radii: The radii values in Angstrom.
        rdf: The radial distribution function values at the
            radii.
        mae: The MAE of the radial distribution function values.
        rmse: The RMSE of the radial distribution function values.
        score: The final score for the benchmark between
            0 and 1.
    """

    radii: list[float]
    rdf: list[float]
    mae: float
    rmse: float


class WaterRadialDistributionBenchmark(Benchmark):
    """Benchmark for water radial distribution function.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `water_radial_distribution`.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `WaterRadialDistributionResult`.
        model_output_class: A reference to
            the `WaterRadialDistributionModelOutput` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
    """

    name = "water_radial_distribution"
    result_class = WaterRadialDistributionResult
    model_output_class = WaterRadialDistributionModelOutput

    required_elements = {"H", "O"}

    def run_model(self) -> None:
        """Run an MD simulation for each structure.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. NOTE: This benchmark runs a simulation in the
        NVT ensemble, which is not recommended for a water RDF calculation.
        """
        logger.info("Running MD for for water radial distribution function.")

        md_engine = JaxMDSimulationEngine(
            atoms=self._water_box_n500,
            force_field=self.force_field,
            config=self._md_config,
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
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        box_length = self._md_config.box

        traj = create_mdtraj_trajectory_from_simulation_state(
            self.model_output.simulation_state,
            self.data_input_dir / self.name / WATERBOX_N500,
            cell_lengths=(box_length, box_length, box_length),
        )

        oxygen_indices = traj.top.select("symbol == O")

        # converting length units to nm for mdtraj
        bin_centers = self._reference_data["r_OO"] * (units.Angstrom / units.nm)
        bin_width = bin_centers[1] - bin_centers[0]

        radii, g_r = md.compute_rdf(
            traj,
            pairs=traj.topology.select_pairs(oxygen_indices, oxygen_indices),
            r_range=(bin_centers[0] - bin_width / 2, bin_centers[-1] + bin_width),
            bin_width=bin_width,
        )

        # converting length units back to angstrom
        radii = (radii * (units.nm / units.Angstrom)).tolist()
        rdf = g_r.tolist()
        mae = mean_absolute_error(g_r, self._reference_data["g_OO"])
        rmse = root_mean_squared_error(g_r, self._reference_data["g_OO"])

        score = compute_benchmark_score(
            [rmse], [WATER_RADIAL_DISTRIBUTION_THRESHOLDS["rmse"]]
        )

        return WaterRadialDistributionResult(
            radii=radii, rdf=rdf, mae=mae, rmse=rmse, score=score
        )

    @functools.cached_property
    def _md_config(self) -> JaxMDSimulationConfig:
        if self.fast_dev_run:
            return JaxMDSimulationConfig(**SIMULATION_CONFIG_FAST)

        return JaxMDSimulationConfig(**SIMULATION_CONFIG)

    @functools.cached_property
    def _water_box_n500(self) -> Atoms:
        return ase_read(self.data_input_dir / self.name / WATERBOX_N500)

    @functools.cached_property
    def _reference_data(self):
        return np.load(self.data_input_dir / self.name / REFERENCE_DATA)
