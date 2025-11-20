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
import math
from typing import Any

import mdtraj as md
import numpy as np
from ase import Atoms, units
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from pydantic import ConfigDict, NonNegativeFloat
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import ALPHA, compute_metric_score
from mlipaudit.utils import run_simulation
from mlipaudit.utils.stability import is_simulation_stable
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
    "num_steps": 250_000,
    "snapshot_interval": 250,
    "num_episodes": 1000,
    "temperature_kelvin": 295.15,
    "box": 24.772,
}

SIMULATION_CONFIG_DEV = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 295.15,
    "box": 24.772,
}

WATERBOX_N500 = "water_box_n500_eq.pdb"
REFERENCE_DATA = "experimental_reference.npz"

RMSE_SCORE_THRESHOLD = 0.1
SOLVENT_PEAK_RANGE = (2.8, 3.0)
RADII_RANGE = (2.5, 10.0)


class WaterRadialDistributionModelOutput(ModelOutput):
    """Model output containing the final simulation state of
    the water box.

    Attributes:
        simulation_state: The final simulation state of the water
            box simulation. None if the simulation failed.
        failed: Whether the simulation failed. Defaults to False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    simulation_state: SimulationState | None = None
    failed: bool = False


class WaterRadialDistributionResult(BenchmarkResult):
    """Result object for the water radial distribution benchmark.

    Attributes:
        radii: The radii values in Angstrom.
        rdf: The radial distribution function values at the
            radii.
        mae: The MAE of the radial distribution function values.
        rmse: The RMSE of the radial distribution function values.
        first_solvent_peak: The first solvent peak, i.e.
            the radius at which the rdf is the maximum.
        peak_deviation: The deviation of the
            first solvent peak from the reference.
        range_of_interest: The range of interest for the
            radial distribution function error metrics.
        failed: Whether all the simulations failed and no analysis could be
            performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    radii: list[float] | None = None
    rdf: list[float] | None = None
    mae: float | None = None
    rmse: float | None = None
    first_solvent_peak: float | None = None
    peak_deviation: NonNegativeFloat | None = None
    range_of_interest: tuple[NonNegativeFloat, NonNegativeFloat] = RADII_RANGE


class WaterRadialDistributionBenchmark(Benchmark):
    """Benchmark for water radial distribution function.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `water_radial_distribution`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Molecular Liquids".
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
    category = "Molecular Liquids"
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

        simulation_state = run_simulation(
            atoms=self._water_box_n500,
            force_field=self.force_field,
            **self._md_kwargs,
        )

        self.model_output = WaterRadialDistributionModelOutput(
            simulation_state=simulation_state, failed=simulation_state is None
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

        if self.model_output.failed or not is_simulation_stable(
            self.model_output.simulation_state
        ):
            return WaterRadialDistributionResult(failed=True, score=0.0)

        box_length = self._md_kwargs["box"]

        traj = create_mdtraj_trajectory_from_simulation_state(
            self.model_output.simulation_state,
            self.data_input_dir / self.name / WATERBOX_N500,
            cell_lengths=(box_length, box_length, box_length),
        )

        oxygen_indices = traj.top.select("symbol == O")

        # Experimental reference data in Angstrom
        exp_r = self._reference_data["r_OO"]
        exp_rdf = self._reference_data["g_OO"]

        # converting length units to nm for mdtraj
        bin_centers = exp_r * (units.Angstrom / units.nm)
        bin_width = bin_centers[1] - bin_centers[0]

        radii, g_r = md.compute_rdf(
            traj,
            pairs=traj.topology.select_pairs(oxygen_indices, oxygen_indices),
            r_range=(bin_centers[0] - bin_width / 2, bin_centers[-1] + bin_width / 2),
            n_bins=2000,
        )

        # converting length units back to Angstrom
        radii = radii * (units.nm / units.Angstrom)
        rdf = g_r.tolist()

        # Only inspect relevant range for experimental data
        exp_radii_mask = (exp_r > RADII_RANGE[0]) & (exp_r < RADII_RANGE[1])
        exp_r_filtered = exp_r[exp_radii_mask]
        exp_rdf_filtered = exp_rdf[exp_radii_mask]

        # Only inspect relevant range for predicted data
        radii_mask = (radii > RADII_RANGE[0]) & (radii < RADII_RANGE[1])
        radii_filtered = radii[radii_mask]
        rdf_filtered = g_r[radii_mask]

        # Interpolate for safety to common r grid (use experimental grid as reference)
        rdf_interp = np.interp(exp_r_filtered, radii_filtered, rdf_filtered)

        # Calculate error metrics
        mae = mean_absolute_error(rdf_interp, exp_rdf_filtered)
        rmse = root_mean_squared_error(rdf_interp, exp_rdf_filtered)

        first_solvent_peak = radii[np.argmax(g_r)].item()

        peak_deviation = max(
            0,
            SOLVENT_PEAK_RANGE[0] - first_solvent_peak,
            first_solvent_peak - SOLVENT_PEAK_RANGE[1],
        )
        peak_deviation_score = math.exp(
            -ALPHA
            * peak_deviation
            / ((SOLVENT_PEAK_RANGE[0] + SOLVENT_PEAK_RANGE[1]) / 2)
        )

        rmse_score = compute_metric_score(np.array([rmse]), RMSE_SCORE_THRESHOLD, ALPHA)
        score = (peak_deviation_score + rmse_score) / 2

        return WaterRadialDistributionResult(
            radii=radii.tolist(),
            rdf=rdf,
            mae=mae,
            rmse=rmse,
            first_solvent_peak=first_solvent_peak,
            peak_deviation=peak_deviation,
            range_of_interest=SOLVENT_PEAK_RANGE,
            score=score,
        )

    @functools.cached_property
    def _md_kwargs(self) -> dict[str, Any]:
        if self.run_mode == RunMode.DEV:
            return SIMULATION_CONFIG_DEV
        if self.run_mode == RunMode.FAST:
            return SIMULATION_CONFIG_FAST

        return SIMULATION_CONFIG

    @functools.cached_property
    def _water_box_n500(self) -> Atoms:
        return ase_read(self.data_input_dir / self.name / WATERBOX_N500)

    @functools.cached_property
    def _reference_data(self):
        """The experimental reference data for the water RDF benchmark.
        Contains keys 'r_OO' and 'g_OO', the radii and RDF values.
        The radii are in Angstrom.
        """
        return np.load(self.data_input_dir / self.name / REFERENCE_DATA)
