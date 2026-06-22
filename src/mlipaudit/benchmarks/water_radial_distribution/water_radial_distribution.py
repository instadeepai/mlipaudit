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
from mlip.simulation.enums import MDIntegrator
from pydantic import ConfigDict, NonNegativeFloat
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mlipaudit.benchmark import (
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    Benchmark,
    BenchmarkResult,
    ModelOutput,
)
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import ALPHA, compute_metric_score
from mlipaudit.utils import run_simulation
from mlipaudit.utils.molecular_liquids import compute_densities
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
    "pressure_bar": 1.01325,
}

SIMULATION_CONFIG_FAST = {
    "num_steps": 250_000,
    "snapshot_interval": 250,
    "num_episodes": 1000,
    "temperature_kelvin": 295.15,
    "pressure_bar": 1.01325,
}

SIMULATION_CONFIG_DEV = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 295.15,
    "pressure_bar": 1.01325,
}

WATERBOX_N500 = "water_box_n500_eq.pdb"
MOLECULE_INDICES_PATH = "water_box_n500_molecule_indices.npy"
REFERENCE_DATA = "experimental_reference.npz"

MOLECULE_WEIGHT = 18.01528  # g/mol
ATOMS_PER_MOLECULE = 3
REFERENCE_PEAK_DISTANCE = 2.80  # A
RMSE_SCORE_THRESHOLD = 0.1
SOLVENT_PEAK_RANGE = (2.8, 3.0)
RADII_RANGE = (2.5, 10.0)

REFERENCE_DENSITY = 0.997773  # g/cm3


class WaterRadialDistributionModelOutput(ModelOutput):
    """Model output containing the final simulation state of the water box.

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
        densities: List of densities in g/cm3.
        average_density: Average density over the final 4 fifths of the frames.
        density_deviation: Deviation of the average density from the reference.
        radii: The radii values in Angstrom.
        rdf: The radial distribution function values at the radii.
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
        score: The final score for the benchmark between 0 and 1.
    """

    densities: list[float] | None = None
    average_density: float | None = None
    density_deviation: NonNegativeFloat | None = None
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
        """Run an MD simulation for the water box system using the NPT ensemble.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. The NPT integrator uses Langevin dynamics with
        a Monte Carlo barostat.
        """
        logger.info("Running MD for water radial distribution function.")

        simulation_state = run_simulation(
            atoms=self._water_box_n500,
            force_field=self.force_field,
            md_integrator=MDIntegrator.NPT_MC_LANGEVIN,
            molecule_indices=self._molecule_indices,
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

        simulation_state = self.model_output.simulation_state

        if self.model_output.failed or not is_simulation_stable(simulation_state):
            return WaterRadialDistributionResult(failed=True, score=0.0)

        densities = compute_densities(
            simulation_state, MOLECULE_WEIGHT, ATOMS_PER_MOLECULE
        )
        n_frames_equilibration = len(densities) // 5
        average_density = np.mean(densities[n_frames_equilibration:])
        density_deviation = abs(average_density - REFERENCE_DENSITY)

        traj = create_mdtraj_trajectory_from_simulation_state(
            simulation_state,
            self.data_input_dir / self.name / WATERBOX_N500,
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

        peak_deviation = abs(first_solvent_peak - REFERENCE_PEAK_DISTANCE)

        peak_deviation_score = math.exp(
            -ALPHA * peak_deviation / REFERENCE_PEAK_DISTANCE
        )

        rmse_score = compute_metric_score(
            np.array([rmse]), RMSE_SCORE_THRESHOLD, ALPHA
        ).item()

        score = (peak_deviation_score + rmse_score) / 2

        # TODO: Remove `range_of_interest=SOLVENT_PEAK_RANGE`?
        return WaterRadialDistributionResult(
            densities=densities,
            average_density=average_density,
            density_deviation=density_deviation,
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
        atoms = ase_read(self.data_input_dir / self.name / WATERBOX_N500)
        atoms.info["charge"] = DEFAULT_CHARGE
        atoms.info["spin"] = DEFAULT_SPIN
        return atoms

    @functools.cached_property
    def _molecule_indices(self) -> np.ndarray:
        molecule_indices = np.load(
            self.data_input_dir / self.name / MOLECULE_INDICES_PATH
        )
        return molecule_indices

    @functools.cached_property
    def _reference_data(self):
        """The experimental reference data for the water RDF benchmark.

        Contains keys 'r_OO' and 'g_OO', the radii and RDF values.
        The radii are in Angstrom.
        """
        return np.load(self.data_input_dir / self.name / REFERENCE_DATA)
