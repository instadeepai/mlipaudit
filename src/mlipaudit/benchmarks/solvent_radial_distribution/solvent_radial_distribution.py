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
import math
import statistics

import mdtraj as md
import numpy as np
from ase import Atoms, units
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.run_mode import RunMode
from mlipaudit.scoring import ALPHA
from mlipaudit.utils import (
    create_mdtraj_trajectory_from_simulation_state,
    run_simulation,
)
from mlipaudit.utils.stability import is_simulation_stable

logger = logging.getLogger("mlipaudit")

SIMULATION_CONFIG = {
    "num_steps": 500_000,
    "snapshot_interval": 500,
    "num_episodes": 1000,
    "temperature_kelvin": 295.15,
}

SIMULATION_CONFIG_DEV = {
    "num_steps": 5,
    "snapshot_interval": 1,
    "num_episodes": 1,
    "temperature_kelvin": 295.15,
}
SIMULATION_CONFIG_FAST = {
    "num_steps": 250_000,
    "snapshot_interval": 250,
    "num_episodes": 1000,
    "temperature_kelvin": 295.15,
}
NUM_DEV_SYSTEMS = 1

BOX_CONFIG = {  # In Angstrom
    "CCl4": 28.575,
    "methanol": 29.592,
    "acetonitrile": 27.816,
}

SYSTEM_ATOM_OF_INTEREST = {
    "CCl4": "C",
    "methanol": "O",
    "acetonitrile": "N",
}

MIN_RADII, MAX_RADII = 0.0, 20.0  # In Angstrom

REFERENCE_MAXIMA = {
    "CCl4": {"type": "C-C", "distance": 5.9, "range": (0.0, 20.0)},
    "acetonitrile": {"type": "N-N", "distance": 4.0, "range": (3.5, 4.5)},
    "methanol": {"type": "O-O", "distance": 2.8, "range": (0.0, 20.0)},
}
RANGES_OF_INTEREST = {
    "CCl4": (0.0, 20.0),
    "acetonitrile": (3.5, 4.5),
    "methanol": (0.0, 20.0),
}


class SolventRadialDistributionModelOutput(ModelOutput):
    """Model output containing the final simulation states for
    each structure.

    Attributes:
        structure_names: The names of the structures.
        simulation_states: `SimulationState` or `None` object for
            each structure in the same order as the structure
            names. `None` if the simulation failed.
    """

    structure_names: list[str]
    simulation_states: list[SimulationState | None]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SolventRadialDistributionStructureResult(BaseModel):
    """Stores the result for a single structure.

    Attributes:
        structure_name: The structure name.
        radii: The radii values in Angstrom.
        rdf: The radial distribution function values at the
            radii.
        first_solvent_peak: The first solvent peak, i.e.
            the radius at which the rdf is the maximum.
        peak_deviation: The deviation of the
            first solvent peak from the reference.
        failed: Whether the simulation was successful. If unsuccessful, the other
            attributes will be not be set.
        score: The score for the molecule.
    """

    structure_name: str
    radii: list[float] | None = None
    rdf: list[float] | None = None
    first_solvent_peak: float | None = None
    peak_deviation: NonNegativeFloat | None = None

    failed: bool = False
    score: float = 0.0


class SolventRadialDistributionResult(BenchmarkResult):
    """Result object for the solvent radial distribution benchmark.

    Attributes:
        structure_names: The names of the structures.
        structures: List of per structure results.
        avg_peak_deviation: The average deviation across all structures.
        failed: Whether all the simulations failed and no analysis could be
            performed. Defaults to False.
        score: The final score for the benchmark between
            0 and 1.
    """

    structure_names: list[str]
    structures: list[SolventRadialDistributionStructureResult]
    avg_peak_deviation: NonNegativeFloat | None = None


class SolventRadialDistributionBenchmark(Benchmark):
    """Benchmark for solvent radial distribution function.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `solvent_radial_distribution`.
        category: A string that describes the category of the benchmark, used for
            example, in the UI app for grouping. Default, if not overridden,
            is "General". This benchmark's category is "Molecular Liquids".
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `SolventRadialDistributionResult`.
        model_output_class: A reference to
            the `SolventRadialDistributionModelOutput` class.
        required_elements: The set of atomic element types that are present in the
            benchmark's input files.
        skip_if_elements_missing: Whether the benchmark should be skipped entirely
            if there are some atomic element types that the model cannot handle. If
            False, the benchmark must have its own custom logic to handle missing atomic
            element types. For this benchmark, the attribute is set to True.
    """

    name = "solvent_radial_distribution"
    category = "Molecular Liquids"
    result_class = SolventRadialDistributionResult
    model_output_class = SolventRadialDistributionModelOutput

    required_elements = {"N", "H", "O", "C", "Cl"}

    def run_model(self) -> None:
        """Run an MD simulation for each structure.
        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. NOTE: This benchmark runs a simulation in the
        NVT ensemble, which is not recommended for a water RDF calculation.
        """
        if self.run_mode == RunMode.DEV:
            md_kwargs = SIMULATION_CONFIG_DEV
        elif self.run_mode == RunMode.FAST:
            md_kwargs = SIMULATION_CONFIG_FAST
        else:
            md_kwargs = SIMULATION_CONFIG

        simulation_states = []
        for system_name in self._system_names:
            logger.info("Running MD for %s radial distribution function.", system_name)

            md_kwargs["box"] = BOX_CONFIG[system_name]
            simulation_state = run_simulation(
                atoms=self._load_system(system_name),
                force_field=self.force_field,
                **md_kwargs,
            )

            simulation_states.append(simulation_state)

        self.model_output = SolventRadialDistributionModelOutput(
            structure_names=self._system_names, simulation_states=simulation_states
        )

    def analyze(self) -> SolventRadialDistributionResult:
        """Calculate how much the radial distribution deviates from the reference.

        Returns:
            A `SolventRadialDistributionResult` object.

        Raises:
            RuntimeError: If called before `run_model()`.
        """
        if self.model_output is None:
            raise RuntimeError("Must call run_model() first.")

        structure_results = []

        num_succeeded = 0

        for system_name, simulation_state in zip(
            self.model_output.structure_names, self.model_output.simulation_states
        ):
            if simulation_state is None or not is_simulation_stable(simulation_state):
                structure_result = SolventRadialDistributionStructureResult(
                    structure_name=system_name,
                    failed=True,
                    score=0.0,
                )
                structure_results.append(structure_result)
                continue

            num_succeeded += 1

            box_length = BOX_CONFIG[system_name]

            traj = create_mdtraj_trajectory_from_simulation_state(
                simulation_state=simulation_state,
                topology_path=self.data_input_dir
                / self.name
                / self._get_pdb_file_name(system_name),
                cell_lengths=(box_length, box_length, box_length),
            )
            pair_indices = traj.top.select(
                f"symbol == {SYSTEM_ATOM_OF_INTEREST[system_name]}"
            )

            # converting length units to nm for mdtraj
            bin_centers = np.arange(
                MIN_RADII * (units.Angstrom / units.nm),
                MAX_RADII * (units.Angstrom / units.nm),
                0.001,
            )
            bin_width = bin_centers[1] - bin_centers[0]

            # Get the radii and the RDF evaluated at the radii
            radii, g_r = md.compute_rdf(
                traj,
                pairs=traj.topology.select_pairs(pair_indices, pair_indices),
                r_range=(
                    bin_centers[0] - bin_width / 2,
                    bin_centers[-1] + bin_width / 2,
                ),
                n_bins=2000,
            )

            # converting length units back to angstrom
            radii = radii * (units.nm / units.Angstrom)
            rdf = g_r.tolist()

            radii_min, radii_max = RANGES_OF_INTEREST[system_name]
            range_of_interest = np.where((radii > radii_min) & (radii <= radii_max))
            first_solvent_peak = radii[range_of_interest][
                np.argmax(g_r[range_of_interest])
            ].item()

            peak_deviation = abs(
                first_solvent_peak - REFERENCE_MAXIMA[system_name]["distance"]
            )
            score = math.exp(
                -ALPHA * peak_deviation / REFERENCE_MAXIMA[system_name]["distance"]
            )

            structure_result = SolventRadialDistributionStructureResult(
                structure_name=system_name,
                radii=radii.tolist(),
                rdf=rdf,
                first_solvent_peak=first_solvent_peak,
                peak_deviation=peak_deviation,
                score=score,
            )

            structure_results.append(structure_result)

        if num_succeeded == 0:
            return SolventRadialDistributionResult(
                structure_names=self.model_output.structure_names,
                structures=structure_results,
                failed=True,
                score=0.0,
            )

        return SolventRadialDistributionResult(
            structure_names=self.model_output.structure_names,
            structures=structure_results,
            avg_peak_deviation=statistics.mean(
                structure.peak_deviation
                for structure in structure_results
                if structure.peak_deviation is not None
            ),
            score=statistics.mean(
                r.score if r.score is not None else 0.0 for r in structure_results
            ),
        )

    @property
    def _system_names(self) -> list[str]:
        if self.run_mode == RunMode.STANDARD:
            return list(BOX_CONFIG.keys())

        # reduced number of cases for DEV and FAST run mode
        return list(BOX_CONFIG.keys())[:NUM_DEV_SYSTEMS]

    def _load_system(self, system_name) -> Atoms:
        return ase_read(
            self.data_input_dir / self.name / self._get_pdb_file_name(system_name)
        )

    @staticmethod
    def _get_pdb_file_name(system_name: str) -> str:
        return f"{system_name}_eq.pdb"
