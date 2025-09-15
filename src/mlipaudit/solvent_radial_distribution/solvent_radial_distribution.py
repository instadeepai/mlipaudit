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
import statistics

import mdtraj as md
import numpy as np
from ase import Atoms, units
from ase.io import read as ase_read
from mlip.simulation import SimulationState
from mlip.simulation.configs import JaxMDSimulationConfig
from mlip.simulation.jax_md import JaxMDSimulationEngine
from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput
from mlipaudit.utils import create_mdtraj_trajectory_from_simulation_state

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
        simulation_states: A list of final simulation states for
            each corresponding structure.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    structure_names: list[str]
    simulation_states: list[SimulationState]


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
    """

    structure_name: str
    radii: list[float]
    rdf: list[float]
    first_solvent_peak: float
    peak_deviation: NonNegativeFloat


class SolventRadialDistributionResult(BenchmarkResult):
    """Result object for the solvent radial distribution benchmark.

    Attributes:
        structure_names: The names of the structures.
        structures: List of per structure results.
        avg_peak_deviation: The average deviation across all structures.
    """

    structure_names: list[str]
    structures: list[SolventRadialDistributionStructureResult]

    avg_peak_deviation: NonNegativeFloat


class SolventRadialDistributionBenchmark(Benchmark):
    """Benchmark for solvent radial distribution function.

    Attributes:
        name: The unique benchmark name that should be used to run the benchmark
            from the CLI and that will determine the output folder name for the result
            file. The name is `solvent_radial_distribution`.
        result_class: A reference to the type of `BenchmarkResult` that will determine
            the return type of `self.analyze()`. The result class type is
            `SolventRadialDistributionResult`.
        model_output_class: A reference to
                            the `SolventRadialDistributionModelOutput` class.
    """

    name = "solvent_radial_distribution"
    result_class = SolventRadialDistributionResult
    model_output_class = SolventRadialDistributionModelOutput

    required_elements = {"N", "H", "O", "C", "Cl"}

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
            md_config.box = BOX_CONFIG[system_name]
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

        for system_name, simulation_state in zip(
            self.model_output.structure_names, self.model_output.simulation_states
        ):
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
                r_range=(bin_centers[0] - bin_width / 2, bin_centers[-1] + bin_width),
                bin_width=bin_width,
            )

            # converting length units back to angstrom
            radii = radii * (units.nm / units.Angstrom)
            radii_min, radii_max = RANGES_OF_INTEREST[system_name]
            range_of_interest = np.where((radii > radii_min) & (radii <= radii_max))
            first_solvent_peak = radii[range_of_interest][
                np.argmax(g_r[range_of_interest])
            ].item()
            rdf = g_r.tolist()

            structure_result = SolventRadialDistributionStructureResult(
                structure_name=system_name,
                radii=radii.tolist(),
                rdf=rdf,
                first_solvent_peak=first_solvent_peak,
                peak_deviation=abs(
                    first_solvent_peak - REFERENCE_MAXIMA[system_name]["distance"]
                ),
            )

            structure_results.append(structure_result)

        return SolventRadialDistributionResult(
            structure_names=self.model_output.structure_names,
            structures=structure_results,
            avg_peak_deviation=statistics.mean(
                structure.peak_deviation for structure in structure_results
            ),
        )

    @property
    def _system_names(self) -> list[str]:
        if not self.fast_dev_run:
            return list(BOX_CONFIG.keys())

        return list(BOX_CONFIG.keys())[:1]

    def _load_system(self, system_name) -> Atoms:
        return ase_read(
            self.data_input_dir / self.name / self._get_pdb_file_name(system_name)
        )

    @staticmethod
    def _get_pdb_file_name(system_name: str) -> str:
        return f"{system_name}_eq.pdb"
