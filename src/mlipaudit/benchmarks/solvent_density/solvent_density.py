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

from mlip.simulation import SimulationState
from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from mlipaudit.benchmark import (
    Benchmark,
    BenchmarkResult,
    ModelOutput,
)
from mlipaudit.utils.molecular_liquids import (
    SOLVENT_DATA_NAME,
    SOLVENT_MOLECULE_CONFIG,
    SOLVENT_REFERENCE_DENSITIES,
    SOLVENT_REUSABLE_OUTPUT_ID,
    average_equilibrated_density,
    compute_densities,
    run_solvent_npt_simulations,
    score_density,
)
from mlipaudit.utils.stability import is_simulation_stable

logger = logging.getLogger("mlipaudit")


class SolventDensityModelOutput(ModelOutput):
    """Model output containing the final simulation states for each structure.

    Shares its field signature with `SolventRadialDistributionModelOutput` so that the
    NPT simulations can be reused between the two benchmarks (see `reusable_output_id`).

    Attributes:
        structure_names: The names of the structures.
        simulation_states: `SimulationState` or `None` object for each structure in
            the same order as the structure names. `None` if the simulation failed.
    """

    structure_names: list[str]
    simulation_states: list[SimulationState | None]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SolventDensityStructureResult(BaseModel):
    """Stores the density result for a single structure.

    Attributes:
        structure_name: The structure name.
        densities: List of per-frame densities in g/cm3.
        average_density: Average density over the final four fifths of the frames.
        density_deviation: Absolute deviation of the average density from the reference.
        reference_density: The reference density in g/cm3.
        failed: Whether the simulation was successful. If unsuccessful, the other
            attributes will be not be set.
        score: The score for the molecule.
    """

    structure_name: str
    densities: list[float] | None = None
    average_density: float | None = None
    density_deviation: NonNegativeFloat | None = None
    reference_density: float | None = None

    failed: bool = False
    score: float = 0.0


class SolventDensityResult(BenchmarkResult):
    """Result object for the solvent density benchmark.

    Attributes:
        structure_names: The names of the structures.
        structures: List of per structure results.
        avg_density_deviation: The average density deviation across all structures.
        failed: Whether all the simulations failed and no analysis could be
            performed. Defaults to False.
        score: The final score for the benchmark between 0 and 1.
    """

    structure_names: list[str]
    structures: list[SolventDensityStructureResult]
    avg_density_deviation: NonNegativeFloat | None = None


class SolventDensityBenchmark(Benchmark):
    """Benchmark for the equilibrium density of molecular solvents.

    Runs the same NPT simulations as `SolventRadialDistributionBenchmark` (reusing
    their output when both are run together) and scores how closely the equilibrium
    density of each solvent matches its reference.

    Attributes:
        name: The unique benchmark name. The name is `solvent_density`.
        category: The benchmark's category, `"Molecular Liquids"`.
        data_name: The input-data directory shared with the solvent RDF benchmark.
        result_class: `SolventDensityResult`.
        model_output_class: `SolventDensityModelOutput`.
        required_elements: The set of atomic element types present in the input files.
        reusable_output_id: Shared with `SolventRadialDistributionBenchmark` so the
            solvent NPT simulations are only run once when both benchmarks are run
            together.
    """

    name = "solvent_density"
    category = "Molecular Liquids"
    data_name = SOLVENT_DATA_NAME
    result_class = SolventDensityResult
    model_output_class = SolventDensityModelOutput

    required_elements = {"N", "H", "O", "C", "Cl"}

    reusable_output_id = SOLVENT_REUSABLE_OUTPUT_ID

    def run_model(self) -> None:
        """Run an MD simulation for each structure using the NPT ensemble.

        The MD simulation is performed using the JAX MD engine and starts from
        the reference structure. The NPT integrator uses Langevin dynamics with
        a Monte Carlo barostat.
        """
        structure_names, simulation_states = run_solvent_npt_simulations(
            self.force_field, self.data_dir, self.run_mode
        )
        self.model_output = SolventDensityModelOutput(
            structure_names=structure_names, simulation_states=simulation_states
        )

    def analyze(self) -> SolventDensityResult:
        """Compute the equilibrium density of each solvent and its deviation.

        Returns:
            A `SolventDensityResult` object.

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
                structure_results.append(
                    SolventDensityStructureResult(
                        structure_name=system_name,
                        failed=True,
                        score=0.0,
                    )
                )
                continue

            num_succeeded += 1
            mol_config = SOLVENT_MOLECULE_CONFIG[system_name]
            densities = compute_densities(
                simulation_state,
                mol_config["molecule_weight"],
                int(mol_config["atoms_per_molecule"]),
            )
            average_density = average_equilibrated_density(densities)
            reference_density = SOLVENT_REFERENCE_DENSITIES[system_name]
            density_deviation, score = score_density(average_density, reference_density)

            structure_results.append(
                SolventDensityStructureResult(
                    structure_name=system_name,
                    densities=densities.tolist(),
                    average_density=average_density,
                    density_deviation=density_deviation,
                    reference_density=reference_density,
                    score=score,
                )
            )

        if num_succeeded == 0:
            return SolventDensityResult(
                structure_names=self.model_output.structure_names,
                structures=structure_results,
                failed=True,
                score=0.0,
            )

        return SolventDensityResult(
            structure_names=self.model_output.structure_names,
            structures=structure_results,
            avg_density_deviation=statistics.mean(
                structure.density_deviation
                for structure in structure_results
                if structure.density_deviation is not None
            ),
            score=statistics.mean(r.score for r in structure_results),
        )
