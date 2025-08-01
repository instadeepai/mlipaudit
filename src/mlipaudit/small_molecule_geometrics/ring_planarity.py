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
import json
import logging

import numpy as np
from ase import Atoms, units
from mlip.inference import run_batched_inference
from pydantic import BaseModel, TypeAdapter
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from mlip.simulation.jax_md import JaxMDSimulationEngine
from mlip.simulation import SimulationState

from mlipaudit.benchmark import Benchmark, BenchmarkResult, ModelOutput

logger = logging.getLogger("mlipaudit")

RING_PLANARITY_DATASET = "small_molecule_qm9.xyz"


# simulation_config:
# num_steps: 1_000_000
# snapshot_interval: 1_000
# num_episodes: 1_000
# temperature_kelvin: 300.0
#
# simulation_config_debug:
# num_steps: 10
# snapshot_interval: 1
# num_episodes: 1
# temperature_kelvin: 300.0


class Molecule(BaseModel):
    """Molecule class.

    Attributes:
        pattern_atoms: The indices of the atoms belonging
            to the ring.
    """
    molecule_name: str
    atom_symbols: list[str]
    coordinates: list[tuple[float, float, float]]
    smiles: str
    pattern_atoms: list[int]
    charge: float

Molecules = TypeAdapter(dict[str, Molecule])

class RingPlanarityResult(BenchmarkResult):
    pass

class MoleculeSimulationOutput(BaseModel):
    """Stores the simulation state for a molecule.

    Attributes:
        molecule_name: The name of the molecule.
        simulation_state: The simulation state.
    """
    molecule_name: str
    simulation_state: SimulationState

class RingPlanarityModelOutput(ModelOutput):
    molecule_simulations: list[MoleculeSimulationOutput]

class RingPlanarityBenchmark(Benchmark):
    """Benchmark for small organic molecule ring planarity."""

    name = "ring_planarity"
    result_class = RingPlanarityResult

    def run_model(self) -> None:

        molecule_outputs = []

        md_config = JaxMDSimulationEngine.Config(
            num_steps=1_000_000 if not self.fast_dev_run else 10,
            snapshot_interval=1_000 if not self.fast_dev_run else 1,
            num_episodes=1_000 if not self.fast_dev_run else 1,
            temperature_kelvin=300.0
        )

        for molecule_name, molecule in self._qm9_structures:
            logger.info("Running MD for %s", molecule_name)

            atoms = Atoms(
                symbols=molecule.atom_symbols,
                positions=molecule.coordinates,
            )
            md_engine = JaxMDSimulationEngine(
                atoms=atoms,
                force_field=self.force_field,
                config=md_config
            )
            md_engine.run()

            molecule_output = MoleculeSimulationOutput(
                molecule_name=molecule_name,
                simulation_state=md_engine.state
            )
            molecule_outputs.append(molecule_output)

        self.model_output = RingPlanarityModelOutput(molecule_simulations=molecule_outputs)

    def analyze(self) -> RingPlanarityResult:
        raise NotImplementedError


    @functools.cached_property
    def _qm9_structures(self) -> dict[str, Molecule]:
        with open(
                self.data_input_dir / self.name / RING_PLANARITY_DATASET,
                "r",
                encoding="utf-8",
        ) as f:
            dataset = Molecules.validate_json(f.read())

        if self.fast_dev_run:
            dataset = {"benzene": dataset["benzene"], "furan": dataset["furan"]}

        return dataset
